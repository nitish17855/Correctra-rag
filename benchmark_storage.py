import os
import json
import time
import statistics
from datetime import datetime, timezone

from langchain_core.documents import Document
from Ingestor.html_loader import load_html
from chunker.recursive_character_chunker import recursive_splitters
from Embedding.bge_embeddings import BGEEmbedding
from storage.Qdrant import QdrantStore
from storage.BM25 import BM25Store


SEC_10K_URL = "https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/aapl-20250927.htm"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CACHE_FILE = os.path.join(DATA_DIR, "aapl_clean_text.txt")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
BENCHMARK_JSON_PATH = os.path.join(RESULTS_DIR, "storage_benchmark.json")
ROOT_BENCHMARK_JSON = os.path.join(os.path.dirname(__file__), "benchmark.json")

BENCHMARK_QUERIES = [
    "What were Apple's total net sales and revenue in fiscal year 2025?",
    "What are the principal risk factors associated with international supply chain and manufacturing in China?",
    "Describe the gross margin performance and operating expenses for Products and Services.",
    "What are Apple's outstanding legal proceedings, antitrust investigations, and regulatory matters?",
    "What is the total cash, cash equivalents, and marketable securities balance at the end of the period?",
]


def get_aapl_10k_text() -> tuple[str, float]:
    """Loads clean 10-K text from local cache or fetches via load_html."""
    os.makedirs(DATA_DIR, exist_ok=True)
    t0 = time.perf_counter()

    if os.path.exists(CACHE_FILE):
        print(f"[*] Reading 10-K filing from local cache: {CACHE_FILE}", flush=True)
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            clean_text = f.read()
        duration = time.perf_counter() - t0
        print(f"[*] Loaded {len(clean_text):,} characters in {duration:.3f}s", flush=True)
        return clean_text, duration

    print(f"[*] Fetching 10-K filing from SEC EDGAR via load_html...", flush=True)
    doc_data = load_html(SEC_10K_URL)
    clean_text = doc_data["page_content"]
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        f.write(clean_text)
    duration = time.perf_counter() - t0
    print(f"[*] Fetched and cached {len(clean_text):,} characters in {duration:.2f}s", flush=True)
    return clean_text, duration


def calculate_latency_stats(latencies_ms: list[float]) -> dict:
    sorted_lats = sorted(latencies_ms)
    p50_idx = int(0.50 * (len(sorted_lats) - 1))
    p95_idx = int(0.95 * (len(sorted_lats) - 1))
    return {
        "mean_ms": round(statistics.mean(latencies_ms), 3),
        "min_ms": round(min(latencies_ms), 3),
        "max_ms": round(max(latencies_ms), 3),
        "p50_ms": round(sorted_lats[p50_idx], 3),
        "p95_ms": round(sorted_lats[p95_idx], 3),
    }


def run_benchmark():
    print("=" * 80, flush=True)
    print("STARTING FINANCIAL RAG STORAGE BENCHMARK PIPELINE", flush=True)
    print(f"Target Document: Apple Inc. Form 10-K (FY2025)", flush=True)
    print("=" * 80, flush=True)

    # 1. Ingestion / Cache
    clean_text, fetch_sec = get_aapl_10k_text()

    # 2. Chunking
    t_chunk_start = time.perf_counter()
    chunk_size = 1000
    chunk_overlap = 100
    documents = recursive_splitters(clean_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    for doc in documents:
        doc.metadata["source"] = "aapl-20250927.htm"
        doc.metadata["ticker"] = "AAPL"
        doc.metadata["filing_type"] = "10-K"
        doc.metadata["fiscal_year"] = "2025"

    chunk_sec = time.perf_counter() - t_chunk_start
    total_chunks = len(documents)
    avg_chars = round(statistics.mean([len(d.page_content) for d in documents]), 1)
    print(f"[1/4] Chunked corpus into {total_chunks} chunks (avg {avg_chars} chars) in {chunk_sec:.3f}s", flush=True)

    # 3. Dense Embeddings Generation
    print(f"[2/4] Generating BGE embeddings (BAAI/bge-base-en-v1.5)...", flush=True)
    embedder = BGEEmbedding()
    t_embed_start = time.perf_counter()
    chunk_texts = [d.page_content for d in documents]
    vectors = embedder.embed_documents(chunk_texts)
    embed_sec = time.perf_counter() - t_embed_start
    print(f"      Encoded {total_chunks} vectors in {embed_sec:.2f}s ({total_chunks / embed_sec:.1f} chunks/sec)", flush=True)

    # 4. Storage Indexing
    print(f"[3/4] Indexing into Storage Stores...", flush=True)

    # A: Qdrant Dense (:memory:)
    t_q_start = time.perf_counter()
    qdrant_store = QdrantStore(
        collection_name="aapl_10k_benchmark",
        vector_size=embedder.dimension,
        location=":memory:",
    )
    qdrant_store.upsert_documents(documents=documents, vectors=vectors)
    qdrant_sec = time.perf_counter() - t_q_start
    print(f"      -> Qdrant Store (In-Memory) upserted {total_chunks} vectors in {qdrant_sec:.3f}s", flush=True)

    # B: BM25 Sparse (In-Memory)
    t_b_start = time.perf_counter()
    bm25_store = BM25Store()
    bm25_store.index_documents(documents=documents)
    bm25_sec = time.perf_counter() - t_b_start
    print(f"      -> BM25 Store (In-Memory) indexed {total_chunks} documents in {bm25_sec:.3f}s", flush=True)

    # 5. Query Latency & Retrieval Benchmarks
    print(f"[4/4] Executing Query Latency Benchmarks (Dense vs Sparse)...", flush=True)
    iterations_per_query = 5
    dense_latencies = []
    sparse_latencies = []
    sample_retrievals = []

    for q_idx, query in enumerate(BENCHMARK_QUERIES):
        print(f"      Benchmarking Query {q_idx + 1}/{len(BENCHMARK_QUERIES)}...", flush=True)
        query_vector = embedder.embed_query(query)

        # Dense Search Latency (Qdrant)
        for _ in range(iterations_per_query):
            t0 = time.perf_counter()
            dense_res = qdrant_store.search(query_vector=query_vector, top_k=3)
            dense_latencies.append((time.perf_counter() - t0) * 1000)

        # Sparse Search Latency (BM25)
        for _ in range(iterations_per_query):
            t0 = time.perf_counter()
            sparse_res = bm25_store.search(query=query, top_k=3)
            sparse_latencies.append((time.perf_counter() - t0) * 1000)

        # Record Sample Retrievals
        dense_top = dense_res[0] if dense_res else None
        sparse_top = sparse_res[0] if sparse_res else None

        sample_retrievals.append({
            "query": query,
            "dense_top_1": {
                "chunk_id": dense_top.metadata.get("chunk_id") if dense_top else None,
                "score": round(dense_top.metadata.get("score", 0.0), 4) if dense_top else None,
                "content_preview": dense_top.page_content[:180].replace("\n", " ") + "..." if dense_top else None,
            },
            "sparse_top_1": {
                "chunk_id": sparse_top.metadata.get("chunk_id") if sparse_top else None,
                "score": round(sparse_top.metadata.get("score", 0.0), 4) if sparse_top else None,
                "content_preview": sparse_top.page_content[:180].replace("\n", " ") + "..." if sparse_top else None,
            },
        })

    dense_stats = calculate_latency_stats(dense_latencies)
    sparse_stats = calculate_latency_stats(sparse_latencies)

    print("\n" + "=" * 80, flush=True)
    print("BENCHMARK RESULTS SUMMARY:", flush=True)
    print(f"Dense (Qdrant :memory:) Latency: p50={dense_stats['p50_ms']}ms | p95={dense_stats['p95_ms']}ms | mean={dense_stats['mean_ms']}ms", flush=True)
    print(f"Sparse (BM25Okapi) Latency:      p50={sparse_stats['p50_ms']}ms | p95={sparse_stats['p95_ms']}ms | mean={sparse_stats['mean_ms']}ms", flush=True)
    print("=" * 80, flush=True)

    # 6. Save Structured Benchmark Record
    os.makedirs(RESULTS_DIR, exist_ok=True)
    existing_records = []
    if os.path.exists(BENCHMARK_JSON_PATH):
        try:
            with open(BENCHMARK_JSON_PATH, "r", encoding="utf-8") as f:
                existing_records = json.load(f)
        except Exception:
            existing_records = []

    test_num = len(existing_records) + 1
    test_id = f"TEST_{test_num:03d}_STORAGE_BASELINE"

    record = {
        "test_run_header": {
            "test_id": test_id,
            "run_index": test_num,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "target_document": "Apple Inc. Form 10-K (FY 2025)",
            "benchmark_objective": "Dense (Qdrant :memory:) vs Sparse (BM25Okapi) latency & retrieval on 10-K corpus",
        },
        "system_architecture_snapshot": {
            "pipeline_stage": "Stage 4: Hybrid Storage Layer (Dense + Sparse In-Memory)",
            "qdrant": {
                "status": "IMPLEMENTED",
                "storage_mode": "in_memory",
                "location": ":memory:",
                "collection_name": "aapl_10k_benchmark",
                "vector_dimension": embedder.dimension,
                "distance_metric": "COSINE",
                "normalized_embeddings": True,
            },
            "bm25": {
                "status": "IMPLEMENTED",
                "storage_mode": "in_memory",
                "algorithm": "BM25Okapi",
                "tokenizer": "regex_alphanumeric_lowercased",
            },
            "postgresql_metadata_filter": {
                "status": "NOT_YET_BUILT",
                "storage_mode": "pending_future_milestone",
            },
            "hybrid_fusion": {
                "status": "NOT_YET_BUILT",
                "fusion_type": "pending_future_milestone",
            },
            "embedding_provider": {
                "model_name": embedder.model_name,
                "dimension": embedder.dimension,
                "batch_size": 32,
                "normalized": True,
            },
            "chunking": {
                "strategy": "recursive_character_chunker",
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
            },
        },
        "corpus_and_indexing_metrics": {
            "source_url": SEC_10K_URL,
            "clean_text_chars": len(clean_text),
            "total_chunks_indexed": total_chunks,
            "avg_chunk_length_chars": avg_chars,
            "timings_seconds": {
                "fetch_and_load_sec": round(fetch_sec, 3),
                "chunking_sec": round(chunk_sec, 3),
                "embedding_generation_sec": round(embed_sec, 3),
                "qdrant_upsert_sec": round(qdrant_sec, 3),
                "bm25_index_sec": round(bm25_sec, 3),
                "total_end_to_end_indexing_sec": round(fetch_sec + chunk_sec + embed_sec + qdrant_sec + bm25_sec, 3),
            },
        },
        "query_latency_benchmarks": {
            "iterations_per_query": iterations_per_query,
            "total_query_runs": len(dense_latencies),
            "dense_qdrant_latency": dense_stats,
            "sparse_bm25_latency": sparse_stats,
        },
        "sample_retrieval_comparison": sample_retrievals,
    }

    existing_records.append(record)
    with open(BENCHMARK_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(existing_records, f, indent=2)

    with open(ROOT_BENCHMARK_JSON, "w", encoding="utf-8") as f:
        json.dump(existing_records, f, indent=2)

    print(f"\n[+] Saved benchmark run to {BENCHMARK_JSON_PATH} and {ROOT_BENCHMARK_JSON}", flush=True)
    print(f"[+] Assigned Test ID: {test_id} (Total runs recorded: {len(existing_records)})\n", flush=True)


if __name__ == "__main__":
    run_benchmark()
