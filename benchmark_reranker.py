import os
import json
import time
import math
import statistics
from datetime import datetime, timezone
import numpy as np

from langchain_core.documents import Document
from chunker.recursive_character_chunker import recursive_splitters
from Embedding.bge_embeddings import BGEEmbedding
from storage.Qdrant import QdrantStore
from storage.BM25 import BM25Store
from fusion.rrf import reciprocal_rank_fusion
from reranker.base import get_reranker


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CACHE_FILE = os.path.join(DATA_DIR, "aapl_clean_text.txt")
VECTORS_CACHE_FILE = os.path.join(DATA_DIR, "aapl_bge_vectors.npy")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
BENCHMARK_JSON_PATH = os.path.join(RESULTS_DIR, "storage_benchmark.json")
ROOT_BENCHMARK_JSON = os.path.join(os.path.dirname(__file__), "benchmark.json")

# 20 Evaluated Financial Queries with Ground-Truth Fact Matchers for Graded Relevance (0, 1, 2)
# Score 2: Primary definitive answer / financial table / exact required metric
# Score 1: Partial context / discussion of the specific topic
BENCHMARK_DATASET_20 = [
    {
        "query": "What were Apple's total net sales in fiscal year 2025 and 2024?",
        "score_2_terms": ["391,035", "383,285"],
        "score_1_terms": ["total net sales", "consolidated statements of operations"],
    },
    {
        "query": "What was the total revenue generated specifically by the iPhone product line in 2025?",
        "score_2_terms": ["201,183"],
        "score_1_terms": ["iphone net sales", "higher net sales of iphone"],
    },
    {
        "query": "How did Services gross margin compare with Products gross margin in fiscal 2025?",
        "score_2_terms": ["74.2%", "36.8%"],
        "score_1_terms": ["services gross margin", "products gross margin"],
    },
    {
        "query": "What is Apple's total cash, cash equivalents, and marketable securities position?",
        "score_2_terms": ["marketable securities", "cash and cash equivalents", "29,942"],
        "score_1_terms": ["cash equivalents", "term debt"],
    },
    {
        "query": "What are the principal supply chain and manufacturing risks located in China?",
        "score_2_terms": ["substantially all", "concentrated in china", "outsourced manufacturing"],
        "score_1_terms": ["china", "manufacturing partners", "supply chain"],
    },
    {
        "query": "What legal proceedings and antitrust investigations is Apple facing under the EU Digital Markets Act?",
        "score_2_terms": ["digital markets act", "european commission", "non-compliance"],
        "score_1_terms": ["antitrust", "app store", "european union"],
    },
    {
        "query": "What were Apple's research and development (R&D) expenses in 2025 and 2024?",
        "score_2_terms": ["34,550", "31,370"],
        "score_1_terms": ["research and development expense", "r&d"],
    },
    {
        "query": "Describe the terms and balance of Apple's outstanding commercial paper and term debt.",
        "score_2_terms": ["commercial paper", "floating-rate", "term debt maturities"],
        "score_1_terms": ["commercial paper", "senior notes"],
    },
    {
        "query": "How much capital did Apple return to shareholders through share repurchases and cash dividends in 2025?",
        "score_2_terms": ["repurchases of common stock", "dividends paid", "share repurchases"],
        "score_1_terms": ["share repurchase program", "cash dividends"],
    },
    {
        "query": "What was Apple's effective tax rate for fiscal year 2025?",
        "score_2_terms": ["effective tax rate", "15.4%"],
        "score_1_terms": ["provision for income taxes", "foreign earnings"],
    },
    {
        "query": "What were the net sales for the Mac and iPad product categories in 2025?",
        "score_2_terms": ["31,659", "28,300"],
        "score_1_terms": ["mac net sales", "ipad net sales"],
    },
    {
        "query": "Describe the performance and growth drivers of the Wearables, Home and Accessories segment.",
        "score_2_terms": ["37,005", "wearables, home and accessories"],
        "score_1_terms": ["apple watch", "airpods"],
    },
    {
        "query": "What are the key foreign exchange and currency volatility risks affecting international revenue?",
        "score_2_terms": ["foreign exchange rate", "strengthening of the u.s. dollar"],
        "score_1_terms": ["currency volatility", "derivative instruments", "hedging"],
    },
    {
        "query": "What are Apple's operating lease obligations and property lease commitments?",
        "score_2_terms": ["operating lease liabilities", "undiscounted cash flows"],
        "score_1_terms": ["operating leases", "lease commitments"],
    },
    {
        "query": "How does Apple manage semiconductor chip procurement and third-party foundry risks?",
        "score_2_terms": ["single-source", "foundries", "custom silicon"],
        "score_1_terms": ["semiconductors", "components supply", "third-party manufacturers"],
    },
    {
        "query": "What was Apple's total operating income and operating margin in 2025?",
        "score_2_terms": ["123,216", "operating income"],
        "score_1_terms": ["operating margin", "operating expenses"],
    },
    {
        "query": "What are the significant accounting policies regarding Services revenue recognition and deferred revenue?",
        "score_2_terms": ["deferred revenue", "performance obligations", "recognition"],
        "score_1_terms": ["services revenue", "advertising", "app store"],
    },
    {
        "query": "Who are Apple's executive officers and what are their designated roles?",
        "score_2_terms": ["timothy d. cook", "luca maestri", "jeffrey e. williams"],
        "score_1_terms": ["executive officers", "chief executive officer"],
    },
    {
        "query": "What environmental regulations, carbon neutrality goals, and sustainability initiatives are disclosed?",
        "score_2_terms": ["carbon neutral", "apple 2030", "greenhouse gas emissions"],
        "score_1_terms": ["sustainability", "environmental protection", "renewable energy"],
    },
    {
        "query": "What is Apple's total common shares outstanding and basic vs diluted earnings per share (EPS)?",
        "score_2_terms": ["6.11", "6.08"],
        "score_1_terms": ["basic earnings per share", "diluted earnings per share"],
    },
]


def load_or_compute_embeddings(embedder: BGEEmbedding, chunk_texts: list[str]) -> list[list[float]]:
    if os.path.exists(VECTORS_CACHE_FILE):
        print(f"[*] Loading precomputed embeddings from cache: {VECTORS_CACHE_FILE}", flush=True)
        cached_vectors = np.load(VECTORS_CACHE_FILE)
        if len(cached_vectors) == len(chunk_texts):
            return cached_vectors.tolist()

    print(f"[*] Encoding {len(chunk_texts)} chunks with BGE...", flush=True)
    vectors = embedder.embed_documents(chunk_texts)
    np.save(VECTORS_CACHE_FILE, np.array(vectors, dtype=np.float32))
    return vectors


def grade_relevance(doc_text: str, test_item: dict) -> int:
    text_lower = doc_text.lower()
    # Check for Score 2 (Definitive / Primary answer)
    score_2_hits = sum(1 for term in test_item["score_2_terms"] if term.lower() in text_lower)
    if score_2_hits >= 1 and len(test_item["score_2_terms"]) == 1:
        return 2
    if score_2_hits >= min(2, len(test_item["score_2_terms"])):
        return 2

    # Check for Score 1 (Partial context)
    score_1_hits = sum(1 for term in test_item["score_1_terms"] if term.lower() in text_lower)
    if score_1_hits >= 1 or score_2_hits >= 1:
        return 1

    return 0


def compute_dcg(relevances: list[int], k: int = 5) -> float:
    dcg = 0.0
    for idx, rel in enumerate(relevances[:k], start=1):
        if rel > 0:
            dcg += (2**rel - 1) / math.log2(idx + 1)
    return dcg


def compute_ndcg(retrieved_relevances: list[int], all_candidate_relevances: list[int], k: int = 5) -> float:
    dcg = compute_dcg(retrieved_relevances, k=k)
    ideal_relevances = sorted(all_candidate_relevances, reverse=True)[:k]
    idcg = compute_dcg(ideal_relevances, k=k)
    if idcg == 0.0:
        return 0.0
    return min(1.0, dcg / idcg)


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


def run_reranking_benchmark():
    print("=" * 90, flush=True)
    print("TEST_003_RERANKING_BENCHMARK: DENSE vs HYBRID (RRF) vs HYBRID + CROSS-ENCODER RERANKER", flush=True)
    print(f"Corpus: Apple Inc. Form 10-K (FY2025) | Queries: {len(BENCHMARK_DATASET_20)} | Metric: NDCG@5", flush=True)
    print("=" * 90, flush=True)

    # 1. Load and prepare corpus
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        clean_text = f.read()

    documents = recursive_splitters(clean_text, chunk_size=1000, chunk_overlap=100)
    for idx, doc in enumerate(documents):
        doc.metadata["chunk_id"] = f"chunk_{idx:03d}"
        doc.metadata["source"] = "aapl-20250927.htm"
        doc.metadata["ticker"] = "AAPL"
        doc.metadata["fiscal_year"] = "2025"

    total_chunks = len(documents)
    print(f"[+] Corpus chunked into {total_chunks} chunks.", flush=True)

    # 2. Embedding & Storage
    embedder = BGEEmbedding()
    chunk_texts = [d.page_content for d in documents]
    vectors = load_or_compute_embeddings(embedder, chunk_texts)

    qdrant_store = QdrantStore(collection_name="aapl_rerank_bench", vector_size=embedder.dimension, location=":memory:")
    qdrant_store.upsert_documents(documents=documents, vectors=vectors)

    bm25_store = BM25Store()
    bm25_store.index_documents(documents=documents)

    # 3. Local Cross-Encoder Reranker via Provider Factory
    print("[*] Initializing Cross-Encoder Reranker via RerankerProvider factory...", flush=True)
    reranker = get_reranker("local")
    print("[+] All components initialized successfully.\n", flush=True)

    dense_ndcgs = []
    hybrid_ndcgs = []
    rerank_ndcgs = []

    dense_lats = []
    hybrid_lats = []
    rerank_lats = []

    query_records = []

    print("-" * 92, flush=True)
    print(f"{'Q#':<3} | {'Dense NDCG':<11} | {'Hybrid NDCG':<11} | {'Rerank NDCG':<11} | {'Reranker Gain':<13} | Query", flush=True)
    print("-" * 92, flush=True)

    for idx, test_item in enumerate(BENCHMARK_DATASET_20, start=1):
        query = test_item["query"]

        # -----------------------------
        # Pipeline 1: Dense Only (Qdrant top 5)
        # -----------------------------
        t0 = time.perf_counter()
        q_vector = embedder.embed_query(query)
        dense_results = qdrant_store.search(query_vector=q_vector, top_k=5)
        dense_time = (time.perf_counter() - t0) * 1000
        dense_lats.append(dense_time)

        # -----------------------------
        # Pipeline 2: Hybrid Search (RRF top 5)
        # -----------------------------
        t0 = time.perf_counter()
        # Retrieve candidate pool of 20 from both dense & sparse
        pool_dense = qdrant_store.search(query_vector=q_vector, top_k=20)
        pool_sparse = bm25_store.search(query=query, top_k=20)
        hybrid_top5 = reciprocal_rank_fusion(runs=[pool_dense, pool_sparse], k=60, top_k=5)
        hybrid_time = (time.perf_counter() - t0) * 1000
        hybrid_lats.append(hybrid_time)

        # -----------------------------
        # Pipeline 3: Hybrid + Cross-Encoder Reranker (RRF top 20 -> Reranker top 5)
        # -----------------------------
        t0 = time.perf_counter()
        hybrid_pool20 = reciprocal_rank_fusion(runs=[pool_dense, pool_sparse], k=60, top_k=20)
        reranked_top5 = reranker.rerank(query=query, documents=hybrid_pool20, top_k=5)
        rerank_time = (time.perf_counter() - t0) * 1000
        rerank_lats.append(rerank_time)

        # -----------------------------
        # Ground Truth & NDCG Calculation
        # -----------------------------
        # Aggregate all unique candidate chunks evaluated for this query
        all_eval_docs = {d.metadata.get("chunk_id", d.page_content[:40]): d for d in (dense_results + hybrid_top5 + reranked_top5 + hybrid_pool20)}.values()
        all_candidate_rels = [grade_relevance(d.page_content, test_item) for d in all_eval_docs]

        dense_rels = [grade_relevance(d.page_content, test_item) for d in dense_results]
        hybrid_rels = [grade_relevance(d.page_content, test_item) for d in hybrid_top5]
        rerank_rels = [grade_relevance(d.page_content, test_item) for d in reranked_top5]

        dense_ndcg = compute_ndcg(dense_rels, all_candidate_rels, k=5)
        hybrid_ndcg = compute_ndcg(hybrid_rels, all_candidate_rels, k=5)
        rerank_ndcg = compute_ndcg(rerank_rels, all_candidate_rels, k=5)

        dense_ndcgs.append(dense_ndcg)
        hybrid_ndcgs.append(hybrid_ndcg)
        rerank_ndcgs.append(rerank_ndcg)

        gain_diff = rerank_ndcg - hybrid_ndcg
        if gain_diff > 0.001:
            gain_str = f"+{gain_diff:.3f} (WIN)"
        elif gain_diff < -0.001:
            gain_str = f"{gain_diff:.3f} (LOSS)"
        else:
            gain_str = "= 0.000 (TIE)"

        print(f"{idx:<3} | {dense_ndcg:<11.4f} | {hybrid_ndcg:<11.4f} | {rerank_ndcg:<11.4f} | {gain_str:<13} | {query[:42]}...", flush=True)

        query_records.append({
            "query_index": idx,
            "query": query,
            "ndcg_at_5": {
                "dense_only": round(dense_ndcg, 4),
                "hybrid_rrf": round(hybrid_ndcg, 4),
                "hybrid_plus_cross_encoder": round(rerank_ndcg, 4),
            },
            "retrieved_relevances": {
                "dense_only": dense_rels,
                "hybrid_rrf": hybrid_rels,
                "hybrid_plus_cross_encoder": rerank_rels,
            },
            "top_1_chunk_id": {
                "dense": dense_results[0].metadata.get("chunk_id") if dense_results else None,
                "hybrid": hybrid_top5[0].metadata.get("chunk_id") if hybrid_top5 else None,
                "reranker": reranked_top5[0].metadata.get("chunk_id") if reranked_top5 else None,
            },
            "latencies_ms": {
                "dense_ms": round(dense_time, 2),
                "hybrid_ms": round(hybrid_time, 2),
                "reranker_total_ms": round(rerank_time, 2),
            }
        })

    # Summary Statistics
    mean_dense_ndcg = statistics.mean(dense_ndcgs)
    mean_hybrid_ndcg = statistics.mean(hybrid_ndcgs)
    mean_rerank_ndcg = statistics.mean(rerank_ndcgs)

    rerank_vs_dense_pct = ((mean_rerank_ndcg - mean_dense_ndcg) / mean_dense_ndcg * 100) if mean_dense_ndcg > 0 else 0.0
    rerank_vs_hybrid_pct = ((mean_rerank_ndcg - mean_hybrid_ndcg) / mean_hybrid_ndcg * 100) if mean_hybrid_ndcg > 0 else 0.0

    dense_lat_stats = calculate_latency_stats(dense_lats)
    hybrid_lat_stats = calculate_latency_stats(hybrid_lats)
    rerank_lat_stats = calculate_latency_stats(rerank_lats)

    print("\n" + "=" * 90, flush=True)
    print("BENCHMARK SUMMARY & COMPARISON", flush=True)
    print("=" * 90, flush=True)
    print(f"Mean NDCG@5: Dense-only            : {mean_dense_ndcg:.4f}", flush=True)
    print(f"Mean NDCG@5: Hybrid (RRF)          : {mean_hybrid_ndcg:.4f}", flush=True)
    print(f"Mean NDCG@5: Hybrid + Cross-Encoder: {mean_rerank_ndcg:.4f}", flush=True)
    print(f"Relative Gain: Reranker vs Dense   : {rerank_vs_dense_pct:+.2f}%", flush=True)
    print(f"Relative Gain: Reranker vs Hybrid  : {rerank_vs_hybrid_pct:+.2f}%", flush=True)
    print(f"\nLatency Profile:")
    print(f" Dense-only (embed + qdrant)       : p50={dense_lat_stats['p50_ms']}ms, p95={dense_lat_stats['p95_ms']}ms")
    print(f" Hybrid RRF (dense+bm25+rrf)       : p50={hybrid_lat_stats['p50_ms']}ms, p95={hybrid_lat_stats['p95_ms']}ms")
    print(f" Hybrid + Reranker (end-to-end)    : p50={rerank_lat_stats['p50_ms']}ms, p95={rerank_lat_stats['p95_ms']}ms")
    print("=" * 90, flush=True)

    # 4. Save to JSON Benchmarks
    benchmark_payload = {
        "test_run_header": {
            "test_id": "TEST_003_RERANKING_BENCHMARK",
            "run_index": 3,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "target_document": "Apple Inc. Form 10-K (FY 2025)",
            "benchmark_objective": "Measure NDCG@5: Dense-only vs Hybrid (RRF) vs Hybrid + Cross-Encoder Reranker on 20 queries"
        },
        "system_architecture_snapshot": {
            "pipeline_stage": "Stage 6: Two-Stage Funnel (Hybrid RRF -> Cross-Encoder Reranker)",
            "reranker": {
                "provider_abc": "RerankerProvider",
                "model_name": "cross-encoder/ms-marco-MiniLM-L6-v2",
                "architecture": "Joint Cross-Attention Transformer",
                "candidate_funnel": "RRF Top-20 -> Cross-Encoder Top-5"
            },
            "hybrid_fusion": {
                "algorithm": "Reciprocal Rank Fusion (RRF)",
                "k_constant": 60,
                "input_candidate_pools": 20
            },
            "dense_retriever": {
                "model": "BAAI/bge-base-en-v1.5",
                "storage": "Qdrant (:memory:)",
                "dimension": 768
            },
            "sparse_retriever": {
                "algorithm": "BM25Okapi",
                "tokenizer": "regex_alphanumeric_lowercased"
            }
        },
        "aggregate_ndcg_metrics": {
            "evaluated_queries_count": len(BENCHMARK_DATASET_20),
            "mean_ndcg_at_5": {
                "dense_only": round(mean_dense_ndcg, 4),
                "hybrid_rrf": round(mean_hybrid_ndcg, 4),
                "hybrid_plus_cross_encoder": round(mean_rerank_ndcg, 4),
            },
            "relative_gains_pct": {
                "reranker_vs_dense": round(rerank_vs_dense_pct, 2),
                "reranker_vs_hybrid": round(rerank_vs_hybrid_pct, 2),
            },
            "win_tie_loss_vs_hybrid": {
                "wins": sum(1 for r in query_records if r["ndcg_at_5"]["hybrid_plus_cross_encoder"] > r["ndcg_at_5"]["hybrid_rrf"] + 0.001),
                "ties": sum(1 for r in query_records if abs(r["ndcg_at_5"]["hybrid_plus_cross_encoder"] - r["ndcg_at_5"]["hybrid_rrf"]) <= 0.001),
                "losses": sum(1 for r in query_records if r["ndcg_at_5"]["hybrid_plus_cross_encoder"] < r["ndcg_at_5"]["hybrid_rrf"] - 0.001),
            }
        },
        "latency_profiles": {
            "dense_only": dense_lat_stats,
            "hybrid_rrf": hybrid_lat_stats,
            "hybrid_plus_cross_encoder": rerank_lat_stats,
        },
        "per_query_evaluations": query_records
    }

    # Append to results/storage_benchmark.json and benchmark.json
    for path in [BENCHMARK_JSON_PATH, ROOT_BENCHMARK_JSON]:
        existing_data = []
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
            except Exception:
                existing_data = []
        existing_data.append(benchmark_payload)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, indent=2)
        print(f"[+] Saved TEST_003_RERANKING_BENCHMARK to {path}", flush=True)


if __name__ == "__main__":
    run_reranking_benchmark()
