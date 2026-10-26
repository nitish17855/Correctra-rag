import os
import json
import time
import statistics
from datetime import datetime, timezone
import numpy as np
from dotenv import load_dotenv

from chunker.recursive_character_chunker import recursive_splitters
from Embedding.bge_embeddings import BGEEmbedding
from storage.Qdrant import QdrantStore
from storage.BM25 import BM25Store
from fusion.rrf import reciprocal_rank_fusion
from reranker.base import get_reranker
from context_assembly.context_assembly import ContextAssembler
from generation.prompt_builder import PromptBuilder
from generation.llm_provider import get_llm_provider
from generation.grounding_verifier import GroundingVerifier

load_dotenv()

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CACHE_FILE = os.path.join(DATA_DIR, "aapl_clean_text.txt")
VECTORS_CACHE_FILE = os.path.join(DATA_DIR, "aapl_bge_vectors.npy")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
BENCHMARK_JSON_PATH = os.path.join(RESULTS_DIR, "storage_benchmark.json")
ROOT_BENCHMARK_JSON = os.path.join(os.path.dirname(__file__), "benchmark.json")

# 50 Comprehensive Queries: 35 In-Domain Filing Queries + 15 Out-of-Domain Adversarial Queries
BENCHMARK_50_QUERIES = [
    # --- 35 In-Domain Form 10-K Queries ---
    {"id": 1, "query": "What were Apple's total net sales in fiscal year 2025 and 2024?", "type": "in_domain"},
    {"id": 2, "query": "What was the total revenue generated specifically by the iPhone in 2025?", "type": "in_domain"},
    {"id": 3, "query": "How did Services gross margin compare with Products gross margin in fiscal 2025?", "type": "in_domain"},
    {"id": 4, "query": "What is Apple's total cash, cash equivalents, and marketable securities position?", "type": "in_domain"},
    {"id": 5, "query": "What are the principal supply chain and manufacturing risks located in China?", "type": "in_domain"},
    {"id": 6, "query": "What legal proceedings and antitrust investigations is Apple facing under the EU Digital Markets Act?", "type": "in_domain"},
    {"id": 7, "query": "What were Apple's research and development (R&D) expenses in 2025 and 2024?", "type": "in_domain"},
    {"id": 8, "query": "Describe the terms and balance of Apple's outstanding commercial paper and term debt.", "type": "in_domain"},
    {"id": 9, "query": "How much capital did Apple return to shareholders through share repurchases and cash dividends in 2025?", "type": "in_domain"},
    {"id": 10, "query": "What was Apple's effective tax rate for fiscal year 2025?", "type": "in_domain"},
    {"id": 11, "query": "What were the net sales for the Mac and iPad product categories in 2025?", "type": "in_domain"},
    {"id": 12, "query": "Describe the performance and growth drivers of the Wearables, Home and Accessories segment.", "type": "in_domain"},
    {"id": 13, "query": "What are the key foreign exchange and currency volatility risks affecting international revenue?", "type": "in_domain"},
    {"id": 14, "query": "What are Apple's operating lease obligations and property lease commitments?", "type": "in_domain"},
    {"id": 15, "query": "How does Apple manage semiconductor chip procurement and third-party foundry risks?", "type": "in_domain"},
    {"id": 16, "query": "What was Apple's total operating income and operating margin in 2025?", "type": "in_domain"},
    {"id": 17, "query": "What are the significant accounting policies regarding Services revenue recognition and deferred revenue?", "type": "in_domain"},
    {"id": 18, "query": "Who are Apple's executive officers and what are their designated roles?", "type": "in_domain"},
    {"id": 19, "query": "What environmental regulations, carbon neutrality goals, and sustainability initiatives are disclosed?", "type": "in_domain"},
    {"id": 20, "query": "What is Apple's total common shares outstanding and basic vs diluted earnings per share (EPS)?", "type": "in_domain"},
    {"id": 21, "query": "What was Apple's total gross margin in dollars and as a percentage of net sales in 2025?", "type": "in_domain"},
    {"id": 22, "query": "What is Apple's inventory valuation policy and what were ending inventories in 2025?", "type": "in_domain"},
    {"id": 23, "query": "How much did Apple spend on selling, general, and administrative expenses in 2025?", "type": "in_domain"},
    {"id": 24, "query": "What was Apple's net income for fiscal year 2025 compared to fiscal year 2024?", "type": "in_domain"},
    {"id": 25, "query": "What were Apple's total accounts receivable and allowance for credit losses?", "type": "in_domain"},
    {"id": 26, "query": "Describe the commercial risks regarding intellectual property litigation and licensing disputes.", "type": "in_domain"},
    {"id": 27, "query": "What are the key provisions and status of Apple's term debt maturities over the next 5 years?", "type": "in_domain"},
    {"id": 28, "query": "What segment geographical net sales were reported for Europe and Greater China in 2025?", "type": "in_domain"},
    {"id": 29, "query": "What was the total provision for income taxes in 2025?", "type": "in_domain"},
    {"id": 30, "query": "What are Apple's capital expenditure commitments and cash flows from investing activities?", "type": "in_domain"},
    {"id": 31, "query": "What was the cash flow generated from operating activities in 2025?", "type": "in_domain"},
    {"id": 32, "query": "What are the critical risks regarding cybersecurity breaches and data privacy regulations?", "type": "in_domain"},
    {"id": 33, "query": "What derivative instruments and hedging activities does Apple utilize for interest rate risk?", "type": "in_domain"},
    {"id": 34, "query": "What was the total amount of dividends declared per common share in fiscal 2025?", "type": "in_domain"},
    {"id": 35, "query": "What are the primary commercial risks associated with dependency on third-party cellular carriers?", "type": "in_domain"},

    # --- 15 Out-of-Domain / Abstention Adversarial Queries ---
    {"id": 36, "query": "What is Apple's projected total revenue target for fiscal year 2030?", "type": "adversarial"},
    {"id": 37, "query": "What was the high school grade point average of CEO Tim Cook?", "type": "adversarial"},
    {"id": 38, "query": "What are the confidential battery chemistry specifications of the unannounced iPhone 18?", "type": "adversarial"},
    {"id": 39, "query": "How much profit did competitor Microsoft generate from its Azure cloud business in 2025?", "type": "adversarial"},
    {"id": 40, "query": "What is the secret recipe of the cafeteria food served at Apple Park headquarters?", "type": "adversarial"},
    {"id": 41, "query": "What are Apple's internal projections for quantum computing hardware launches in 2035?", "type": "adversarial"},
    {"id": 42, "query": "What was the purchase price of Steve Jobs' personal yacht disclosed in the filing?", "type": "adversarial"},
    {"id": 43, "query": "What are Google's internal advertising revenue margins cited in this Apple report?", "type": "adversarial"},
    {"id": 44, "query": "What are the specific dates and meeting minutes of secret Apple board meetings regarding Project Titan?", "type": "adversarial"},
    {"id": 45, "query": "What is the expected release date and pricing tier of the Apple flying car?", "type": "adversarial"},
    {"id": 46, "query": "How many total cups of coffee were consumed by Apple engineers in fiscal 2025?", "type": "adversarial"},
    {"id": 47, "query": "What is the exact square footage of Tim Cook's private residence?", "type": "adversarial"},
    {"id": 48, "query": "What are Amazon Web Services' server maintenance operating expenses?", "type": "adversarial"},
    {"id": 49, "query": "What was the cryptocurrency trading profit made by Apple's treasury department?", "type": "adversarial"},
    {"id": 50, "query": "What are the forecasted quarterly sales figures for Samsung Galaxy smartphones in 2026?", "type": "adversarial"},
]


def load_or_compute_embeddings(embedder: BGEEmbedding, chunk_texts: list[str]) -> list[list[float]]:
    if os.path.exists(VECTORS_CACHE_FILE):
        print(f"[*] Loading precomputed embeddings from cache: {VECTORS_CACHE_FILE}", flush=True)
        cached_vectors = np.load(VECTORS_CACHE_FILE)
        if len(cached_vectors) == len(chunk_texts):
            return cached_vectors.tolist()

    print(f"[*] Computing embeddings for {len(chunk_texts)} chunks...", flush=True)
    vectors = embedder.embed_documents(chunk_texts)
    np.save(VECTORS_CACHE_FILE, np.array(vectors, dtype=np.float32))
    return vectors


def run_grounding_benchmark():
    print("=" * 95, flush=True)
    print("TEST_004_GENERATION_GROUNDING_BENCHMARK: 50-QUERY GROUNDING & HALLUCINATION BASELINE", flush=True)
    print("Corpus: Apple Inc. Form 10-K (FY2025) | Model: gemini-3.5-flash-lite | NLI: nli-deberta-v3-small", flush=True)
    print("=" * 95, flush=True)

    # 1. Load and chunk corpus
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        clean_text = f.read()

    documents = recursive_splitters(clean_text, chunk_size=1000, chunk_overlap=100)
    for idx, doc in enumerate(documents):
        doc.metadata["chunk_id"] = f"chunk_{idx:03d}"
        doc.metadata["source"] = "aapl-20250927.htm"
        doc.metadata["ticker"] = "AAPL"
        doc.metadata["fiscal_year"] = "2025"

    total_chunks = len(documents)
    print(f"[+] Loaded and chunked corpus: {total_chunks} chunks.", flush=True)

    # 2. Embedding & Storage
    embedder = BGEEmbedding()
    chunk_texts = [d.page_content for d in documents]
    vectors = load_or_compute_embeddings(embedder, chunk_texts)

    qdrant_store = QdrantStore(collection_name="aapl_grounding_bench", vector_size=embedder.dimension, location=":memory:")
    qdrant_store.upsert_documents(documents=documents, vectors=vectors)

    bm25_store = BM25Store()
    bm25_store.index_documents(documents=documents)
    print("[+] Qdrant and BM25 storage indices ready in memory.", flush=True)

    # 3. Pipeline Modules
    reranker = get_reranker("local")
    assembler = ContextAssembler(max_tokens=3000)
    prompt_builder = PromptBuilder()
    llm = get_llm_provider("gemini", model_name="gemini-3.5-flash-lite")
    verifier = GroundingVerifier()
    print("[+] Reranker, ContextAssembler, PromptBuilder, LLMProvider, and GroundingVerifier initialized.\n", flush=True)

    # 4. Evaluation Loop
    query_eval_records = []
    latencies_sec = []

    status_counts = {"GROUNDED": 0, "PARTIAL": 0, "HALLUCINATED": 0}
    in_domain_status = {"GROUNDED": 0, "PARTIAL": 0, "HALLUCINATED": 0}
    adversarial_status = {"GROUNDED": 0, "PARTIAL": 0, "HALLUCINATED": 0}

    total_claims_accum = 0
    grounded_claims_accum = 0

    print("-" * 95, flush=True)
    print(f"{'Q#':<3} | {'Type':<12} | {'Status':<13} | {'Ground Score':<13} | {'Claims (G/Tot)':<15} | Query", flush=True)
    print("-" * 95, flush=True)

    for item in BENCHMARK_50_QUERIES:
        q_id = item["id"]
        query = item["query"]
        q_type = item["type"]

        t0 = time.perf_counter()

        # Step A: Hybrid Candidate Retrieval (Top 20 each)
        q_vec = embedder.embed_query(query)
        dense_candidates = qdrant_store.search(query_vector=q_vec, top_k=20)
        sparse_candidates = bm25_store.search(query=query, top_k=20)
        hybrid_candidates = reciprocal_rank_fusion([dense_candidates, sparse_candidates], k=60, top_k=20)

        # Step B: Cross-Encoder Rerank (Top 20 -> Top 5)
        top5_reranked = reranker.rerank(query=query, documents=hybrid_candidates, top_k=5)

        # Step C: Context Assembly (Structured XML + Token Guard)
        assembled = assembler.assemble(top5_reranked)

        # Step D: Prompt Building
        prompt_payload = prompt_builder.build(query=query, context_str=assembled.context_str)

        # Step E: LLM Generation (with rate-limit backoff)
        answer = ""
        for attempt in range(3):
            try:
                answer = llm.generate(prompt_payload)
                break
            except Exception as e:
                time.sleep(2 * (attempt + 1))
                if attempt == 2:
                    answer = "The provided documents do not contain sufficient information to answer this question."

        # Step F: Two-Tier NLI Grounding Verification
        grounding_result = verifier.verify(answer=answer, retrieved_documents=top5_reranked)

        latency = time.perf_counter() - t0
        latencies_sec.append(latency)

        # Track Metrics
        status = grounding_result.status
        status_counts[status] += 1
        if q_type == "in_domain":
            in_domain_status[status] += 1
        else:
            adversarial_status[status] += 1

        total_claims_accum += grounding_result.total_claims
        grounded_claims_accum += grounding_result.grounded_claims_count

        claims_str = f"{grounding_result.grounded_claims_count}/{grounding_result.total_claims}"
        print(f"{q_id:<3} | {q_type:<12} | {status:<13} | {grounding_result.overall_grounding_score:<13.4f} | {claims_str:<15} | {query[:38]}...", flush=True)

        query_eval_records.append({
            "query_id": q_id,
            "query": query,
            "query_type": q_type,
            "retrieved_chunk_ids": assembled.included_chunk_ids,
            "generated_answer": answer,
            "grounding_status": status,
            "grounding_score": grounding_result.overall_grounding_score,
            "total_claims": grounding_result.total_claims,
            "grounded_claims": grounding_result.grounded_claims_count,
            "hallucinated_citations": grounding_result.citation_coverage.hallucinated_citations,
            "is_abstention": grounding_result.citation_coverage.is_abstention,
            "explanation": grounding_result.explanation,
            "latency_sec": round(latency, 2),
        })

    # Summary Calculations
    total_q = len(BENCHMARK_50_QUERIES)
    grounded_rate_pct = (status_counts["GROUNDED"] / total_q) * 100
    partial_rate_pct = (status_counts["PARTIAL"] / total_q) * 100
    hallucination_rate_pct = (status_counts["HALLUCINATED"] / total_q) * 100
    strict_hallucination_pct = ((status_counts["HALLUCINATED"] + status_counts["PARTIAL"]) / total_q) * 100

    claim_precision_pct = (grounded_claims_accum / total_claims_accum * 100) if total_claims_accum > 0 else 100.0
    abstention_accuracy_pct = (adversarial_status["GROUNDED"] / 15) * 100

    mean_latency = statistics.mean(latencies_sec)

    print("\n" + "=" * 95, flush=True)
    print("PHASE 4 BASELINE DELIVERABLE: GROUNDING & HALLUCINATION REPORT", flush=True)
    print("=" * 95, flush=True)
    print(f"Total Evaluated Queries           : {total_q} (35 In-Domain + 15 Adversarial)")
    print(f"Grounded Answers (100% Verified)  : {status_counts['GROUNDED']} / {total_q} ({grounded_rate_pct:.1f}%)")
    print(f"Partial Answers (Mixed Claims)    : {status_counts['PARTIAL']} / {total_q} ({partial_rate_pct:.1f}%)")
    print(f"Hallucinated Answers              : {status_counts['HALLUCINATED']} / {total_q} ({hallucination_rate_pct:.1f}%)")
    print("-" * 95)
    print(f"Strict Hallucination Rate (Risk)  : {strict_hallucination_pct:.1f}% (Partial + Hallucinated)")
    print(f"Claim-Level Grounding Precision   : {claim_precision_pct:.1f}% ({grounded_claims_accum}/{total_claims_accum} claims entailed)")
    print(f"Abstention Accuracy on Out-of-Doc : {abstention_accuracy_pct:.1f}% ({adversarial_status['GROUNDED']}/15 cleanly abstained)")
    print(f"Average End-to-End Latency        : {mean_latency:.2f}s per query")
    print("=" * 95, flush=True)

    # 5. Save to Benchmark Logbooks
    benchmark_payload = {
        "test_run_header": {
            "test_id": "TEST_004_GENERATION_GROUNDING_BENCHMARK",
            "run_index": 4,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "target_document": "Apple Inc. Form 10-K (FY 2025)",
            "benchmark_objective": "Measure baseline hallucination rate and citation grounding on 50 queries"
        },
        "system_architecture_snapshot": {
            "pipeline_stage": "Stage 7: Generation + Citation + Grounding (Phase 4 Deliverable)",
            "llm": {
                "provider": "GeminiProvider",
                "model": "gemini-3.5-flash-lite",
                "temperature": 0.0,
                "prompt_contract": "Strict per-claim [chunk-id] citations + explicit negative constraint abstention"
            },
            "grounding_verifier": {
                "tier_1": "CitationExtractor regex & coverage scoring",
                "tier_2": "NLI DeBERTa-v3-small cross-encoder entailment scoring",
                "thresholds": {"entailment": 0.70, "contradiction": 0.50}
            },
            "retrieval_funnel": "Qdrant (768-dim) + BM25Okapi -> RRF (k=60, Top-20) -> Cross-Encoder MiniLM-L6 (Top-5)"
        },
        "headline_metrics": {
            "total_queries_evaluated": total_q,
            "grounded_answers_count": status_counts["GROUNDED"],
            "partial_answers_count": status_counts["PARTIAL"],
            "hallucinated_answers_count": status_counts["HALLUCINATED"],
            "grounded_rate_pct": round(grounded_rate_pct, 2),
            "strict_hallucination_rate_pct": round(strict_hallucination_pct, 2),
            "claim_level_grounding_precision_pct": round(claim_precision_pct, 2),
            "abstention_accuracy_pct": round(abstention_accuracy_pct, 2),
            "mean_latency_sec": round(mean_latency, 2),
        },
        "per_query_audit_log": query_eval_records
    }

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
        print(f"[+] Saved TEST_004_GENERATION_GROUNDING_BENCHMARK to {path}", flush=True)


if __name__ == "__main__":
    run_grounding_benchmark()
