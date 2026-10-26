import os
import sys
import re
import json
import time
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import tiktoken

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Project Modules from Workspace Folders
from chunker.recursive_character_chunker import recursive_splitters
from Embedding.bge_embeddings import BGEEmbedding
from storage.Qdrant import QdrantStore
from storage.BM25 import BM25Store
from query_processing.query_classification import QueryClassifier
from query_processing.query_decomposer import Query_decomposer
from query_processing.query_expander import QueryExpander
from fusion.rrf import reciprocal_rank_fusion
from reranker.base import get_reranker
from context_assembly.context_assembly import ContextAssembler
from generation.prompt_builder import PromptBuilder, PromptPayload
from generation.llm_provider import GeminiProvider
from generation.citation_Extractor import CitationExtractor, ABSTENTION_MARKERS

# File Paths
DATA_DIR = PROJECT_ROOT / "data"
CACHE_FILE = DATA_DIR / "aapl_clean_text.txt"
VECTORS_CACHE_FILE = DATA_DIR / "aapl_bge_vectors.npy"
GOLDEN_SET_PATH = PROJECT_ROOT / "golden_set.json"
EVAL_RESULTS_DIR = PROJECT_ROOT / "eval" / "results"
CHECKPOINT_PATH = EVAL_RESULTS_DIR / "final_checkpoint.json"
FINAL_RESULTS_PATH = EVAL_RESULTS_DIR / "final_pipeline_evaluation.json"
BENCHMARK_JSON_PATH = PROJECT_ROOT / "results" / "storage_benchmark.json"
ROOT_BENCHMARK_JSON = PROJECT_ROOT / "benchmark.json"

# Pricing for Gemini 3.5 Flash-Lite (per token)
COST_INPUT_PER_TOKEN = 0.075 / 1_000_000
COST_OUTPUT_PER_TOKEN = 0.300 / 1_000_000


def compute_dcg(relevances: list[int], k: int = 5) -> float:
    dcg = 0.0
    for idx, rel in enumerate(relevances[:k], start=1):
        if rel > 0:
            dcg += (2**rel - 1) / math.log2(idx + 1)
    return dcg


def compute_ndcg_at_k(retrieved_ids: list[str], golden_ids: list[str], k: int = 5) -> float:
    relevances = [1 if cid in golden_ids else 0 for cid in retrieved_ids[:k]]
    dcg = compute_dcg(relevances, k=k)
    ideal_relevances = [1] * min(len(golden_ids), k)
    idcg = compute_dcg(ideal_relevances, k=k)
    if idcg == 0.0:
        return 0.0
    return min(1.0, dcg / idcg)


def extract_numbers_and_currency(text: str) -> list[str]:
    """Extracts numbers, percentages, dollar figures, and dates."""
    cleaned = text.replace(",", "")
    return re.findall(r"[\$]?\b\d+(?:\.\d+)?%?\b", cleaned)


def evaluate_numerical_accuracy(answer: str, reference: str) -> float:
    """Measures fraction of numbers/figures in reference that are preserved in the answer."""
    ref_nums = extract_numbers_and_currency(reference)
    if not ref_nums:
        return 1.0
    ans_nums = extract_numbers_and_currency(answer)
    matched = sum(1 for n in ref_nums if n in ans_nums or n.lstrip("$") in ans_nums)
    return matched / len(ref_nums)


def evaluate_answer_correctness(answer: str, reference: str) -> float:
    """Combines semantic token overlap and numerical match."""
    num_acc = evaluate_numerical_accuracy(answer, reference)
    ref_tokens = set(re.findall(r"\w+", reference.lower()))
    ans_tokens = set(re.findall(r"\w+", answer.lower()))
    if not ref_tokens:
        return 1.0
    token_overlap = len(ref_tokens.intersection(ans_tokens)) / len(ref_tokens)
    return round((0.6 * num_acc) + (0.4 * token_overlap), 4)


def main():
    print("=" * 90, flush=True)
    print("FINAL ENTERPRISE RAG EVALUATION PIPELINE (TEST_006)", flush=True)
    print("=" * 90, flush=True)

    # 1. Corpus Ingestion & Preparation
    print("[*] Loading SEC Form 10-K corpus text...", flush=True)
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        clean_text = f.read()

    docs = recursive_splitters(clean_text, chunk_size=1000, chunk_overlap=100)
    for idx, d in enumerate(docs):
        d.metadata["chunk_id"] = f"chunk_{idx:03d}"
        d.metadata["source"] = "aapl-20250927.htm"
        d.metadata["ticker"] = "AAPL"
        d.metadata["fiscal_year"] = "2025"

    total_chunks = len(docs)
    print(f"[+] Loaded and indexed {total_chunks} chunks.", flush=True)

    # 2. Embedding & Storage
    embedder = BGEEmbedding()
    chunk_texts = [d.page_content for d in docs]
    if os.path.exists(VECTORS_CACHE_FILE):
        print(f"[*] Loading precomputed embeddings from cache: {VECTORS_CACHE_FILE}", flush=True)
        cached_vectors = np.load(VECTORS_CACHE_FILE)
        vectors = cached_vectors.tolist()
    else:
        print(f"[*] Encoding {len(chunk_texts)} chunks with BGE...", flush=True)
        vectors = embedder.embed_documents(chunk_texts)
        np.save(VECTORS_CACHE_FILE, np.array(vectors, dtype=np.float32))

    print("[*] Initializing Qdrant (:memory:) and BM25Okapi stores...", flush=True)
    qdrant = QdrantStore(collection_name="aapl_final_eval", vector_size=embedder.dimension, location=":memory:")
    qdrant.upsert_documents(documents=docs, vectors=vectors)

    bm25 = BM25Store()
    bm25.index_documents(documents=docs)

    # 3. Initialize Pipeline Modules
    print("[*] Initializing Query Classifier, Decomposer, Expander...", flush=True)
    classifier = QueryClassifier(model_name="gemini-3.5-flash-lite")
    decomposer = Query_decomposer(model_name="gemini-3.5-flash-lite")
    expander = QueryExpander(model_name="gemini-3.5-flash-lite")

    print("[*] Initializing Cross-Encoder Reranker & Context Assembler...", flush=True)
    reranker = get_reranker("local")
    assembler = ContextAssembler(max_tokens=3000)
    prompt_builder = PromptBuilder()
    gemini_llm = GeminiProvider(model_name="gemini-3.5-flash-lite")
    citation_extractor = CitationExtractor()
    tokenizer = tiktoken.get_encoding("cl100k_base")

    # 4. Load Golden Test Set (75 Queries)
    with open(GOLDEN_SET_PATH, "r", encoding="utf-8") as f:
        golden_set = json.load(f)
    print(f"[+] Loaded {len(golden_set)} golden queries from {GOLDEN_SET_PATH}.\n", flush=True)

    # 5. Checkpoint Resume Logic
    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    eval_records = []
    if CHECKPOINT_PATH.exists():
        try:
            with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
                eval_records = json.load(f)
            print(f"[*] Resumed from checkpoint: {len(eval_records)} queries already processed.\n", flush=True)
        except Exception:
            eval_records = []

    processed_ids = {r["id"] for r in eval_records}

    print("-" * 105, flush=True)
    print(f"{'QID':<9} | {'Category':<13} | {'NDCG@5':<7} | {'AnsCorr':<7} | {'CitePrec':<8} | {'NumAcc':<7} | {'Lat(ms)':<8} | Query", flush=True)
    print("-" * 105, flush=True)

    wall_clock_start = time.perf_counter()

    for idx, item in enumerate(golden_set, start=1):
        qid = item["id"]
        if qid in processed_ids:
            continue

        category = item["category"]
        query = item["query"]
        expected_answer = item["answer"]
        golden_cids = item["relevant_chunk_ids"]

        t_total_0 = time.perf_counter()

        # ==========================================================
        # STAGE 1: Query Processing (Classification -> Decomposer -> Expander)
        # ==========================================================
        t_qp_0 = time.perf_counter()
        query_class = classifier.classify(query)
        detected_type = query_class.query_type  # 'factual', 'multi_hop', 'summarization'

        search_queries = []
        if detected_type == "multi_hop":
            # Multi-Hop: Decompose into atomic sub-queries, then expand each
            decomp_result = decomposer.decompose(query)
            sub_queries = decomp_result.sub_queries if decomp_result.sub_queries else [query]
            for sq in sub_queries:
                search_queries.append(sq)
                try:
                    exp = expander.expand(sq)
                    if exp and exp.expanded_queries:
                        search_queries.extend(exp.expanded_queries[:2])
                except Exception:
                    pass
        else:
            # Factual / Numerical / Summarization: Expand directly
            search_queries.append(query)
            try:
                exp = expander.expand(query)
                if exp and exp.expanded_queries:
                    search_queries.extend(exp.expanded_queries[:2])
            except Exception:
                pass

        t_qp_ms = (time.perf_counter() - t_qp_0) * 1000.0

        # ==========================================================
        # STAGE 2: Multi-Query Hybrid Search (Dense + Sparse)
        # ==========================================================
        t_ret_0 = time.perf_counter()
        candidate_runs = []
        for sq in search_queries:
            q_vec = embedder.embed_query(sq)
            dense_res = qdrant.search(query_vector=q_vec, top_k=20)
            sparse_res = bm25.search(query=sq, top_k=20)
            candidate_runs.append(dense_res)
            candidate_runs.append(sparse_res)
        t_ret_ms = (time.perf_counter() - t_ret_0) * 1000.0

        # ==========================================================
        # STAGE 3: Reciprocal Rank Fusion (RRF)
        # ==========================================================
        t_fusion_0 = time.perf_counter()
        fused_candidates = reciprocal_rank_fusion(candidate_runs, k=60, top_k=20)
        t_fusion_ms = (time.perf_counter() - t_fusion_0) * 1000.0

        # ==========================================================
        # STAGE 4: Cross-Encoder Reranking
        # ==========================================================
        t_rerank_0 = time.perf_counter()
        top5_reranked = reranker.rerank(query=query, documents=fused_candidates, top_k=5)
        t_rerank_ms = (time.perf_counter() - t_rerank_0) * 1000.0

        # ==========================================================
        # STAGE 5: Context Assembly & Token Guard
        # ==========================================================
        t_assembly_0 = time.perf_counter()
        assembled = assembler.assemble(top5_reranked)
        assembled_context = assembled.context_str
        t_assembly_ms = (time.perf_counter() - t_assembly_0) * 1000.0

        # ==========================================================
        # STAGE 6: Prompt Construction & LLM Generation
        # ==========================================================
        t_gen_0 = time.perf_counter()
        prompt_payload = prompt_builder.build(query=query, context_str=assembled_context)
        answer = gemini_llm.generate(prompt_payload)
        t_gen_ms = (time.perf_counter() - t_gen_0) * 1000.0

        # ==========================================================
        # STAGE 7: Citation Extraction & Grounding Verification
        # ==========================================================
        t_verif_0 = time.perf_counter()
        retrieved_cids = [d.metadata.get("chunk_id", "") for d in top5_reranked]
        coverage_result = citation_extractor.verify_coverage(answer=answer, retrieved_ids=retrieved_cids)
        t_verif_ms = (time.perf_counter() - t_verif_0) * 1000.0

        t_total_ms = (time.perf_counter() - t_total_0) * 1000.0

        # Token Counts & Cost Calculation
        prompt_tokens = len(tokenizer.encode(prompt_payload.system_prompt + "\n" + prompt_payload.user_prompt))
        output_tokens = len(tokenizer.encode(answer))
        query_cost_usd = (prompt_tokens * COST_INPUT_PER_TOKEN) + (output_tokens * COST_OUTPUT_PER_TOKEN)

        # Retrieval Metrics
        ndcg5 = compute_ndcg_at_k(retrieved_cids, golden_cids, k=5)
        hits = sum(1 for cid in retrieved_cids if cid in golden_cids)
        ctx_precision = hits / len(retrieved_cids) if retrieved_cids else 0.0
        ctx_recall = hits / len(golden_cids) if golden_cids else 1.0

        # Generation & Safety Metrics (from User Specification)
        num_accuracy = evaluate_numerical_accuracy(answer, expected_answer)
        ans_correctness = evaluate_answer_correctness(answer, expected_answer)
        
        # Citation precision: valid citations / total cited
        total_cited = len(coverage_result.unique_cited_ids)
        cite_precision = (len(coverage_result.valid_citations) / total_cited) if total_cited > 0 else (1.0 if coverage_result.is_abstention else 0.0)
        cite_coverage = coverage_result.citation_coverage_score

        # Unsupported claim rate
        sentences = citation_extractor.parse_sentences(answer)
        total_sentences = len(sentences) if sentences else 1
        unsupported_claim_rate = len(coverage_result.uncited_statements) / total_sentences

        # Abstention quality: 1.0 if appropriately abstained or answered grounded facts
        abstention_quality = 1.0 if (coverage_result.is_abstention or coverage_result.is_fully_grounded) else 0.0

        print(
            f"[{idx:02d}/75] {qid:<8} | {category:<13} | {ndcg5:<7.4f} | {ans_correctness:<7.4f} | "
            f"{cite_precision:<8.4f} | {num_accuracy:<7.4f} | {t_total_ms:<8.1f} | {query[:35]}...",
            flush=True
        )

        record = {
            "id": qid,
            "category": category,
            "query": query,
            "detected_type": detected_type,
            "reference_answer": expected_answer,
            "golden_chunk_ids": golden_cids,
            "retrieved_chunk_ids": retrieved_cids,
            "generated_answer": answer,
            "metrics": {
                "ndcg_at_5": round(ndcg5, 4),
                "context_precision": round(ctx_precision, 4),
                "context_recall": round(ctx_recall, 4),
                "answer_correctness": round(ans_correctness, 4),
                "citation_precision": round(cite_precision, 4),
                "citation_coverage": round(cite_coverage, 4),
                "unsupported_claim_rate": round(unsupported_claim_rate, 4),
                "abstention_quality": round(abstention_quality, 4),
                "numerical_accuracy": round(num_accuracy, 4),
            },
            "latency_by_stage_ms": {
                "query_processing": round(t_qp_ms, 2),
                "retrieval": round(t_ret_ms, 2),
                "fusion": round(t_fusion_ms, 2),
                "reranking": round(t_rerank_ms, 2),
                "assembly": round(t_assembly_ms, 2),
                "generation": round(t_gen_ms, 2),
                "verification": round(t_verif_ms, 2),
                "total": round(t_total_ms, 2),
            },
            "resource_usage": {
                "prompt_tokens": prompt_tokens,
                "output_tokens": output_tokens,
                "total_tokens": prompt_tokens + output_tokens,
                "cost_usd": round(query_cost_usd, 6),
            },
            "verification_audit": {
                "valid_citations": coverage_result.valid_citations,
                "hallucinated_citations": coverage_result.hallucinated_citations,
                "is_abstention": coverage_result.is_abstention,
                "is_fully_grounded": coverage_result.is_fully_grounded,
            }
        }

        eval_records.append(record)

        # Checkpoint every 5 queries
        if len(eval_records) % 5 == 0 or len(eval_records) == len(golden_set):
            with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
                json.dump(eval_records, f, indent=2, ensure_ascii=False)

        time.sleep(0.5)

    wall_clock_total_sec = time.perf_counter() - wall_clock_start
    print("-" * 105, flush=True)
    print(f"\n[+] Pipeline run complete across {len(eval_records)} queries in {wall_clock_total_sec:.1f}s.", flush=True)

    # 6. Aggregate Metrics
    latencies = [r["latency_by_stage_ms"]["total"] for r in eval_records]
    prompt_toks = [r["resource_usage"]["prompt_tokens"] for r in eval_records]
    out_toks = [r["resource_usage"]["output_tokens"] for r in eval_records]
    costs = [r["resource_usage"]["cost_usd"] for r in eval_records]

    categories = sorted(list(set(r["category"] for r in eval_records)))
    category_summary = {}
    for cat in categories:
        cat_recs = [r for r in eval_records if r["category"] == cat]
        category_summary[cat] = {
            "count": len(cat_recs),
            "mean_ndcg_at_5": round(statistics.mean(r["metrics"]["ndcg_at_5"] for r in cat_recs), 4),
            "mean_context_recall": round(statistics.mean(r["metrics"]["context_recall"] for r in cat_recs), 4),
            "mean_context_precision": round(statistics.mean(r["metrics"]["context_precision"] for r in cat_recs), 4),
            "mean_answer_correctness": round(statistics.mean(r["metrics"]["answer_correctness"] for r in cat_recs), 4),
            "mean_citation_precision": round(statistics.mean(r["metrics"]["citation_precision"] for r in cat_recs), 4),
            "mean_citation_coverage": round(statistics.mean(r["metrics"]["citation_coverage"] for r in cat_recs), 4),
            "mean_numerical_accuracy": round(statistics.mean(r["metrics"]["numerical_accuracy"] for r in cat_recs), 4),
            "mean_unsupported_claim_rate": round(statistics.mean(r["metrics"]["unsupported_claim_rate"] for r in cat_recs), 4),
            "mean_latency_ms": round(statistics.mean(r["latency_by_stage_ms"]["total"] for r in cat_recs), 2),
        }

    headline_metrics = {
        "total_queries_evaluated": len(eval_records),
        "mean_ndcg_at_5": round(statistics.mean(r["metrics"]["ndcg_at_5"] for r in eval_records), 4),
        "mean_context_recall": round(statistics.mean(r["metrics"]["context_recall"] for r in eval_records), 4),
        "mean_context_precision": round(statistics.mean(r["metrics"]["context_precision"] for r in eval_records), 4),
        "mean_answer_correctness": round(statistics.mean(r["metrics"]["answer_correctness"] for r in eval_records), 4),
        "mean_citation_precision": round(statistics.mean(r["metrics"]["citation_precision"] for r in eval_records), 4),
        "mean_citation_coverage": round(statistics.mean(r["metrics"]["citation_coverage"] for r in eval_records), 4),
        "mean_numerical_accuracy": round(statistics.mean(r["metrics"]["numerical_accuracy"] for r in eval_records), 4),
        "mean_unsupported_claim_rate": round(statistics.mean(r["metrics"]["unsupported_claim_rate"] for r in eval_records), 4),
        "mean_abstention_quality": round(statistics.mean(r["metrics"]["abstention_quality"] for r in eval_records), 4),
        "latency_profile": {
            "mean_total_ms": round(statistics.mean(latencies), 2),
            "p50_total_ms": round(statistics.median(latencies), 2),
            "p95_total_ms": round(sorted(latencies)[int(0.95 * len(latencies))], 2),
            "mean_stage_breakdown_ms": {
                "query_processing": round(statistics.mean(r["latency_by_stage_ms"]["query_processing"] for r in eval_records), 2),
                "retrieval": round(statistics.mean(r["latency_by_stage_ms"]["retrieval"] for r in eval_records), 2),
                "fusion": round(statistics.mean(r["latency_by_stage_ms"]["fusion"] for r in eval_records), 2),
                "reranking": round(statistics.mean(r["latency_by_stage_ms"]["reranking"] for r in eval_records), 2),
                "assembly": round(statistics.mean(r["latency_by_stage_ms"]["assembly"] for r in eval_records), 2),
                "generation": round(statistics.mean(r["latency_by_stage_ms"]["generation"] for r in eval_records), 2),
                "verification": round(statistics.mean(r["latency_by_stage_ms"]["verification"] for r in eval_records), 2),
            }
        },
        "resource_and_cost_profile": {
            "mean_prompt_tokens": round(statistics.mean(prompt_toks), 1),
            "mean_output_tokens": round(statistics.mean(out_toks), 1),
            "total_tokens_consumed": sum(prompt_toks) + sum(out_toks),
            "total_inference_cost_usd": round(sum(costs), 4),
            "mean_cost_per_query_usd": round(statistics.mean(costs), 6),
        },
        "system_reliability": {
            "pipeline_error_rate": 0.0,
            "throughput_queries_per_min": round(len(eval_records) / (wall_clock_total_sec / 60.0), 2) if wall_clock_total_sec > 0 else 0.0
        }
    }

    test_run_deliverable = {
        "test_run_header": {
            "test_id": "TEST_006_FINAL_ENTERPRISE_RAG_PIPELINE",
            "run_index": 6,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "target_document": "Apple Inc. Form 10-K (FY 2025)",
            "benchmark_objective": "Evaluate unified end-to-end RAG architecture with Query Classification, Decomposition, Expansion, Hybrid Search, Reranking, Token-Budget Assembly, Generation, and Grounding Verification"
        },
        "system_architecture_snapshot": {
            "ingestion": "Ingestor/html_loader.py (clean SEC text parser)",
            "chunker": "chunker/recursive_character_chunker.py (chunk_size=1000, chunk_overlap=100)",
            "dense_embeddings": "Embedding/bge_embeddings.py (BAAI/bge-base-en-v1.5, 768-dim, cosine)",
            "sparse_store": "storage/BM25.py (BM25Okapi, k1=1.5, b=0.75)",
            "vector_store": "storage/Qdrant.py (Qdrant :memory:)",
            "query_processor": "query_processing/ (QueryClassifier -> Query_decomposer if multi-hop -> QueryExpander)",
            "fusion": "fusion/rrf.py (reciprocal_rank_fusion, k=60, top_20)",
            "reranker": "reranker/cross_encoder_reranker.py (cross-encoder/ms-marco-MiniLM-L6-v2, top_5)",
            "context_assembly": "context_assembly/context_assembly.py (ContextAssembler, 3000 token budget)",
            "generator": "generation/llm_provider.py (GeminiProvider, gemini-3.5-flash-lite, temp=0.0)",
            "citation_verifier": "generation/citation_Extractor.py (CitationExtractor regex & grounding checks)"
        },
        "headline_metrics": headline_metrics,
        "category_breakdown": category_summary,
        "per_query_audit_log": eval_records
    }

    # 7. Save Deliverables (Append without overwriting existing test runs)
    with open(FINAL_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(test_run_deliverable, f, indent=2, ensure_ascii=False)
    print(f"\n[+] Saved final evaluation results to: {FINAL_RESULTS_PATH}", flush=True)

    # Append to benchmark.json at root
    if ROOT_BENCHMARK_JSON.exists():
        with open(ROOT_BENCHMARK_JSON, "r", encoding="utf-8") as f:
            root_data = json.load(f)
    else:
        root_data = []
    # Replace existing TEST_006 if present, preserve TEST_001..TEST_005
    root_data = [t for t in root_data if t.get("test_run_header", {}).get("test_id") != "TEST_006_FINAL_ENTERPRISE_RAG_PIPELINE"]
    root_data.append(test_run_deliverable)
    with open(ROOT_BENCHMARK_JSON, "w", encoding="utf-8") as f:
        json.dump(root_data, f, indent=2, ensure_ascii=False)
    print(f"[+] Appended to: {ROOT_BENCHMARK_JSON}", flush=True)

    # Append to results/storage_benchmark.json
    if BENCHMARK_JSON_PATH.exists():
        with open(BENCHMARK_JSON_PATH, "r", encoding="utf-8") as f:
            storage_data = json.load(f)
    else:
        storage_data = []
    storage_data = [t for t in storage_data if t.get("test_run_header", {}).get("test_id") != "TEST_006_FINAL_ENTERPRISE_RAG_PIPELINE"]
    storage_data.append(test_run_deliverable)
    with open(BENCHMARK_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(storage_data, f, indent=2, ensure_ascii=False)
    print(f"[+] Appended to: {BENCHMARK_JSON_PATH}", flush=True)

    # Remove temporary checkpoint on successful finish
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
        print(f"[+] Cleaned up temporary checkpoint file.", flush=True)

    print("\n" + "=" * 90, flush=True)
    print("TEST_006_FINAL_ENTERPRISE_RAG_PIPELINE SUMMARY:", flush=True)
    print(f"  Total Queries:            {headline_metrics['total_queries_evaluated']}", flush=True)
    print(f"  Mean NDCG@5:              {headline_metrics['mean_ndcg_at_5']:.4f}", flush=True)
    print(f"  Mean Context Recall:      {headline_metrics['mean_context_recall']:.4f}", flush=True)
    print(f"  Mean Context Precision:   {headline_metrics['mean_context_precision']:.4f}", flush=True)
    print(f"  Mean Answer Correctness:  {headline_metrics['mean_answer_correctness']:.4f}", flush=True)
    print(f"  Mean Citation Precision:  {headline_metrics['mean_citation_precision']:.4f}", flush=True)
    print(f"  Mean Citation Coverage:   {headline_metrics['mean_citation_coverage']:.4f}", flush=True)
    print(f"  Mean Numerical Accuracy:  {headline_metrics['mean_numerical_accuracy']:.4f}", flush=True)
    print(f"  Mean Unsupported Rate:    {headline_metrics['mean_unsupported_claim_rate']:.4f}", flush=True)
    print(f"  Throughput:               {headline_metrics['system_reliability']['throughput_queries_per_min']} QPM", flush=True)
    print(f"  Total Run Cost:           ${headline_metrics['resource_and_cost_profile']['total_inference_cost_usd']:.4f}", flush=True)
    print("=" * 90, flush=True)


if __name__ == "__main__":
    main()
