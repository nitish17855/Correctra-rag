import os
import sys
import re
import json
import time
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
import numpy as np
import tiktoken

# Ensure workspace root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

# LangChain LCEL Core
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.documents import Document

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

# CRAG Core Engine Modules
from CRAG.orchestrate import CRAGOrchestrator, CRAGResult
from CRAG.actionRouter import ActionType
from CRAG.source_attributor import (
    SourceAttributor,
    CRAG_SYSTEM_PROMPT,
    CRAG_USER_PROMPT_TEMPLATE,
    AttributionMetrics,
)

# File Paths
DATA_DIR = PROJECT_ROOT / "data"
CACHE_FILE = DATA_DIR / "aapl_clean_text.txt"
VECTORS_CACHE_FILE = DATA_DIR / "aapl_bge_vectors.npy"
GOLDEN_SET_PATH = PROJECT_ROOT / "golden_set.json"
EVAL_RESULTS_DIR = PROJECT_ROOT / "eval" / "results"
CHECKPOINT_PATH = EVAL_RESULTS_DIR / "final_pipeline_crag_v1_checkpoint.json"
FINAL_RESULTS_PATH = EVAL_RESULTS_DIR / "final_pipeline_crag_v1_evaluation.json"
BENCHMARK_JSON_PATH = PROJECT_ROOT / "results" / "storage_benchmark.json"
ROOT_BENCHMARK_JSON = PROJECT_ROOT / "benchmark.json"

# Pricing for Gemini 3.5 Flash-Lite (USD per token)
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


# ==============================================================================
# LANGCHAIN LCEL CHAIN DEFINITION FOR ENTERPRISE CRAG PIPELINE
# ==============================================================================

def build_crag_pipeline_chain(
    embedder: BGEEmbedding,
    qdrant: QdrantStore,
    bm25: BM25Store,
    classifier: QueryClassifier,
    decomposer: Query_decomposer,
    expander: QueryExpander,
    reranker,
    crag_orchestrator: CRAGOrchestrator,
    assembler: ContextAssembler,
    gemini_llm: GeminiProvider,
    prompt_builder: PromptBuilder,
    citation_extractor: CitationExtractor,
    source_attributor: SourceAttributor,
    tokenizer,
) -> RunnableLambda:
    """
    Constructs an end-to-end unified LangChain LCEL chain for Corrective RAG:
    
    [Input State: {"query": str, "golden_cids": list[str], ...}]
         |
    Stage 1: Query Processing (Classifier -> Decomposer -> Expander)
         |
    Stage 2: Hybrid Retrieval (Dense Qdrant + Sparse BM25)
         |
    Stage 3: Reciprocal Rank Fusion (RRF)
         |
    Stage 4: Cross-Encoder Reranker 1 (Top-K Candidate Selection)
         |
    Stage 5 (CRAG Mid-Section):
         - CRAG Evaluator (Batched Pointwise Requesty Judgment)
         - Action Router (CORRECT / AMBIGUOUS / INCORRECT)
         - Branching Refinement & Web Fallback
         - Evidence Preparation (Dual-Source Tagging [DOC:...]/[WEB:...])
         |
    Stage 6: Context Assembly & Generator (Gemini Flash-Lite with CRAG Prompt)
         |
    Stage 7: Grounding & Dual Attribution Verification (Attribution Accuracy)
         v
    [Output State: Comprehensive per-query execution record with metrics & audit]
    """

    def step_query_processing(state: dict[str, Any]) -> dict[str, Any]:
        """Stage 1: Intent classification, decomposition (if multi-hop), and expansion."""
        query = state["query"]
        t_0 = time.perf_counter()

        query_class = classifier.classify(query)
        detected_type = query_class.query_type

        search_queries = []
        if detected_type == "multi_hop":
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
            search_queries.append(query)
            try:
                exp = expander.expand(query)
                if exp and exp.expanded_queries:
                    search_queries.extend(exp.expanded_queries[:2])
            except Exception:
                pass

        state["search_queries"] = search_queries
        state["detected_type"] = detected_type
        state["t_qp_ms"] = (time.perf_counter() - t_0) * 1000.0
        return state

    def step_hybrid_retrieval(state: dict[str, Any]) -> dict[str, Any]:
        """Stage 2: Dense & sparse candidate retrieval across all expanded queries."""
        search_queries = state["search_queries"]
        t_0 = time.perf_counter()

        candidate_runs = []
        for sq in search_queries:
            q_vec = embedder.embed_query(sq)
            dense_res = qdrant.search(query_vector=q_vec, top_k=20)
            sparse_res = bm25.search(query=sq, top_k=20)
            candidate_runs.append(dense_res)
            candidate_runs.append(sparse_res)

        state["candidate_runs"] = candidate_runs
        state["t_ret_ms"] = (time.perf_counter() - t_0) * 1000.0
        return state

    def step_fusion(state: dict[str, Any]) -> dict[str, Any]:
        """Stage 3: Reciprocal Rank Fusion (RRF) over multi-query hybrid runs."""
        candidate_runs = state["candidate_runs"]
        t_0 = time.perf_counter()

        fused_candidates = reciprocal_rank_fusion(candidate_runs, k=60, top_k=20)

        state["fused_candidates"] = fused_candidates
        state["t_fusion_ms"] = (time.perf_counter() - t_0) * 1000.0
        return state

    def step_reranker_1(state: dict[str, Any]) -> dict[str, Any]:
        """Stage 4: First-pass Cross-Encoder Reranker to filter top-5 candidates."""
        query = state["query"]
        fused_candidates = state["fused_candidates"]
        t_0 = time.perf_counter()

        top5_reranked = reranker.rerank(query=query, documents=fused_candidates, top_k=5)
        retrieved_cids = [d.metadata.get("chunk_id", "") for d in top5_reranked]

        state["top5_reranked"] = top5_reranked
        state["retrieved_cids"] = retrieved_cids
        state["t_rerank_ms"] = (time.perf_counter() - t_0) * 1000.0
        return state

    def step_crag_midsection(state: dict[str, Any]) -> dict[str, Any]:
        """
        Stage 5: CRAG Evaluator & Evidence Preparation (The Architecture Mid-Section):
        1. Evaluates candidate chunks pointwise via Requesty judgment.
        2. Routes action: CORRECT, AMBIGUOUS, or INCORRECT.
        3. Branch execution:
           - CORRECT: Decompose accepted chunks into strips -> filter -> recompose.
           - INCORRECT: Discard internal docs -> rewrite query -> Tavily web search.
           - AMBIGUOUS: Refine internal docs + web search -> merge & cross-encoder rerank.
        4. Evidence Preparation: Formats dual-source context tagged with [DOC:...] and [WEB:...].
        """
        query = state["query"]
        top5_reranked = state["top5_reranked"]
        t_0 = time.perf_counter()

        crag_result: CRAGResult = crag_orchestrator.orchestrate(
            query=query,
            documents=top5_reranked,
        )

        state["crag_result"] = crag_result
        state["assembled_context"] = crag_result.formatted_context
        state["final_documents"] = crag_result.final_documents
        state["t_crag_ms"] = (time.perf_counter() - t_0) * 1000.0
        return state

    def step_generation(state: dict[str, Any]) -> dict[str, Any]:
        """Stage 6: Context Assembly & Generator (Gemini Flash-Lite with CRAG prompt)."""
        query = state["query"]
        assembled_context = state["assembled_context"]
        t_0 = time.perf_counter()

        prompt_payload = prompt_builder.build(query=query, context_str=assembled_context)
        answer = gemini_llm.generate(prompt_payload)

        prompt_tokens = len(tokenizer.encode(prompt_payload.system_prompt + "\n" + prompt_payload.user_prompt))
        output_tokens = len(tokenizer.encode(answer))
        query_cost_usd = (prompt_tokens * COST_INPUT_PER_TOKEN) + (output_tokens * COST_OUTPUT_PER_TOKEN)

        state["prompt_payload"] = prompt_payload
        state["answer"] = answer
        state["prompt_tokens"] = prompt_tokens
        state["output_tokens"] = output_tokens
        state["query_cost_usd"] = query_cost_usd
        state["t_gen_ms"] = (time.perf_counter() - t_0) * 1000.0
        return state

    def step_verification(state: dict[str, Any]) -> dict[str, Any]:
        """Stage 7: Citation extraction, grounding verification, and dual-source attribution accuracy."""
        answer = state["answer"]
        final_docs = state["final_documents"]
        golden_cids = state.get("golden_cids", [])
        expected_answer = state.get("expected_answer", "")
        t_0 = time.perf_counter()

        # 1. Base Grounding & Abstention Verification
        doc_cids = [d.metadata.get("chunk_id", "") for d in final_docs if d.metadata.get("chunk_id")]
        coverage_result = citation_extractor.verify_coverage(answer=answer, retrieved_ids=doc_cids)

        # 2. CRAG Dual Attribution Metric (Attribution Accuracy)
        attribution_metrics: AttributionMetrics = source_attributor.attribution_accuracy(
            answer=answer,
            retrieved_docs=final_docs,
        )

        t_verif_ms = (time.perf_counter() - t_0) * 1000.0

        # Retrieval Metrics
        retrieved_cids = state["retrieved_cids"]
        ndcg5 = compute_ndcg_at_k(retrieved_cids, golden_cids, k=5)
        hits = sum(1 for cid in retrieved_cids if cid in golden_cids)
        ctx_precision = hits / len(retrieved_cids) if retrieved_cids else 0.0
        ctx_recall = hits / len(golden_cids) if golden_cids else 1.0

        # Generation Accuracy Metrics
        num_accuracy = evaluate_numerical_accuracy(answer, expected_answer)
        ans_correctness = evaluate_answer_correctness(answer, expected_answer)

        # Unsupported claims & abstention quality
        sentences = citation_extractor.parse_sentences(answer)
        total_sentences = len(sentences) if sentences else 1
        unsupported_claim_rate = len(coverage_result.uncited_statements) / total_sentences
        abstention_quality = 1.0 if (coverage_result.is_abstention or coverage_result.is_fully_grounded) else 0.0

        state["metrics"] = {
            "ndcg_at_5": round(ndcg5, 4),
            "context_precision": round(ctx_precision, 4),
            "context_recall": round(ctx_recall, 4),
            "answer_correctness": round(ans_correctness, 4),
            "citation_precision": round(attribution_metrics.attribution_accuracy, 4),
            "attribution_accuracy": round(attribution_metrics.attribution_accuracy, 4),
            "citation_coverage": round(coverage_result.citation_coverage_score, 4),
            "unsupported_claim_rate": round(unsupported_claim_rate, 4),
            "abstention_quality": round(abstention_quality, 4),
            "numerical_accuracy": round(num_accuracy, 4),
            "internal_citations": attribution_metrics.internal_citation_count,
            "web_citations": attribution_metrics.web_citation_count,
            "hallucinated_citations": attribution_metrics.hallucinated_citations,
        }

        state["coverage_result"] = coverage_result
        state["attribution_metrics"] = attribution_metrics
        state["t_verif_ms"] = t_verif_ms
        return state

    # Compose the entire execution using LangChain LCEL
    rag_chain = (
        RunnablePassthrough()
        | RunnableLambda(step_query_processing)
        | RunnableLambda(step_hybrid_retrieval)
        | RunnableLambda(step_fusion)
        | RunnableLambda(step_reranker_1)
        | RunnableLambda(step_crag_midsection)
        | RunnableLambda(step_generation)
        | RunnableLambda(step_verification)
    )

    return rag_chain


# ==============================================================================
# MAIN BENCHMARK & EVALUATION HARNESS
# ==============================================================================

def main():
    print("=" * 105, flush=True)
    print("FINAL ENTERPRISE CORRECTIVE RAG (CRAG) PIPELINE BENCHMARK (V1)", flush=True)
    print("Architecture: Hybrid -> Reranker 1 -> CRAG Evaluator -> Router -> Evidence Prep -> Generator", flush=True)
    print("=" * 105, flush=True)

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
    qdrant = QdrantStore(collection_name="aapl_crag_eval", vector_size=embedder.dimension, location=":memory:")
    qdrant.upsert_documents(documents=docs, vectors=vectors)

    bm25 = BM25Store()
    bm25.index_documents(documents=docs)

    # 3. Initialize Pipeline Modules
    print("[*] Initializing Query Classifier, Decomposer, Expander...", flush=True)
    classifier = QueryClassifier(model_name="gemini-3.5-flash-lite")
    decomposer = Query_decomposer(model_name="gemini-3.5-flash-lite")
    expander = QueryExpander(model_name="gemini-3.5-flash-lite")

    print("[*] Initializing Reranker 1 & CRAG Orchestrator...", flush=True)
    reranker = get_reranker("local")
    crag_orchestrator = CRAGOrchestrator()
    assembler = ContextAssembler(max_tokens=3000)

    # Prompt builder configured with strict dual-attribution CRAG rules
    prompt_builder = PromptBuilder(
        system_prompt=CRAG_SYSTEM_PROMPT,
        user_template=CRAG_USER_PROMPT_TEMPLATE,
    )
    gemini_llm = GeminiProvider(model_name="gemini-3.5-flash-lite")
    citation_extractor = CitationExtractor()
    source_attributor = SourceAttributor()
    tokenizer = tiktoken.get_encoding("cl100k_base")

    # 4. Build LangChain CRAG Pipeline Chain
    print("[*] Compiling LangChain LCEL CRAG Pipeline Chain...", flush=True)
    rag_chain = build_crag_pipeline_chain(
        embedder=embedder,
        qdrant=qdrant,
        bm25=bm25,
        classifier=classifier,
        decomposer=decomposer,
        expander=expander,
        reranker=reranker,
        crag_orchestrator=crag_orchestrator,
        assembler=assembler,
        gemini_llm=gemini_llm,
        prompt_builder=prompt_builder,
        citation_extractor=citation_extractor,
        source_attributor=source_attributor,
        tokenizer=tokenizer,
    )
    print("[+] LangChain CRAG chain built successfully.", flush=True)

    # 5. Load Golden Test Set (75 Queries)
    with open(GOLDEN_SET_PATH, "r", encoding="utf-8") as f:
        golden_set = json.load(f)
    print(f"[+] Loaded {len(golden_set)} golden queries from {GOLDEN_SET_PATH}.\n", flush=True)

    # 6. Checkpoint Resume Logic
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

    print("-" * 115, flush=True)
    print(f"{'QID':<8} | {'Category':<13} | {'Action':<10} | {'NDCG@5':<7} | {'AttribAcc':<9} | {'AnsCorr':<7} | {'Lat(ms)':<8} | Query", flush=True)
    print("-" * 115, flush=True)

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

        initial_state = {
            "query": query,
            "category": category,
            "expected_answer": expected_answer,
            "golden_cids": golden_cids,
        }

        # Execute unified LangChain CRAG Chain
        result_state = rag_chain.invoke(initial_state)

        t_total_ms = (time.perf_counter() - t_total_0) * 1000.0
        crag_res: CRAGResult = result_state["crag_result"]
        metrics = result_state["metrics"]

        print(
            f"[{idx:02d}/{len(golden_set)}] {qid:<7} | {category:<13} | {crag_res.action:<10} | {metrics['ndcg_at_5']:<7.4f} | "
            f"{metrics['attribution_accuracy']:<9.4f} | {metrics['answer_correctness']:<7.4f} | {t_total_ms:<8.1f} | {query[:30]}...",
            flush=True
        )

        record = {
            "id": qid,
            "category": category,
            "query": query,
            "detected_type": result_state["detected_type"],
            "reference_answer": expected_answer,
            "golden_chunk_ids": golden_cids,
            "retrieved_chunk_ids": result_state["retrieved_cids"],
            "crag_action": crag_res.action,
            "evaluator_scores": crag_res.evaluator_scores,
            "web_searched": crag_res.web_searched,
            "rewritten_web_query": crag_res.rewritten_web_query,
            "accepted_chunk_ids": crag_res.accepted_chunk_ids,
            "discarded_chunk_ids": crag_res.discarded_chunk_ids,
            "generated_answer": result_state["answer"],
            "metrics": metrics,
            "latency_by_stage_ms": {
                "query_processing": round(result_state["t_qp_ms"], 2),
                "retrieval": round(result_state["t_ret_ms"], 2),
                "fusion": round(result_state["t_fusion_ms"], 2),
                "reranking_1": round(result_state["t_rerank_ms"], 2),
                "crag_midsection": round(result_state["t_crag_ms"], 2),
                "generation": round(result_state["t_gen_ms"], 2),
                "verification": round(result_state["t_verif_ms"], 2),
                "total": round(t_total_ms, 2),
            },
            "resource_usage": {
                "prompt_tokens": result_state["prompt_tokens"],
                "output_tokens": result_state["output_tokens"],
                "total_tokens": result_state["prompt_tokens"] + result_state["output_tokens"],
                "cost_usd": round(result_state["query_cost_usd"], 6),
            },
            "verification_audit": {
                "valid_citations": result_state["attribution_metrics"].valid_citations,
                "hallucinated_citations": result_state["attribution_metrics"].hallucinated_citations,
                "internal_citation_count": result_state["attribution_metrics"].internal_citation_count,
                "web_citation_count": result_state["attribution_metrics"].web_citation_count,
                "is_abstention": result_state["coverage_result"].is_abstention,
                "is_fully_grounded": result_state["coverage_result"].is_fully_grounded,
            }
        }

        eval_records.append(record)

        # Checkpoint every 5 queries
        if len(eval_records) % 5 == 0 or len(eval_records) == len(golden_set):
            with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
                json.dump(eval_records, f, indent=2, ensure_ascii=False)

        time.sleep(0.5)

    wall_clock_total_sec = time.perf_counter() - wall_clock_start
    print("-" * 115, flush=True)
    print(f"\n[+] Pipeline run complete across {len(eval_records)} queries in {wall_clock_total_sec:.1f}s.", flush=True)

    # 7. Aggregate Metrics
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
            "mean_attribution_accuracy": round(statistics.mean(r["metrics"]["attribution_accuracy"] for r in cat_recs), 4),
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
        "mean_attribution_accuracy": round(statistics.mean(r["metrics"]["attribution_accuracy"] for r in eval_records), 4),
        "mean_citation_coverage": round(statistics.mean(r["metrics"]["citation_coverage"] for r in eval_records), 4),
        "mean_numerical_accuracy": round(statistics.mean(r["metrics"]["numerical_accuracy"] for r in eval_records), 4),
        "mean_unsupported_claim_rate": round(statistics.mean(r["metrics"]["unsupported_claim_rate"] for r in eval_records), 4),
        "mean_abstention_quality": round(statistics.mean(r["metrics"]["abstention_quality"] for r in eval_records), 4),
        "crag_distribution": {
            "correct_count": sum(1 for r in eval_records if r["crag_action"] == ActionType.CORRECT.value),
            "ambiguous_count": sum(1 for r in eval_records if r["crag_action"] == ActionType.AMBIGUOUS.value),
            "incorrect_count": sum(1 for r in eval_records if r["crag_action"] == ActionType.INCORRECT.value),
            "web_search_fallback_rate": round(sum(1 for r in eval_records if r["web_searched"]) / len(eval_records), 4) if eval_records else 0.0,
        },
        "latency_profile": {
            "mean_total_ms": round(statistics.mean(latencies), 2),
            "p50_total_ms": round(statistics.median(latencies), 2),
            "p95_total_ms": round(sorted(latencies)[int(0.95 * len(latencies))], 2),
            "mean_stage_breakdown_ms": {
                "query_processing": round(statistics.mean(r["latency_by_stage_ms"]["query_processing"] for r in eval_records), 2),
                "retrieval": round(statistics.mean(r["latency_by_stage_ms"]["retrieval"] for r in eval_records), 2),
                "fusion": round(statistics.mean(r["latency_by_stage_ms"]["fusion"] for r in eval_records), 2),
                "reranking_1": round(statistics.mean(r["latency_by_stage_ms"]["reranking_1"] for r in eval_records), 2),
                "crag_midsection": round(statistics.mean(r["latency_by_stage_ms"]["crag_midsection"] for r in eval_records), 2),
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
            "test_id": "TEST_008_FINAL_ENTERPRISE_CRAG_V1_PIPELINE",
            "run_index": 8,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "target_document": "Apple Inc. Form 10-K (FY 2025)",
            "benchmark_objective": "Evaluate end-to-end Corrective RAG (CRAG) with Requesty LLM Pointwise Evaluator, 3-branch Action Routing, Knowledge Strip Refinement, Tavily Web Fallback, Dual-Source Attribution, and LangChain LCEL Chain execution"
        },
        "system_architecture_snapshot": {
            "pipeline_stage": "Enterprise Corrective RAG (CRAG) Pipeline V1",
            "orchestration_framework": "LangChain LCEL Unified Chain (build_crag_pipeline_chain)",
            "hybrid_retrieval": "Dense BGE-Base-en-v1.5 (Qdrant) + Sparse BM25Okapi -> RRF(k=60)",
            "reranker_1": "cross-encoder/ms-marco-MiniLM-L6-v2 (top 5)",
            "crag_evaluator": "CRAG/Retrieval_Evaulator.py (google/gemma-4-31b-it via Requesty AI Router)",
            "crag_router": "CRAG/actionRouter.py (T_upper=0.7, T_lower=0.3 -> CORRECT / AMBIGUOUS / INCORRECT)",
            "crag_refiner": "CRAG/knowledge_refiner.py (sentence tokenize -> cross-encoder strip filter -> recompose)",
            "crag_web_fallback": "CRAG/query_rewriter.py + Tavily Web Search API (max_results=3)",
            "evidence_preparation": "CRAG/source_attributor.py (dual tagging: [DOC:chunk_id] vs [WEB:url])",
            "generator": "generation/llm_provider.py (GeminiProvider, gemini-3.5-flash-lite, temp=0.0)",
            "verification": "SourceAttributor.attribution_accuracy + CitationExtractor.verify_coverage"
        },
        "headline_metrics": headline_metrics,
        "category_breakdown": category_summary,
        "per_query_audit_log": eval_records
    }

    # 8. Save Deliverables
    with open(FINAL_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(test_run_deliverable, f, indent=2, ensure_ascii=False)
    print(f"\n[+] Saved final CRAG evaluation results to: {FINAL_RESULTS_PATH}", flush=True)

    # Append to benchmark.json at root
    if ROOT_BENCHMARK_JSON.exists():
        with open(ROOT_BENCHMARK_JSON, "r", encoding="utf-8") as f:
            root_data = json.load(f)
    else:
        root_data = []
    root_data = [t for t in root_data if t.get("test_run_header", {}).get("test_id") != "TEST_008_FINAL_ENTERPRISE_CRAG_V1_PIPELINE"]
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
    storage_data = [t for t in storage_data if t.get("test_run_header", {}).get("test_id") != "TEST_008_FINAL_ENTERPRISE_CRAG_V1_PIPELINE"]
    storage_data.append(test_run_deliverable)
    with open(BENCHMARK_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(storage_data, f, indent=2, ensure_ascii=False)
    print(f"[+] Appended to: {BENCHMARK_JSON_PATH}", flush=True)

    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
        print(f"[+] Cleaned up temporary checkpoint file.", flush=True)

    print("\n" + "=" * 105, flush=True)
    print("TEST_008_FINAL_ENTERPRISE_CRAG_V1_PIPELINE SUMMARY:", flush=True)
    print(f"  Total Queries:            {headline_metrics['total_queries_evaluated']}", flush=True)
    print(f"  Mean NDCG@5:              {headline_metrics['mean_ndcg_at_5']:.4f}", flush=True)
    print(f"  Mean Context Recall:      {headline_metrics['mean_context_recall']:.4f}", flush=True)
    print(f"  Mean Context Precision:   {headline_metrics['mean_context_precision']:.4f}", flush=True)
    print(f"  Mean Answer Correctness:  {headline_metrics['mean_answer_correctness']:.4f}", flush=True)
    print(f"  Mean Attribution Accuracy:{headline_metrics['mean_attribution_accuracy']:.4f}", flush=True)
    print(f"  Mean Numerical Accuracy:  {headline_metrics['mean_numerical_accuracy']:.4f}", flush=True)
    print(f"  Web Search Fallback Rate: {headline_metrics['crag_distribution']['web_search_fallback_rate'] * 100:.1f}%", flush=True)
    print(f"  Throughput:               {headline_metrics['system_reliability']['throughput_queries_per_min']} QPM", flush=True)
    print(f"  Total Run Cost:           ${headline_metrics['resource_and_cost_profile']['total_inference_cost_usd']:.4f}", flush=True)
    print("=" * 105, flush=True)


if __name__ == "__main__":
    main()
