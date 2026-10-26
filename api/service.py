import time
from typing import Any
from pathlib import Path
import tiktoken

from langchain_core.documents import Document

from api.schemas import (
    QueryRequest,
    GlassBoxQueryResponse,
    CandidateChunk,
    QueryProcessingStage,
    HybridRetrievalStage,
    FusionAndRerankStage,
    WebSearchResult,
    CRAGEngineStage,
    GenerationStage,
    AttributionAuditStage,
    LatencyBreakdown,
)

# Pricing for Gemini 3.5 Flash-Lite (USD per token)
COST_INPUT_PER_TOKEN = 0.075 / 1_000_000
COST_OUTPUT_PER_TOKEN = 0.300 / 1_000_000


def run_glassbox_crag_pipeline(
    request: QueryRequest,
    pipeline_context: dict[str, Any],
) -> GlassBoxQueryResponse:
    """
    Executes the complete financial intelligence RAG pipeline and returns
    a fully transparent, glass-box audit response displaying the intermediate
    artifacts of every single stage.
    """
    t_start = time.perf_counter()
    query = request.query.strip()

    # Unpack pipeline resources from application state
    embedder = pipeline_context["embedder"]
    qdrant = pipeline_context["qdrant"]
    bm25 = pipeline_context["bm25"]
    classifier = pipeline_context["classifier"]
    decomposer = pipeline_context["decomposer"]
    expander = pipeline_context["expander"]
    reranker = pipeline_context["reranker"]
    crag_orchestrator = pipeline_context["crag_orchestrator"]
    prompt_builder = pipeline_context["prompt_builder"]
    gemini_llm = pipeline_context["gemini_llm"]
    citation_extractor = pipeline_context["citation_extractor"]
    source_attributor = pipeline_context["source_attributor"]
    tokenizer: tiktoken.Encoding = pipeline_context["tokenizer"]

    # ==========================================================================
    # STAGE 1: Query Processing (Classification -> Decomposition -> Expansion)
    # ==========================================================================
    t_qp_0 = time.perf_counter()
    query_class = classifier.classify(query)
    detected_type = query_class.query_type

    search_queries: list[str] = []
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

    # Ensure search queries are deduplicated
    unique_search_queries = list(dict.fromkeys(search_queries))
    t_qp_ms = (time.perf_counter() - t_qp_0) * 1000.0

    stage_1_result = QueryProcessingStage(
        detected_type=detected_type,
        search_queries=unique_search_queries,
        latency_ms=round(t_qp_ms, 2),
    )

    # ==========================================================================
    # STAGE 2: Hybrid Retrieval (Dense Qdrant + Sparse BM25)
    # ==========================================================================
    t_ret_0 = time.perf_counter()
    candidate_runs: list[list[Document]] = []
    dense_inspect_candidates: list[CandidateChunk] = []
    sparse_inspect_candidates: list[CandidateChunk] = []

    for sq in unique_search_queries:
        q_vec = embedder.embed_query(sq)
        dense_res = qdrant.search(query_vector=q_vec, top_k=20)
        sparse_res = bm25.search(query=sq, top_k=20)
        candidate_runs.append(dense_res)
        candidate_runs.append(sparse_res)

        # Collect top candidates for glass-box inspection
        for d in dense_res[:3]:
            cid = d.metadata.get("chunk_id", "unknown")
            score = float(d.metadata.get("score", 0.0))
            snippet = d.page_content[:180].replace("\n", " ").strip() + "..."
            dense_inspect_candidates.append(
                CandidateChunk(chunk_id=cid, score=score, content_snippet=snippet, metadata=d.metadata)
            )

        for d in sparse_res[:3]:
            cid = d.metadata.get("chunk_id", "unknown")
            score = float(d.metadata.get("bm25_score", 0.0))
            snippet = d.page_content[:180].replace("\n", " ").strip() + "..."
            sparse_inspect_candidates.append(
                CandidateChunk(chunk_id=cid, score=score, content_snippet=snippet, metadata=d.metadata)
            )

    t_ret_ms = (time.perf_counter() - t_ret_0) * 1000.0

    # Deduplicate inspect candidates by chunk_id
    dense_unique = {c.chunk_id: c for c in dense_inspect_candidates}.values()
    sparse_unique = {c.chunk_id: c for c in sparse_inspect_candidates}.values()

    stage_2_result = HybridRetrievalStage(
        dense_qdrant_candidates=list(dense_unique)[:5],
        sparse_bm25_candidates=list(sparse_unique)[:5],
        latency_ms=round(t_ret_ms, 2),
    )

    # ==========================================================================
    # STAGE 3: Reciprocal Rank Fusion (RRF) & Cross-Encoder Reranking
    # ==========================================================================
    from fusion.rrf import reciprocal_rank_fusion

    t_fusion_0 = time.perf_counter()
    fused_candidates = reciprocal_rank_fusion(candidate_runs, k=60, top_k=20)
    t_fusion_ms = (time.perf_counter() - t_fusion_0) * 1000.0

    t_rerank_0 = time.perf_counter()
    top5_reranked = reranker.rerank(query=query, documents=fused_candidates, top_k=5)
    t_rerank_ms = (time.perf_counter() - t_rerank_0) * 1000.0

    reranked_inspect = [
        CandidateChunk(
            chunk_id=d.metadata.get("chunk_id", "unknown"),
            score=round(float(d.metadata.get("rerank_score", 0.0)), 4),
            content_snippet=d.page_content[:180].replace("\n", " ").strip() + "...",
            metadata=d.metadata,
        )
        for d in top5_reranked
    ]

    stage_3_result = FusionAndRerankStage(
        rrf_fused_chunk_ids=[d.metadata.get("chunk_id", "") for d in fused_candidates[:8]],
        reranked_top5_candidates=reranked_inspect,
        fusion_latency_ms=round(t_fusion_ms, 2),
        rerank_latency_ms=round(t_rerank_ms, 2),
    )

    # ==========================================================================
    # STAGE 4: Corrective RAG (CRAG) Mid-Section Engine
    # ==========================================================================
    t_crag_0 = time.perf_counter()
    if request.enable_crag:
        # Dynamically set threshold parameters from user request if customized
        crag_orchestrator.router.upper_threshold = request.confidence_threshold_upper
        crag_orchestrator.router.lower_threshold = request.confidence_threshold_lower
        crag_orchestrator.web_search_max_results = request.web_search_max_results

        crag_res = crag_orchestrator.orchestrate(query=query, documents=top5_reranked)

        action = crag_res.action
        evaluator_scores = crag_res.evaluator_scores
        accepted_chunk_ids = crag_res.accepted_chunk_ids
        discarded_chunk_ids = crag_res.discarded_chunk_ids
        web_searched = crag_res.web_searched
        rewritten_web_query = crag_res.rewritten_web_query
        final_documents = crag_res.final_documents
        formatted_context = crag_res.formatted_context

        web_results_inspect = []
        for d in final_documents:
            if d.metadata.get("is_web"):
                web_results_inspect.append(
                    WebSearchResult(
                        title=d.metadata.get("title", "Web Result"),
                        url=d.metadata.get("source", "https://..."),
                        content_snippet=d.page_content[:200].replace("\n", " ") + "...",
                    )
                )
    else:
        # Baseline RAG Fallback
        action = "CRAG_DISABLED_BASELINE"
        evaluator_scores = []
        accepted_chunk_ids = [d.metadata.get("chunk_id", "") for d in top5_reranked]
        discarded_chunk_ids = []
        web_searched = False
        rewritten_web_query = None
        web_results_inspect = []
        final_documents = top5_reranked
        formatted_context = source_attributor.format_context(final_documents)

    t_crag_ms = (time.perf_counter() - t_crag_0) * 1000.0

    stage_4_result = CRAGEngineStage(
        action=action,
        evaluator_scores=evaluator_scores,
        accepted_chunk_ids=accepted_chunk_ids,
        discarded_chunk_ids=discarded_chunk_ids,
        web_searched=web_searched,
        rewritten_web_query=rewritten_web_query,
        web_results=web_results_inspect,
        final_evidence_count=len(final_documents),
        latency_ms=round(t_crag_ms, 2),
    )

    # ==========================================================================
    # STAGE 5: Context Assembly & Generator (Gemini Flash-Lite)
    # ==========================================================================
    t_gen_0 = time.perf_counter()
    prompt_payload = prompt_builder.build(query=query, context_str=formatted_context)
    answer = gemini_llm.generate(prompt_payload)

    prompt_tokens = len(tokenizer.encode(prompt_payload.system_prompt + "\n" + prompt_payload.user_prompt))
    output_tokens = len(tokenizer.encode(answer))
    cost_usd = (prompt_tokens * COST_INPUT_PER_TOKEN) + (output_tokens * COST_OUTPUT_PER_TOKEN)
    t_gen_ms = (time.perf_counter() - t_gen_0) * 1000.0

    stage_5_result = GenerationStage(
        answer=answer,
        formatted_context_preview=formatted_context[:400] + "\n...[truncated for preview]",
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        estimated_cost_usd=round(cost_usd, 6),
        latency_ms=round(t_gen_ms, 2),
    )

    # ==========================================================================
    # STAGE 6: Citation Extraction & Attribution Audit
    # ==========================================================================
    t_verif_0 = time.perf_counter()
    doc_cids = [d.metadata.get("chunk_id", "") for d in final_documents if d.metadata.get("chunk_id")]
    coverage_result = citation_extractor.verify_coverage(answer=answer, retrieved_ids=doc_cids)

    attribution_metrics = source_attributor.attribution_accuracy(
        answer=answer,
        retrieved_docs=final_documents,
    )
    t_verif_ms = (time.perf_counter() - t_verif_0) * 1000.0

    stage_6_result = AttributionAuditStage(
        attribution_accuracy=round(attribution_metrics.attribution_accuracy, 4),
        valid_citations=[r.raw_tag for r in attribution_metrics.resolved_citations if r.is_valid],
        hallucinated_citations=[r.raw_tag for r in attribution_metrics.resolved_citations if not r.is_valid],
        internal_citation_count=attribution_metrics.internal_citation_count,
        web_citation_count=attribution_metrics.web_citation_count,
        is_abstention=coverage_result.is_abstention,
        is_fully_grounded=coverage_result.is_fully_grounded,
        latency_ms=round(t_verif_ms, 2),
    )

    t_total_ms = (time.perf_counter() - t_start) * 1000.0

    latencies = LatencyBreakdown(
        query_processing=round(t_qp_ms, 2),
        retrieval=round(t_ret_ms, 2),
        fusion=round(t_fusion_ms, 2),
        reranking_1=round(t_rerank_ms, 2),
        crag_midsection=round(t_crag_ms, 2),
        generation=round(t_gen_ms, 2),
        verification=round(t_verif_ms, 2),
        total=round(t_total_ms, 2),
    )

    return GlassBoxQueryResponse(
        query=query,
        target_corpus="Apple Inc. Form 10-K (FY 2025)",
        answer=answer,
        crag_action=action,
        attribution_accuracy=round(attribution_metrics.attribution_accuracy, 4),
        stage_1_query_processing=stage_1_result,
        stage_2_hybrid_retrieval=stage_2_result,
        stage_3_fusion_and_rerank=stage_3_result,
        stage_4_crag_engine=stage_4_result,
        stage_5_generation=stage_5_result,
        stage_6_attribution_audit=stage_6_result,
        latencies_ms=latencies,
    )
