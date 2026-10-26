from typing import Optional, Any
from pydantic import BaseModel, Field


# ==============================================================================
# REQUEST SCHEMAS
# ==============================================================================

class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=2,
        max_length=1500,
        description="The financial intelligence query or corporate disclosure question.",
        examples=["What were Apple's total net sales and Services revenue in fiscal 2025?"]
    )
    enable_crag: bool = Field(
        default=True,
        description="Toggle Corrective RAG (CRAG) self-evaluation and web fallback vs baseline RAG."
    )
    web_search_max_results: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum live web pages to retrieve via Tavily if fallback is triggered."
    )
    confidence_threshold_upper: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Upper threshold (T_H): scores >= T_H route to CORRECT (internal refinement)."
    )
    confidence_threshold_lower: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Lower threshold (T_L): scores <= T_L route to INCORRECT (web fallback)."
    )


# ==============================================================================
# GLASS-BOX STAGE INSPECTION SCHEMAS
# ==============================================================================

class CandidateChunk(BaseModel):
    chunk_id: str
    score: float
    content_snippet: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryProcessingStage(BaseModel):
    detected_type: str = Field(description="'factual', 'multi_hop', or 'summarization'")
    search_queries: list[str] = Field(description="Decomposed and expanded query variations sent to search stores.")
    latency_ms: float


class HybridRetrievalStage(BaseModel):
    dense_qdrant_candidates: list[CandidateChunk] = Field(description="Top semantic vector matches from Qdrant.")
    sparse_bm25_candidates: list[CandidateChunk] = Field(description="Top lexical keyword matches from BM25Okapi.")
    latency_ms: float


class FusionAndRerankStage(BaseModel):
    rrf_fused_chunk_ids: list[str] = Field(description="Top candidate chunk IDs ranked by Reciprocal Rank Fusion (k=60).")
    reranked_top5_candidates: list[CandidateChunk] = Field(description="Top-5 chunks selected after Cross-Encoder scoring.")
    fusion_latency_ms: float
    rerank_latency_ms: float


class WebSearchResult(BaseModel):
    title: str
    url: str
    content_snippet: str


class CRAGEngineStage(BaseModel):
    action: str = Field(description="Routing decision: 'CORRECT', 'AMBIGUOUS', or 'INCORRECT'")
    evaluator_scores: list[tuple[str, float]] = Field(description="Pointwise LLM relevance score per candidate chunk (0.0 to 1.0).")
    accepted_chunk_ids: list[str] = Field(description="Chunks meeting or exceeding confidence threshold.")
    discarded_chunk_ids: list[str] = Field(description="Chunks filtered out as irrelevant.")
    web_searched: bool = Field(description="True if external search was triggered (INCORRECT or AMBIGUOUS).")
    rewritten_web_query: Optional[str] = Field(default=None, description="Cleaned keyword search query used for Tavily search.")
    web_results: list[WebSearchResult] = Field(default_factory=list, description="External web evidence retrieved.")
    final_evidence_count: int = Field(description="Total chunks/strips prepared for generator context.")
    latency_ms: float


class GenerationStage(BaseModel):
    answer: str = Field(description="Final generated response citing [DOC:chunk_id] or [WEB:url].")
    formatted_context_preview: str = Field(description="XML context string injected into the generation prompt.")
    prompt_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    latency_ms: float


class AttributionAuditStage(BaseModel):
    attribution_accuracy: float = Field(description="Ratio of valid citations to total cited tags (1.0 = 100% precision).")
    valid_citations: list[str] = Field(description="Cited tags present in the retrieved context.")
    hallucinated_citations: list[str] = Field(description="Cited tags invented by the LLM without context backing.")
    internal_citation_count: int = Field(description="Number of [DOC:...] citations.")
    web_citation_count: int = Field(description="Number of [WEB:...] citations.")
    is_abstention: bool = Field(description="True if the model appropriately abstained from answering.")
    is_fully_grounded: bool
    latency_ms: float


class LatencyBreakdown(BaseModel):
    query_processing: float
    retrieval: float
    fusion: float
    reranking_1: float
    crag_midsection: float
    generation: float
    verification: float
    total: float


# ==============================================================================
# MAIN GLASS-BOX RESPONSE SCHEMA
# ==============================================================================

class GlassBoxQueryResponse(BaseModel):
    query: str
    target_corpus: str = "Apple Inc. Form 10-K (FY 2025)"
    answer: str
    crag_action: str
    attribution_accuracy: float
    
    # Detailed Glass-Box Inner Inspection
    stage_1_query_processing: QueryProcessingStage
    stage_2_hybrid_retrieval: HybridRetrievalStage
    stage_3_fusion_and_rerank: FusionAndRerankStage
    stage_4_crag_engine: CRAGEngineStage
    stage_5_generation: GenerationStage
    stage_6_attribution_audit: AttributionAuditStage
    
    # Latencies
    latencies_ms: LatencyBreakdown


# ==============================================================================
# HEALTH & METADATA SCHEMAS
# ==============================================================================

class HealthResponse(BaseModel):
    status: str = "healthy"
    target_document: str
    total_indexed_chunks: int
    dense_vector_dim: int
    models_ready: bool
    uptime_seconds: float


class SystemInfoResponse(BaseModel):
    system_name: str = "Enterprise Corrective RAG (CRAG) Intelligence API"
    version: str = "1.0.0"
    target_corpus: str = "Apple Inc. Form 10-K (FY 2025)"
    architecture: dict[str, Any]
    benchmarks_summary: dict[str, Any]
