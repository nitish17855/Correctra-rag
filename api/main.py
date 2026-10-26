import os
import sys
import time
from pathlib import Path
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
import numpy as np
import tiktoken

# Ensure workspace root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

# Project Core Modules
from chunker.recursive_character_chunker import recursive_splitters
from Embedding.bge_embeddings import BGEEmbedding
from storage.Qdrant import QdrantStore
from storage.BM25 import BM25Store
from query_processing.query_classification import QueryClassifier
from query_processing.query_decomposer import Query_decomposer
from query_processing.query_expander import QueryExpander
from reranker.base import get_reranker
from generation.prompt_builder import PromptBuilder
from generation.llm_provider import GeminiProvider
from generation.citation_Extractor import CitationExtractor

# CRAG Core Engine Modules
from CRAG.orchestrate import CRAGOrchestrator
from CRAG.source_attributor import (
    SourceAttributor,
    CRAG_SYSTEM_PROMPT,
    CRAG_USER_PROMPT_TEMPLATE,
)

# API Schemas & Service
from api.schemas import (
    QueryRequest,
    GlassBoxQueryResponse,
    HealthResponse,
    SystemInfoResponse,
)
from api.service import run_glassbox_crag_pipeline

# File Paths
DATA_DIR = PROJECT_ROOT / "data"
CACHE_FILE = DATA_DIR / "aapl_clean_text.txt"
VECTORS_CACHE_FILE = DATA_DIR / "aapl_bge_vectors.npy"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    ASGI Lifespan Manager:
    Executes once on server startup to warm up and load all heavy ML assets
    (BGE embeddings, Qdrant in-memory, BM25, CrossEncoder, CRAG Orchestrator)
    into global application memory (app.state), avoiding per-request initialization.
    """
    print("=" * 80, flush=True)
    print("[*] FASTAPI STARTUP: Initializing Enterprise CRAG Intelligence Engine...", flush=True)
    print("=" * 80, flush=True)

    app.state.start_time = time.time()
    t_boot_0 = time.perf_counter()

    # 1. Corpus Text & Document Chunks
    if not CACHE_FILE.exists():
        raise FileNotFoundError(f"Required corpus file missing: {CACHE_FILE}")

    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        clean_text = f.read()

    docs = recursive_splitters(clean_text, chunk_size=1000, chunk_overlap=100)
    for idx, d in enumerate(docs):
        d.metadata["chunk_id"] = f"chunk_{idx:03d}"
        d.metadata["source"] = "aapl-20250927.htm"
        d.metadata["ticker"] = "AAPL"
        d.metadata["fiscal_year"] = "2025"
    app.state.total_chunks = len(docs)
    print(f"[+] Loaded and indexed {len(docs)} chunks from SEC 10-K filing.", flush=True)

    # 2. Dense Embeddings & Vector Storage
    embedder = BGEEmbedding()
    if VECTORS_CACHE_FILE.exists():
        print(f"[*] Loading precomputed embeddings from cache: {VECTORS_CACHE_FILE}", flush=True)
        cached_vectors = np.load(VECTORS_CACHE_FILE)
        vectors = cached_vectors.tolist()
    else:
        print("[*] Generating BGE embeddings...", flush=True)
        chunk_texts = [d.page_content for d in docs]
        vectors = embedder.embed_documents(chunk_texts)
        np.save(VECTORS_CACHE_FILE, np.array(vectors, dtype=np.float32))

    qdrant = QdrantStore(collection_name="aapl_production", vector_size=embedder.dimension, location=":memory:")
    qdrant.upsert_documents(documents=docs, vectors=vectors)

    # 3. Sparse BM25 Lexical Store
    bm25 = BM25Store()
    bm25.index_documents(documents=docs)

    # 4. Query Processing Models
    classifier = QueryClassifier(model_name="gemini-3.5-flash-lite")
    decomposer = Query_decomposer(model_name="gemini-3.5-flash-lite")
    expander = QueryExpander(model_name="gemini-3.5-flash-lite")

    # 5. Reranker & CRAG Orchestration
    reranker = get_reranker("local")
    crag_orchestrator = CRAGOrchestrator(reranker=reranker)

    # 6. Generator & Citation Verifier
    prompt_builder = PromptBuilder(
        system_prompt=CRAG_SYSTEM_PROMPT,
        user_template=CRAG_USER_PROMPT_TEMPLATE,
    )
    gemini_llm = GeminiProvider(model_name="gemini-3.5-flash-lite")
    citation_extractor = CitationExtractor()
    source_attributor = SourceAttributor()
    tokenizer = tiktoken.get_encoding("cl100k_base")

    # Store loaded resources in app.state for request handlers
    app.state.pipeline_context = {
        "embedder": embedder,
        "qdrant": qdrant,
        "bm25": bm25,
        "classifier": classifier,
        "decomposer": decomposer,
        "expander": expander,
        "reranker": reranker,
        "crag_orchestrator": crag_orchestrator,
        "prompt_builder": prompt_builder,
        "gemini_llm": gemini_llm,
        "citation_extractor": citation_extractor,
        "source_attributor": source_attributor,
        "tokenizer": tokenizer,
    }

    t_boot_sec = time.perf_counter() - t_boot_0
    print(f"[+] All models, stores, and CRAG chains initialized in {t_boot_sec:.2f}s.", flush=True)
    print("[+] Server ready to accept incoming production queries!", flush=True)

    yield

    # Shutdown logic
    print("[*] FASTAPI SHUTDOWN: Releasing resources and clearing caches...", flush=True)
    app.state.pipeline_context.clear()
    print("[+] Shutdown complete.", flush=True)


app = FastAPI(
    title="Enterprise Corrective RAG (CRAG) Intelligence API",
    description=(
        "Production financial intelligence backend serving Apple Inc. Form 10-K (FY 2025) disclosures. "
        "Implements hybrid dense/sparse retrieval, Reciprocal Rank Fusion, Cross-Encoder reranking, "
        "and a 3-state Corrective RAG (CRAG) self-evaluating engine with Tavily web search fallback."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for external frontends (Streamlit, React, Next.js, Vite)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# ROUTE HANDLERS
# ==============================================================================

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System Health"],
    summary="Health check probe for container orchestrators"
)
async def health_check(request: Request):
    """
    Returns server health, uptime, and index readiness.
    Used by Kubernetes, AWS ECS, or Render health probes.
    """
    uptime = time.time() - getattr(request.app.state, "start_time", time.time())
    models_ready = bool(getattr(request.app.state, "pipeline_context", None))
    total_chunks = getattr(request.app.state, "total_chunks", 0)

    return HealthResponse(
        status="healthy" if models_ready else "initializing",
        target_document="Apple Inc. Form 10-K (FY 2025)",
        total_indexed_chunks=total_chunks,
        dense_vector_dim=768,
        models_ready=models_ready,
        uptime_seconds=round(uptime, 2),
    )


@app.get(
    "/api/v1/info",
    response_model=SystemInfoResponse,
    tags=["System Info"],
    summary="Detailed architecture specifications and benchmark achievements"
)
async def system_info():
    """
    Returns comprehensive metadata about the underlying CRAG architecture,
    dual-threshold parameters, and historical benchmark milestones.
    """
    return SystemInfoResponse(
        system_name="Enterprise Corrective RAG (CRAG) Intelligence API",
        version="1.0.0",
        target_corpus="Apple Inc. Form 10-K (FY 2025)",
        architecture={
            "retrieval_dense": "BAAI/bge-base-en-v1.5 (768-dim, cosine distance, Qdrant in-memory)",
            "retrieval_sparse": "BM25Okapi (regex alphanumeric tokenizer, in-memory index)",
            "hybrid_fusion": "Reciprocal Rank Fusion (RRF, k=60, top-20 candidates)",
            "initial_reranker": "cross-encoder/ms-marco-MiniLM-L6-v2 (top-5 candidate selection)",
            "crag_evaluator": "google/gemma-4-31b-it (pointwise score via Requesty AI Router)",
            "crag_routing_thresholds": {"upper_T_H": 0.7, "lower_T_L": 0.3},
            "crag_knowledge_refinement": "Sentence tokenizer -> 2-sentence strips -> CrossEncoder strip filter",
            "crag_web_fallback": "QueryRewriter LCEL chain -> Tavily Search API",
            "generator": "Google Gemini 3.5 Flash-Lite (temperature=0.0)",
            "attribution_verification": "Dual-source citation auditor ([DOC:chunk_id] vs [WEB:url])"
        },
        benchmarks_summary={
            "latest_test_id": "TEST_008_FINAL_ENTERPRISE_CRAG_V1_PIPELINE",
            "evaluated_queries": 85,
            "mean_answer_correctness": 0.8860,
            "mean_numerical_accuracy": 0.9339,
            "web_search_fallback_rate": "16.5%",
            "out_of_corpus_attribution_accuracy": "100.0%",
            "pipeline_error_rate": "0.0%"
        }
    )


@app.post(
    "/api/v1/query",
    response_model=GlassBoxQueryResponse,
    tags=["Financial Intelligence Query"],
    summary="Run query through the complete Glass-Box CRAG pipeline"
)
async def query_crag_pipeline(
    query_request: QueryRequest,
    request: Request,
):
    """
    Main entry point for user financial queries.
    Returns the final synthesized answer along with a full 'Glass-Box' audit:
    - Stage 1: Query classification & decomposition
    - Stage 2: Qdrant vs. BM25 top candidates
    - Stage 3: RRF fusion & Cross-Encoder initial ranks
    - Stage 4: CRAG evaluation scores, routing decision, strip refinement / web fallback
    - Stage 5: Generation context & token cost
    - Stage 6: Attribution accuracy and hallucinated citation verification
    """
    pipeline_context = getattr(request.app.state, "pipeline_context", None)
    if not pipeline_context:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pipeline models are still initializing. Please retry in a few seconds.",
        )

    try:
        # Offload CPU-bound inference (BGE, Cross-Encoder) to worker thread pool
        # to ensure the asyncio event loop remains non-blocking for other clients.
        response = await run_in_threadpool(
            run_glassbox_crag_pipeline,
            request=query_request,
            pipeline_context=pipeline_context,
        )
        return response
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline execution error: {str(e)}",
        )


# Mount static frontend assets directly for full-stack unified deployment
FRONTEND_DIR = PROJECT_ROOT / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
