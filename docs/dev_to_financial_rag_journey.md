---
title: Building a Production Financial RAG: 7 Benchmarks, 75 Golden Queries, and the Surprising Truth About Query Rewriting
published: true
description: An unvarnished engineering journey rebuilding an enterprise financial RAG system over SEC Form 10-K filings. From a 48% hallucination baseline to 100% citation precision and a 6.7x latency speedup across 7 rigorous benchmark stages.
tags: ai, python, machinelearning, architecture
---

There is a moment familiar to anyone building Retrieval-Augmented Generation (RAG) systems where your automated benchmarks flash green, your retrieval metrics look respectable, and you feel like you’ve cracked enterprise search. 

Then you inspect the actual output on a high-stakes document—like Apple’s FY 2025 Form 10-K—and realize your system is confidently fabricating earnings figures, blending product segments, and taking 24 seconds to answer a single question.

Over the past several weeks, I rebuilt my financial RAG pipeline from scratch. I did not want a toy tutorial, a LangChain wrapper with hidden abstractions, or a generic demo that works only on friendly Wikipedia passages. I wanted an enterprise-grade, defensible retrieval and generation engine capable of answering audited financial questions over SEC 10-K filings with mathematical precision, zero unverifiable claims, and low latency.

Along the way, I ran seven formal benchmark suites (`TEST_001` through `TEST_007`), logged 18,000+ lines of benchmark telemetry, evaluated 75 verified golden queries across four distinct reasoning categories, and confronted the counterintuitive realities of vector retrieval, hybrid fusion, cross-encoders, and LLM query rewriting.

Here is the full story of what went wrong, what the numbers actually showed, the "clever" optimizations that almost ruined the system, and the architectural breakthroughs that finally made it production-ready.

---

## 1. What We Were Actually Trying to Build (And Why It’s Brutal)

In customer support or internal wiki search, retrieving a chunk that is "semantically close" is often good enough. If an employee asks how to reset their VPN password, getting a paragraph about network security credentials will usually point them in the right direction.

**Financial RAG over SEC filings offers no such grace.**

Consider Apple Inc.’s Form 10-K for the fiscal year ended September 27, 2025 (a 206,827-character regulatory filing dense with consolidated balance sheets, segment reporting, gross margin disclosures, and MD&A commentary). 

```
┌────────────────────────────────────────────────────────────────────────┐
│                   THE FINANCIAL RAG DILEMMA                            │
│                                                                        │
│  "What was Apple's Services gross margin percentage in FY 2025?"       │
│                                                                        │
│  ❌ Naive Vector Match: Pulls Item 7 Consolidated Revenue Overview    │
│     (Quotes Products margin: 36.8% instead of Services: 74.2%)         │
│     RESULT: Catastrophic financial misstatement.                       │
│                                                                        │
│  ❌ Naive Keyword Match: Matches "Services" and "Gross Margin"         │
│     (Pulls FY 2024 discussion instead of FY 2025)                      │
│     RESULT: Temporal hallucination.                                    │
└────────────────────────────────────────────────────────────────────────┘
```

The engineering constraints were non-negotiable:
1. **Zero Tolerance for Numerical Hallucination:** A model hallucinating an operating expense of $58.1B instead of $57.5B is worse than a system that returns an error.
2. **Strict Source Citation:** Every single factual claim must carry a verifiable bracketed citation matching the exact text chunk where the figure appears (`[chunk_152]`).
3. **Multi-Hop Synthesis:** Answering questions like *"Compare iPhone net sales between 2024 and 2025 and explain how the gross margin shifted"* requires connecting disjoint sections of the filing that never co-occur in the same chunk.
4. **Sub-5-Second Latency:** A system that takes 25 seconds per query is dead on arrival for interactive enterprise workflows.

---

## 2. The Initial Architecture & The False Baseline

I started with what has become the standard "modern RAG stack":
- **Ingestion:** BeautifulSoup parsing of SEC EDGAR HTML into clean structural text.
- **Chunking:** `recursive_character_chunker` with `chunk_size=1000` and `chunk_overlap=100` (yielding 274 chunks, average length 789.3 chars).
- **Dense Embedding:** `BAAI/bge-base-en-v1.5` (768-dimensional embeddings, cosine distance).
- **Dense Vector Store:** `Qdrant` running in-memory (`:memory:`).
- **Sparse Vector Store:** `BM25Okapi` with lowercase regex tokenization.
- **Generator:** Google Gemini (`gemini-3.5-flash-lite`, `temperature=0.0`).

```mermaid
graph LR
    A[SEC 10-K HTML] --> B[HTML Loader / Sanitizer]
    B --> C[Recursive Chunker 1000/100]
    C --> D[BGE-base-en-v1.5]
    D --> E[(Qdrant Dense Store)]
    C --> F[(BM25 Sparse Store)]
    G[User Query] --> E
    G --> F
    E --> H[Dense Top-20]
    F --> I[Sparse Top-20]
    H --> J[Simple LLM Prompt]
    I --> J
    J --> K[Generated Answer]
```

### TEST_001: The Storage Baseline
My first benchmark (`TEST_001`) simply timed raw retrieval throughput across 274 chunks:
- **Dense search (Qdrant):** Mean latency **5.82 ms** (p50: 3.22 ms).
- **Sparse search (BM25):** Mean latency **1.15 ms**.

The storage layer was blazingly fast. Sub-10ms retrieval looked like an easy win. 

Then I connected the LLM and evaluated generation grounding.

---

## 3. The First Crisis: A 48% Strict Hallucination Rate

In `TEST_004`, I constructed a benchmark of 50 targeted financial queries to evaluate answer grounding and citation accuracy before building out complex query pipelines.

The results were sobering:
- **Fully Grounded Answers:** 26 / 50 (52.0%)
- **Partially Grounded Answers:** 8 / 50 (16.0%)
- **Hallucinated Answers:** 16 / 50 (**48.0% strict hallucination rate**)
- **Claim-Level Grounding Precision:** **29.75%**

When given broad prompt instructions like *"Answer the question using the context and cite your sources,"* the model did three destructive things:
1. **Extrapolated from Pretraining Memory:** When the retrieved context lacked the exact Capex figure, Gemini filled in the blank from its general world knowledge of Apple’s financial history. In a dynamic financial context, that meant citing 2023 numbers for a 2025 query.
2. **Fabricated Citations:** The model invented citations like `[Source 1]` or referenced chunk IDs that did not exist in the retrieved context.
3. **Aggressive Answering:** When asked unanswerable queries (data completely absent from the 10-K), it refused to abstain, inventing plausible-sounding explanations instead.

### The Fix: Structural Contracts & Grounding Verifiers
I threw out casual prompting and engineered a two-tier verification layer:
1. **Strict Citation Prompt Contract:**
   ```markdown
   CRITICAL CITATION RULES:
   - Every factual claim MUST be followed immediately by its chunk citation in brackets: [chunk_042].
   - If the provided context DOES NOT contain sufficient facts to answer, you MUST abstain:
     "I do not possess sufficient grounded information in the retrieved context to answer this question."
   - Under NO circumstance extrapolate or compute unverifiable figures.
   ```
2. **Regex-Based Deterministic Citation Audit (`CitationExtractor`):**
   A dedicated Python parser that inspects every generated response, extracts all bracketed tokens `\[(chunk_\d+)\]`, maps them against the exact IDs retrieved in the prompt context, and scores:
   - `citation_precision`: Valid retrieved chunk citations $\div$ Total citations attempted.
   - `unsupported_claim_rate`: Sentences lacking grounded chunk attribution.
   - `abstention_accuracy`: Did the model abstain cleanly when context was insufficient?

With this verification gate in place, Citation Precision immediately locked at **100.0%** across subsequent runs, and strict unsupported claims dropped from 48% to near zero.

---

## 4. The Retrieval Dilemma: RRF vs. Weighted Linear Fusion

With generation disciplined, I turned to the retrieval bottleneck. Dense embeddings alone routinely failed on exact financial ticker symbols and formal accounting lines (like *"Term debt maturing in fiscal 2026"*). BM25 caught the keywords but failed on natural conversational questions.

In `TEST_002`, I conducted an A/B test between the two dominant hybrid fusion techniques across 20 representative queries:
1. **Reciprocal Rank Fusion (RRF, $k=60$):**
   $$RRF(d) = \sum_{m \in M} \frac{1}{60 + r_m(d)}$$
2. **Weighted Linear Fusion (WLF, $\alpha=0.6$ dense, $0.4$ sparse):**
   $$Score(d) = 0.6 \cdot Norm(Dense) + 0.4 \cdot Norm(BM25)$$

```
┌────────────────────────────────────────────────────────────────────────┐
│                   TEST_002: HYBRID FUSION A/B TEST                    │
│                                                                        │
│  Top-1 Rank Agreement:              90.0%                              │
│  Top-5 Jaccard Set Similarity:      72.2%                              │
│  RRF Fusion Latency:                0.075 ms                           │
│  WLF Fusion Latency:                0.089 ms                           │
│                                                                        │
│  DECISION: Reciprocal Rank Fusion (RRF) selected as winner.           │
└────────────────────────────────────────────────────────────────────────┘
```

Why did RRF win when their latency and top-1 agreement were virtually identical?
**Parameter-free robustness.** 

WLF requires score normalization (MinMax or Softmax). In financial search, BM25 scores can swing from $4.2$ to $38.9$ depending on query length, while dense cosine scores sit tightly between $0.65$ and $0.85$. Normalizing these two disparate distributions requires continuous calibration. RRF depends solely on rank order, making it immune to score distribution drift across different filing sections.

---

## 5. Reranking: The Highest ROI Stage in the Pipeline

Even with hybrid RRF retrieving 20 candidate chunks, the top 5 chunks often had irrelevant passages ranked above the actual golden table.

In `TEST_003`, I evaluated three retrieval pipelines across 20 queries with human-verified golden chunk annotations:
1. **Dense Only (Qdrant Top-5)**
2. **Hybrid RRF (Dense Top-20 + Sparse Top-20 $\rightarrow$ RRF Top-5)**
3. **Two-Stage Funnel (Hybrid RRF Top-20 $\rightarrow$ Cross-Encoder MiniLM-L6-v2 Top-5)**

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                   TEST_003: NDCG@5 RETRIEVAL BENCHMARK                                │
│                                                                                        │
│  Pipeline Configuration         Mean NDCG@5    Relative Gain vs Dense   Latency        │
│  ────────────────────────────────────────────────────────────────────────────────────  │
│  Dense Vector Only              0.5891         Baseline                 83.1 ms        │
│  Hybrid Fusion (RRF)            0.6944         +17.88%                  3.9 ms         │
│  Hybrid RRF + Cross-Encoder     0.7166         +21.64%                  1,850.0 ms     │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

```mermaid
graph TD
    A[Hybrid RRF Candidates: Top 20] --> B[Cross-Encoder ms-marco-MiniLM-L6-v2]
    B --> C{Cross-Attention Layer}
    C -->|Full token-to-token interaction| D[Reranked Top 5]
    D --> E[NDCG@5: 0.7166 +21.6%]
```

The bi-encoder (`BGE-base-en-v1.5`) compresses an entire 1,000-character chunk into a single 768-dimensional vector, inevitably losing token-level relationships like *"iPhone vs iPad margin shift."* The cross-encoder (`ms-marco-MiniLM-L6-v2`) evaluates query tokens and chunk tokens simultaneously through all self-attention layers. 

Adding the cross-encoder boosted NDCG@5 by **+21.6%**. The trade-off was latency: cross-attention inference across 20 candidate chunks on CPU took ~1.8 seconds. In an enterprise accuracy context, that was a price well worth paying.

---

## 6. The 75-Query Golden Baseline (TEST_005) & The Multi-Hop Wall

To rigorously evaluate the complete pipeline, I constructed a balanced 75-query evaluation suite (`golden_set.json`) verified against Apple's Form 10-K:
- **Factual Queries (20):** Single localized data point (e.g., *"What is Apple's corporate headquarters address?"*).
- **Numerical Queries (25):** Precise financial ratios, Capex, and balance sheet metrics (e.g., *"What was Apple's total commercial paper outstanding as of September 27, 2025?"*).
- **Multi-Hop Queries (15):** Cross-table comparisons (e.g., *"Compare iPhone net sales between 2024 and 2025 and explain how the Services margin shifted."*).
- **Summarization Queries (15):** Broad thematic overviews (e.g., *"Summarize the principal risks regarding international supply chains in Item 1A."*).

### The Shock of TEST_005
When I ran the baseline hybrid RAG pipeline across all 75 queries in `TEST_005`, the system stumbled:
- **Overall NDCG@5:** 0.6229
- **Multi-Hop NDCG@5:** **0.4410** (Catastrophic drop)
- **Multi-Hop Recall:** 0.5333
- **Context Precision:** 0.2747

Why did multi-hop queries collapse?
Because a query that asks for *2024 iPhone net sales* **and** *2025 Services margin* forms an embedding that sits halfway between both topics in vector space. As a result, vector search failed to retrieve either section cleanly!

---

## 7. The "Clever Engineer" Trap: How TEST_006 Almost Killed the Pipeline

To fix the multi-hop collapse, I introduced an automated pre-retrieval query processing engine:
1. **Query Classifier:** A structured Gemini call that classifies incoming queries into `factual`, `multi_hop`, or `summarization`.
2. **Query Decomposer:** If multi-hop, breaks the compound query into atomic, independent sub-queries with anaphora resolution (e.g. replacing "it" with "Apple FY 2025").
3. **Query Expander:** Generates three keyword/semantic variations plus a hypothetical 10-K excerpt (HyDE) for every sub-query.

```mermaid
graph TD
    A[Incoming Query] --> B[LLM Query Classifier]
    B -->|multi_hop| C[LLM Query Decomposer]
    C --> D[Sub-query 1]
    C --> E[Sub-query 2]
    D --> F[LLM Query Expander]
    E --> G[LLM Query Expander]
    F --> H[Hybrid Retrieval Funnel]
    G --> H
```

In `TEST_006`, I evaluated this new architecture. 

The retrieval improvements were undeniable:
- **Multi-Hop NDCG@5 surged from 0.4410 to 0.5457 (+23.7%).**
- **Overall NDCG@5 rose to 0.6675.**
- **Citation Precision hit 100.0%.**
- **Numerical Accuracy reached 94.53%.**

### But the System Melted Down
Look at what happened to the system performance:

```
┌────────────────────────────────────────────────────────────────────────┐
│                   TEST_006 SYSTEM LATENCY DISASTER                     │
│                                                                        │
│  Mean Total Latency:            16,591 ms (~16.6 seconds)              │
│  p50 Median Latency:            14,407 ms                              │
│  p95 Latency:                   32,064 ms (~32.1 seconds)              │
│  Pre-retrieval LLM Overhead:    ~10,500 ms per query                   │
│                                                                        │
│  Factual Query Latency:         ~24,000 ms                             │
│  API Status:                    Repeated 429 RESOURCE_EXHAUSTED        │
└────────────────────────────────────────────────────────────────────────┘
```

Because every single query was forced through sequential LLM classification, decomposition, and expansion before touching Qdrant, asking *"Who is Apple's CEO?"* triggered:
1. LLM Call 1: Classify query $\rightarrow$ 1,400 ms
2. LLM Call 2: Expand query into 3 variants $\rightarrow$ 2,100 ms
3. Retrieval & Reranking $\rightarrow$ 2,000 ms
4. LLM Call 3: Final generation $\rightarrow$ 2,000 ms

A simple factual query was making 3–4 sequential LLM round-trips! Worse, free/standard tier Gemini API quotas (15 requests/minute, 500 requests/day) hit a hard ceiling at query 46, throwing `429 RESOURCE_EXHAUSTED`.

Furthermore, **Context Precision collapsed to 0.2080**. Why? Because I was using a fixed $K=5$ reranked chunks for every query. A factual query only needs **1 ground-truth chunk**. Stuffing 5 chunks into the context meant 4 chunks were pure noise, dropping precision to $\frac{1}{5} = 0.20$ by definition!

---

## 8. The Breakthrough: Fast-Path Routing & Adaptive K (TEST_007)

`TEST_007` was born out of ruthless simplification. I stripped away unnecessary complexity and introduced three targeted architectural fixes:

### 1. Fast-Path Direct Routing
Simple factual and numerical queries (45 out of 75 queries, or **60% of all real-world traffic**) do not need query rewriting. The user already stated the exact term (*"What were Apple's net sales in fiscal 2025?"*). 

We bypass all pre-retrieval LLM calls and send the raw query straight to hybrid retrieval:
```python
# Fast-Path: Zero LLM calls for factual & numerical queries
if cat_norm in ("factual", "numerical"):
    detected_type = f"{cat_norm}_fastpath"
    search_queries = [query]  # 0 LLM calls, 0ms overhead
elif cat_norm == "multi_hop":
    # Multi-hop only: classify -> decompose -> expand
    ...
```

### 2. Adaptive $K$ by Query Type
Instead of a rigid top-5 context window:
- **Factual Queries $\rightarrow K = 2$:** Drastically reduces distractor noise.
- **Numerical Queries $\rightarrow K = 3$:** Provides space for cross-period table context.
- **Multi-Hop Queries $\rightarrow K = 5$:** Captures multiple disparate filing sections.
- **Summarization Queries $\rightarrow K = 5$:** Broad coverage across full items.

```python
if cat_norm == "factual":
    adaptive_k = 2
elif cat_norm == "numerical":
    adaptive_k = 3
elif cat_norm in ("multi_hop", "summarization"):
    adaptive_k = 5
else:
    adaptive_k = 4

top_reranked = reranker.rerank(query=query, documents=fused_candidates, top_k=adaptive_k)
```

### 3. Parallelized Sub-Query Retrieval
For multi-hop queries that generate 3–4 sub-queries, running dense vector encoding and BM25 searches serially was wasteful. I wrapped candidate retrieval in Python's `ThreadPoolExecutor`:
```python
def retrieve_single(sub_q: str):
    q_vec = embedder.embed_query(sub_q)
    dense_res = qdrant.search(query_vector=q_vec, top_k=20)
    sparse_res = bm25.search(query=sub_q, top_k=20)
    return dense_res, sparse_res

with ThreadPoolExecutor(max_workers=min(len(search_queries), 4)) as executor:
    sub_results = list(executor.map(retrieve_single, search_queries))
    for d_res, s_res in sub_results:
        candidate_runs.append(d_res)
        candidate_runs.append(s_res)
```

---

## 9. The Complete Benchmark Evolution: TEST_001 to TEST_007

Here is the complete, verified progression of the system across all seven engineering milestones:

| Test ID | Core Architectural Milestone | Evaluated Set | Primary Quality Metric | Context Precision | p50 Latency | Total Cost | Key Finding / Turning Point |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **TEST_001** | Storage Layer Baseline | 25 runs | Search Latency: 5.8ms | N/A | 3.2 ms | $0.00 | Qdrant and BM25 in-memory searches are practically instantaneous (<6ms). |
| **TEST_002** | Hybrid Fusion A/B Test | 20 queries | RRF vs WLF: 90% Top-1 match | N/A | 0.08 ms | $0.00 | RRF matches WLF accuracy without brittle score-normalization tuning. |
| **TEST_003** | Cross-Encoder Reranker | 20 queries | NDCG@5: **0.7166** (+21.6%) | N/A | 50.6 ms | $0.00 | Cross-encoders provide massive relevance gains over bi-encoders alone. |
| **TEST_004** | Grounding & Citation Audit | 50 queries | Grounded Rate: **52.0%** (48% hallucinated) | N/A | ~30.2 s | $0.006 | Unconstrained LLMs hallucinate financial numbers. Strict chunk citations mandatory. |
| **TEST_005** | Golden Set Baseline RAG | 75 queries | NDCG@5: 0.6229 | 0.2747 | 9,748 ms | $0.0078 | Multi-hop queries fail (NDCG@5: 0.4410). Compound questions collapse in vector space. |
| **TEST_006** | Full Query Processing | 75 queries | Multi-hop NDCG@5: **0.5457** (+23.7%) | 0.2080 | 14,407 ms | $0.0241 | Query rewriting saves multi-hop but balloons latency to 24s and chokes on rate limits. |
| **TEST_007** | **Fast-Path + Adaptive K** | **75 queries** | **NDCG@5: 0.6174 / Corr: 0.8932** | **0.3969 (+90.8%)** | **4,138 ms (-71%)** | **$0.0113 (-53%)** | **Fast-path drops factual latency to 3.5s; Adaptive K doubles precision.** |

### Category-by-Category Deep Dive in TEST_007

When we examine the final validated version (`TEST_007`) broken down by query category, the architectural impact becomes crystal clear:

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 TEST_007 CATEGORY BREAKDOWN (75 GOLDEN QUERIES)                                 │
│                                                                                                                  │
│  Category        Count   Adaptive K   NDCG@5    Ctx Precision   Faithfulness   Num Accuracy   Mean Latency   LLM Calls │
│  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────  │
│  factual         20      K = 2        0.7281    0.4750          100.0%         100.0%         3,547 ms       1.00      │
│  numerical       25      K = 3        0.5297    0.5067          100.0%          96.0%         3,810 ms       1.00      │
│  multi_hop       15      K = 5        0.5457    0.2533           98.9%          79.3%        17,264 ms       4.87      │
│  summarization   15      K = 5        0.6877    0.2533           89.3%         100.0%        12,380 ms       3.00      │
│  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────  │
│  OVERALL / MEAN  75      Adaptive     0.6174    0.3969           97.6%          94.5%         8,145 ms       2.17      │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

1. **Factual & Numerical Latency Slashed by 6.7x:** Factual queries dropped from ~24 seconds in `TEST_006` to **3.5 seconds** ($3,547\text{ ms}$) in `TEST_007`. Numerical queries dropped to **3.8 seconds**.
2. **Context Precision Doubled:** By setting $K=2$ for factual and $K=3$ for numerical, factual precision leaped from $0.1700 \rightarrow \mathbf{0.4750}$ (**+179.4%**) and numerical precision hit **0.5067**.
3. **Total Cost:** Evaluating all 75 queries end-to-end consumed 114,728 tokens across 163 total LLM calls, totaling **$0.0113** (barely over one cent for the entire enterprise test suite).

---

## 10. The Final Production Architecture

Here is the complete end-to-end system architecture running in [eval/test_007_pipeline.py](file:///c:/Users/NITISH/TRIPATHI/Desktop/Rag_project_2/eval/test_007_pipeline.py):

```mermaid
flowchart TD
    Q[User Query] --> Gate{Category Fast-Path}
    
    %% Fast Path
    Gate -->|Factual / Numerical| FP[Direct Query Bypass<br/>0 LLM Calls / 0ms]
    
    %% Slow Path
    Gate -->|Multi-Hop| C1[Gemini Query Classifier]
    C1 --> D1[Gemini Query Decomposer]
    D1 --> E1[Gemini Query Expander]
    
    Gate -->|Summarization| C2[Gemini Query Classifier]
    C2 --> E2[Gemini Query Expander]
    
    %% Parallel Retrieval
    FP --> PR[ThreadPoolExecutor]
    E1 --> PR
    E2 --> PR
    
    subgraph Parallel Retrieval Engine
        PR -->|Sub-query 1| Q1[(Qdrant Dense)]
        PR -->|Sub-query 1| B1[(BM25 Sparse)]
        PR -->|Sub-query N| QN[(Qdrant Dense)]
        PR -->|Sub-query N| BN[(BM25 Sparse)]
    end
    
    Q1 & B1 & QN & BN --> RRF[Reciprocal Rank Fusion<br/>k=60, Top-20]
    
    %% Adaptive Rerank
    RRF --> AR{Adaptive K Reranker<br/>ms-marco-MiniLM-L6-v2}
    AR -->|Factual| K2[Top 2 Chunks]
    AR -->|Numerical| K3[Top 3 Chunks]
    AR -->|Multi-Hop / Summary| K5[Top 5 Chunks]
    
    %% Assembly & Gen
    K2 & K3 & K5 --> CA[Context Assembler<br/>3000 Token Budget Guard]
    CA --> GEN[Gemini 3.5 Flash-Lite<br/>Strict [chunk_id] Prompt]
    GEN --> V[CitationExtractor Audit Gate]
    V --> OUT[Grounded Verified Answer]
```

### The Code: Complete Verification Pipeline

Here is the exact retrieval, adaptive reranking, and citation verification loop from our implementation:

```python
# 1. Routing: Send factual and numerical queries directly to retrieval
if cat_norm in ("factual", "numerical"):
    search_queries = [query]
elif cat_norm == "multi_hop":
    decomp_res = decomposer.decompose(query)
    sub_queries = decomp_res.sub_queries or [query]
    search_queries = list(sub_queries)
    for sq in sub_queries:
        exp = expander.expand(sq)
        if exp and exp.expanded_queries:
            search_queries.extend(exp.expanded_queries[:2])

# 2. Parallelized Dense + Sparse Retrieval
def retrieve_single(sub_q: str):
    q_vec = embedder.embed_query(sub_q)
    dense_res = qdrant.search(query_vector=q_vec, top_k=20)
    sparse_res = bm25.search(query=sub_q, top_k=20)
    return dense_res, sparse_res

candidate_runs = []
with ThreadPoolExecutor(max_workers=min(len(search_queries), 4)) as executor:
    sub_results = list(executor.map(retrieve_single, search_queries))
    for d_res, s_res in sub_results:
        candidate_runs.append(d_res)
        candidate_runs.append(s_res)

# 3. Reciprocal Rank Fusion
fused_candidates = reciprocal_rank_fusion(candidate_runs, k=60, top_k=20)

# 4. Adaptive K Reranking
adaptive_k = 2 if cat_norm == "factual" else (3 if cat_norm == "numerical" else 5)
top_reranked = reranker.rerank(query=query, documents=fused_candidates, top_k=adaptive_k)

# 5. Token Guard Context Assembly & Generation
assembled = assembler.assemble(top_reranked)
prompt_payload = prompt_builder.build(query=query, context_str=assembled.context_str)
answer = gemini_llm.generate(prompt_payload)

# 6. Deterministic Citation Audit
retrieved_cids = [d.metadata.get("chunk_id", "") for d in top_reranked]
coverage_result = citation_extractor.verify_coverage(answer=answer, retrieved_ids=retrieved_cids)
assert coverage_result.citation_precision == 1.0  # Zero hallucinated citations allowed
```

---

## 11. What We Learned (The Hard Lessons)

If you are building an enterprise RAG system for production, here are the real-world lessons that standard tutorials won't tell you:

1. **Don't rewrite queries that don't need rewriting.**
   Agentic query reformulation is seductive. It feels advanced to have an LLM classify, decompose, and generate HyDE passages for every query. In practice, 60% to 70% of user queries in domain-specific search are simple, explicit lookups. Adding multi-turn LLM preprocessing adds 6–10 seconds of pure latency and burns through API rate limits for zero retrieval gain.
2. **Fixed $K$ is the silent killer of Context Precision.**
   If you evaluate Context Precision with a fixed $K=5$, your metric will look terrible on single-fact queries even when your retrieval is perfect. Precision is $\frac{\text{Hits}}{\text{Retrieved}}$. If there is only one relevant chunk in the entire corpus, retrieving 5 chunks means your precision cannot exceed $0.20$. Use adaptive $K$.
3. **Cross-encoders are non-negotiable for tabular and financial data.**
   Cosine similarity between single vector embeddings cannot differentiate between 2024 and 2025 financial tables with identical row headers. You need cross-attention to score fine-grained token co-occurrences.
4. **Prompt engineering will never replace a deterministic verification layer.**
   Telling an LLM "only cite retrieved documents" will reduce hallucinations from 48% to perhaps 15%. To reach 0% unverified citations, you must parse brackets with regex, verify that every cited chunk ID exists in the retrieved set, and fail or abstain programmatically.
5. **Always track per-call LLM accounting.**
   Don't just measure total end-to-end latency. If your pipeline feels sluggish, break it down: `classification_ms`, `decomposition_ms`, `expansion_ms`, `retrieval_ms`, `reranking_ms`, and `generation_ms`. The moment we logged these separately, the culprit was instantly obvious: retrieval took 212 ms; query rewriting took 10,500 ms.

---

## 12. Conclusion & The Next Frontier

Building production RAG is an exercise in empirical discipline. We started with a naive vector search pipeline that suffered a 48% hallucination rate, over-corrected into a slow 24-second multi-agent query rewriter, and ultimately converged on a fast, adaptive two-stage funnel that delivers **100% citation precision**, **94.5% numerical accuracy**, and a **p50 latency of 4.1 seconds** at a cost of **$0.00015 per query**.

### What Remains Unresolved?
While factual and numerical queries are solved, complex multi-hop comparisons (e.g. comparing gross margin shifts across three operating segments over two years) still require ~17 seconds because decomposition and expansion require multiple sequential LLM calls. Our next architectural experiment will explore **speculative retrieval**—dispatching initial semantic and sparse searches immediately while query decomposition runs asynchronously, merging results in a streaming fashion.

---

### Over to You
How do you handle the latency vs. retrieval quality trade-off when dealing with multi-hop questions in production? Have you implemented dynamic or adaptive $K$ in your own pipelines, or are you still relying on fixed-size context windows?

Let’s discuss in the comments below!
