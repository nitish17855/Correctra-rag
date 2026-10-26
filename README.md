---
title: AURA CRAG — Enterprise Financial Intelligence
emoji: ⚡
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: 4.26.0
app_file: app.py
pinned: false
---

# AURA CRAG — Enterprise Financial Intelligence & Glass-Box Studio

> Production-grade **Corrective RAG (CRAG)** intelligence system indexing **Apple Inc. (AAPL) Form 10-K (FY 2025)** with dual-threshold adaptive routing, dynamic query rewriting, live Tavily web fallback, and a full 5-stage glass-box inspection studio.

---

## 🏛️ System Architecture

AURA CRAG implements an enterprise 5-layer retrieval and synthesis state machine:

1. **Query Processing Layer:** Intent classification (`factual`, `numerical`, `multi-hop`, `comparative`) and atomic sub-question decomposition.
2. **Dual Hybrid Retrieval:**
   - **Dense Semantic:** In-memory `Qdrant` with `BAAI/bge-base-en-v1.5` embeddings (768-dim, cosine distance).
   - **Sparse Lexical:** In-memory `BM25Okapi` with regex alphanumeric tokenization.
3. **Fusion & Cross-Encoder Reranker:**
   - **Reciprocal Rank Fusion (RRF, $k=60$):** Merges dense and sparse candidate rankings.
   - **Cross-Encoder:** `cross-encoder/ms-marco-MiniLM-L6-v2` selects top 5 high-confidence chunks.
4. **3-Branch Corrective RAG (CRAG) Engine:**
   - **Pointwise Evaluator:** `google/gemma-4-31b-it` evaluates chunk relevance on an absolute $0.0 - 1.0$ scale.
   - **Dual-Threshold Router:** $T_{\text{High}} = 0.70$, $T_{\text{Low}} = 0.30$.
     - **CORRECT ($\ge 0.70$):** Sentence strip decomposition (2-sentence strips) & Cross-Encoder filtering.
     - **AMBIGUOUS ($0.30 - 0.70$):** Refined internal strips + Tavily web search results merged and joint Cross-Encoder reranked.
     - **INCORRECT ($\le 0.30$):** Internal docs discarded $\to$ `QueryRewriter` generates corporate keyword query $\to$ Tavily search $\to$ Cross-Encoder web rerank.
5. **Context Assembly, Generation & Attribution Audit:**
   - **Generator:** `Google Gemini 3.5 Flash-Lite` (temperature $0.0$).
   - **Dual Attribution:** Enforces `[DOC:chunk_id]` and `[WEB:url]` citation tags.
   - **Audit Auditor:** Resolves every citation against context and computes the `attribution_accuracy` metric.

---

## 📊 Benchmark Milestones (85 Golden Queries)

- **Answer Correctness:** 88.60%
- **Numerical Accuracy:** 93.39%
- **Attribution Accuracy:** 77.65% (100.0% on out-of-corpus web search)
- **Web Fallback Rate:** 16.5% (14 / 85 queries triggered live web search)
- **Pipeline Execution Error Rate:** 0.0%

---

## 🚀 Running Locally

```bash
# 1. Clone repository
git clone https://github.com/your-username/aura-crag.git
cd aura-crag

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment variables in .env
GEMINI_API_KEY=your_key
REQUESTY_API_KEY=your_key
TAVILY_API_KEY=your_key

# 4. Start full-stack application
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Visit `http://localhost:8000` to interact with the Split-Screen Studio.
