/**
 * AURA CRAG — Enterprise Financial Intelligence & Glass-Box Studio
 * Frontend Client & Interactive State Machine
 * Fully synchronized with api/schemas.py (GlassBoxQueryResponse)
 */

document.addEventListener('DOMContentLoaded', () => {
  // --- DOM Elements ---
  const backendStatusEl = document.getElementById('backend-status');
  const statusTextEl = document.getElementById('status-text');
  const corpusStatEl = document.querySelector('.corpus-stat');
  const queryForm = document.getElementById('query-form');
  const queryInput = document.getElementById('query-input');
  const toggleCrag = document.getElementById('toggle-crag');
  const selectTopK = document.getElementById('select-top-k');
  const btnSubmit = document.getElementById('btn-submit');
  const emptyStateEl = document.getElementById('empty-state');
  const answerCardEl = document.getElementById('answer-card');
  const answerBodyEl = document.getElementById('answer-body');
  const cragActionBadge = document.getElementById('crag-action-badge');
  const citationChipsEl = document.getElementById('citation-chips');
  const btnCopyContext = document.getElementById('btn-copy-context');

  // Metrics Bar
  const metricLatency = document.getElementById('metric-latency');
  const metricAttribution = document.getElementById('metric-attribution');
  const metricTokens = document.getElementById('metric-tokens');
  const metricCost = document.getElementById('metric-cost');

  // Stage 1 Elements
  const stg1QueryType = document.getElementById('stg1-query-type');
  const stg1QueryLen = document.getElementById('stg1-query-len');
  const stg1Subqueries = document.getElementById('stg1-subqueries');
  const stg1Expanded = document.getElementById('stg1-expanded');

  // Stage 2 Elements
  const stg2DenseCount = document.getElementById('stg2-dense-count');
  const stg2DenseList = document.getElementById('stg2-dense-list');
  const stg2SparseCount = document.getElementById('stg2-sparse-count');
  const stg2SparseList = document.getElementById('stg2-sparse-list');

  // Stage 3 Elements
  const stg3RerankedList = document.getElementById('stg3-reranked-list');
  const stg3RrfPool = document.getElementById('stg3-rrf-pool');

  // Stage 4 Elements
  const stg4DecisionBanner = document.getElementById('stg4-decision-banner');
  const stg4ActionText = document.getElementById('stg4-action-text');
  const stg4ConfidenceVal = document.getElementById('stg4-confidence-val');
  const stg4EvalScores = document.getElementById('stg4-eval-scores');
  const stg4DynamicHeader = document.getElementById('stg4-dynamic-header');
  const stg4DynamicBody = document.getElementById('stg4-dynamic-body');

  // Stage 5 Elements
  const stg5VerifiedCites = document.getElementById('stg5-verified-cites');
  const stg5HallucinatedCites = document.getElementById('stg5-hallucinated-cites');
  const stg5AttributionScore = document.getElementById('stg5-attribution-score');
  const stg5LatencyWaterfall = document.getElementById('stg5-latency-waterfall');
  const stg5FinalContext = document.getElementById('stg5-final-context');

  // --- Defensive Formatting Helpers ---
  function formatNum(val, decimals = 2, fallback = '--') {
    return (typeof val === 'number' && !isNaN(val)) ? val.toFixed(decimals) : fallback;
  }

  function formatPct(val, decimals = 1, fallback = '--%') {
    return (typeof val === 'number' && !isNaN(val)) ? (val * 100).toFixed(decimals) + '%' : fallback;
  }

  function formatCurrency(val, decimals = 5, fallback = '< $0.0001') {
    return (typeof val === 'number' && !isNaN(val)) ? '$' + val.toFixed(decimals) : fallback;
  }

  // --- Health Check Polling ---
  async function checkBackendHealth() {
    try {
      const res = await fetch('/health');
      if (res.ok) {
        const data = await res.json();
        backendStatusEl.className = 'status-pill status-ready';
        const totalChunks = data.total_indexed_chunks || 274;
        statusTextEl.textContent = `Engine Online • Qdrant Ready (${totalChunks} chunks)`;
        if (corpusStatEl) {
          corpusStatEl.textContent = `${totalChunks} Chunks • BGE + BM25`;
        }
      } else {
        throw new Error('Health check returned non-200');
      }
    } catch (err) {
      backendStatusEl.className = 'status-pill status-checking';
      statusTextEl.textContent = 'Connecting to Engine...';
      setTimeout(checkBackendHealth, 4000);
    }
  }
  checkBackendHealth();

  // --- Stage Tab Navigation ---
  const stageTabs = document.querySelectorAll('.stage-tab');
  stageTabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const targetTabId = tab.getAttribute('data-tab');
      switchTab(targetTabId);
    });
  });

  function switchTab(tabId) {
    stageTabs.forEach(t => {
      t.classList.toggle('active', t.getAttribute('data-tab') === tabId);
    });
    document.querySelectorAll('.tab-content').forEach(content => {
      content.classList.toggle('active', content.id === tabId);
    });
  }

  // --- Preset Query Chips ---
  document.querySelectorAll('.chip').forEach(chip => {
    chip.addEventListener('click', () => {
      const query = chip.getAttribute('data-query');
      if (query) {
        queryInput.value = query;
        queryInput.focus();
        submitQuery(query);
      }
    });
  });

  // --- Form Submission ---
  queryForm.addEventListener('submit', (e) => {
    e.preventDefault();
    const query = queryInput.value.trim();
    if (!query) return;
    submitQuery(query);
  });

  async function submitQuery(query) {
    const enableCrag = toggleCrag.checked;
    const topK = parseInt(selectTopK.value, 10) || 5;

    // Loading State
    btnSubmit.disabled = true;
    btnSubmit.innerHTML = `<span class="btn-text">Analyzing & Routing...</span> <span class="btn-icon">⏳</span>`;

    try {
      const res = await fetch('/api/v1/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: query,
          enable_crag: enableCrag,
          top_k: topK
        })
      });

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server responded with status ${res.status}`);
      }

      const data = await res.json();
      renderPipelineResponse(data);

    } catch (err) {
      alert(`Query Pipeline Error: ${err.message}`);
      console.error(err);
    } finally {
      btnSubmit.disabled = false;
      btnSubmit.innerHTML = `<span class="btn-text">Execute CRAG</span> <span class="btn-icon">⚡</span>`;
    }
  }

  // --- Render Complete Pipeline Response ---
  function renderPipelineResponse(data) {
    if (!data) return;

    emptyStateEl.classList.add('hidden');
    answerCardEl.classList.remove('hidden');

    // 1. Action Badge (from data.crag_action)
    const action = data.crag_action || 'CORRECT';
    cragActionBadge.textContent = action;
    cragActionBadge.className = 'action-badge ' + (
      action === 'CORRECT' ? 'badge-correct' :
      action === 'AMBIGUOUS' ? 'badge-ambiguous' : 'badge-incorrect'
    );

    // 2. Synthesized Answer with Markdown & Citation Pills
    let rawAnswer = data.answer || 'No answer generated.';
    
    // Replace [DOC:chunk_xxx] with interactive pill
    rawAnswer = rawAnswer.replace(/\[DOC:([^\]]+)\]/g, (match, chunkId) => {
      return `<span class="cite-badge cite-doc" data-chunk="${escapeHtml(chunkId)}">📄 ${escapeHtml(chunkId)}</span>`;
    });

    // Replace [WEB:url] with interactive web pill
    rawAnswer = rawAnswer.replace(/\[WEB:([^\]]+)\]/g, (match, url) => {
      let displayUrl = url;
      try {
        const parsed = new URL(url);
        displayUrl = parsed.hostname;
      } catch (e) {}
      return `<a href="${encodeURI(url)}" target="_blank" rel="noopener" class="cite-badge cite-web" title="${escapeHtml(url)}">🌐 ${escapeHtml(displayUrl)}</a>`;
    });

    if (window.marked) {
      answerBodyEl.innerHTML = marked.parse(rawAnswer);
    } else {
      answerBodyEl.innerHTML = rawAnswer;
    }

    // Attach click listeners to citation badges to highlight in inspector
    attachCitationListeners();

    // 3. Citations Panel (from data.stage_6_attribution_audit)
    citationChipsEl.innerHTML = '';
    const stg6 = data.stage_6_attribution_audit || {};
    const validCites = stg6.valid_citations || [];

    // Parse distinct docs and web citations
    const docCites = validCites.filter(c => c.startsWith('[DOC:')).map(c => c.slice(5, -1));
    const webCites = validCites.filter(c => c.startsWith('[WEB:')).map(c => c.slice(5, -1));

    if (validCites.length === 0) {
      citationChipsEl.innerHTML = `<span class="empty-text">No citations flagged in final response.</span>`;
    } else {
      docCites.forEach(c => {
        const chip = document.createElement('span');
        chip.className = 'citation-item-chip cite-doc';
        chip.innerHTML = `📄 ${escapeHtml(c)}`;
        chip.addEventListener('click', () => {
          switchTab('tab-fusion');
          highlightChunk(c);
        });
        citationChipsEl.appendChild(chip);
      });

      webCites.forEach(w => {
        const chip = document.createElement('a');
        chip.className = 'citation-item-chip cite-web';
        chip.href = encodeURI(w);
        chip.target = '_blank';
        chip.rel = 'noopener';
        let domain = w;
        try { domain = new URL(w).hostname; } catch(e){}
        chip.innerHTML = `🌐 ${escapeHtml(domain)}`;
        citationChipsEl.appendChild(chip);
      });
    }

    // 4. Answer Metrics Footer (from data.latencies_ms & data.stage_5_generation)
    const latencies = data.latencies_ms || {};
    const stg5 = data.stage_5_generation || {};
    
    metricLatency.textContent = `${formatNum(latencies.total, 0)} ms`;
    metricAttribution.textContent = formatPct(data.attribution_accuracy ?? stg6.attribution_accuracy ?? 1.0, 1);
    metricTokens.textContent = stg5.prompt_tokens != null ? stg5.prompt_tokens : '--';
    metricCost.textContent = formatCurrency(stg5.estimated_cost_usd, 5);

    // -------------------------------------------------------------
    // STAGE 1: Query Processing
    // -------------------------------------------------------------
    const stg1 = data.stage_1_query_processing || {};
    stg1QueryType.textContent = (stg1.detected_type || 'FACTUAL').toUpperCase();
    stg1QueryLen.textContent = `${data.query ? data.query.length : 0} chars`;

    const searchQueries = stg1.search_queries || [];
    if (searchQueries.length > 0) {
      stg1Subqueries.innerHTML = `<ol style="margin-left: 20px;">${searchQueries.map(sq => `<li style="margin-bottom: 6px; font-size: 0.85rem; color: var(--text-main);">${escapeHtml(sq)}</li>`).join('')}</ol>`;
    } else {
      stg1Subqueries.innerHTML = `<span class="empty-text">Single atomic query (no decomposition required).</span>`;
    }

    stg1Expanded.innerHTML = `<span class="empty-text">Latency: ${formatNum(stg1.latency_ms, 2)} ms</span>`;

    // -------------------------------------------------------------
    // STAGE 2: Hybrid Retrieval (Dense vs Sparse)
    // -------------------------------------------------------------
    const stg2 = data.stage_2_hybrid_retrieval || {};
    const denseItems = stg2.dense_qdrant_candidates || [];
    const sparseItems = stg2.sparse_bm25_candidates || [];

    stg2DenseCount.textContent = `${denseItems.length} chunks`;
    stg2SparseCount.textContent = `${sparseItems.length} chunks`;

    if (denseItems.length > 0) {
      stg2DenseList.innerHTML = denseItems.map(d => `
        <div class="candidate-item" id="dense-${escapeHtml(d.chunk_id)}">
          <div class="candidate-header">
            <span class="candidate-id">⚡ ${escapeHtml(d.chunk_id)}</span>
            <span class="candidate-score">Score: ${formatNum(d.score, 4)}</span>
          </div>
          <div class="candidate-snippet">${escapeHtml(d.content_snippet)}</div>
        </div>
      `).join('');
    } else {
      stg2DenseList.innerHTML = `<span class="empty-text">No dense candidates retrieved.</span>`;
    }

    if (sparseItems.length > 0) {
      stg2SparseList.innerHTML = sparseItems.map(s => `
        <div class="candidate-item" id="sparse-${escapeHtml(s.chunk_id)}">
          <div class="candidate-header">
            <span class="candidate-id">📖 ${escapeHtml(s.chunk_id)}</span>
            <span class="candidate-score">BM25: ${formatNum(s.score, 2)}</span>
          </div>
          <div class="candidate-snippet">${escapeHtml(s.content_snippet)}</div>
        </div>
      `).join('');
    } else {
      stg2SparseList.innerHTML = `<span class="empty-text">No sparse candidates retrieved.</span>`;
    }

    // -------------------------------------------------------------
    // STAGE 3: RRF Fusion & Cross-Encoder
    // -------------------------------------------------------------
    const stg3 = data.stage_3_fusion_and_rerank || {};
    const rerankedList = stg3.reranked_top5_candidates || [];
    const rrfPool = stg3.rrf_fused_chunk_ids || [];

    if (rerankedList.length > 0) {
      stg3RerankedList.innerHTML = rerankedList.map((item, idx) => {
        const normScore = Math.max(0, Math.min(1, ((item.score ?? 0) + 10) / 20));
        return `
          <div class="rerank-item ${idx === 0 ? 'top-1' : ''}" id="rerank-${escapeHtml(item.chunk_id)}">
            <div class="rerank-item-header">
              <div class="rerank-rank-badge">
                <span class="rank-num">#${idx + 1}</span>
                <span class="candidate-id">${escapeHtml(item.chunk_id)}</span>
              </div>
              <div class="rerank-score-bar-wrap">
                <span class="candidate-score font-mono">${formatNum(item.score, 3)}</span>
                <div class="score-bar-bg"><div class="score-bar-fill" style="width: ${normScore * 100}%"></div></div>
              </div>
            </div>
            <div class="candidate-snippet" style="-webkit-line-clamp: 4;">${escapeHtml(item.content_snippet)}</div>
          </div>
        `;
      }).join('');
    } else {
      stg3RerankedList.innerHTML = `<span class="empty-text">No reranked items.</span>`;
    }

    if (rrfPool.length > 0) {
      stg3RrfPool.innerHTML = rrfPool.map(cId => `<span class="rrf-tag">${escapeHtml(cId)}</span>`).join('');
    } else {
      stg3RrfPool.innerHTML = `<span class="empty-text">RRF pool empty.</span>`;
    }

    // -------------------------------------------------------------
    // STAGE 4: CRAG State Machine
    // -------------------------------------------------------------
    const stg4 = data.stage_4_crag_engine || {};
    const cragAction = stg4.action || action;
    stg4ActionText.textContent = cragAction;

    stg4DecisionBanner.className = 'decision-banner ' + (
      cragAction === 'CORRECT' ? 'banner-correct' :
      cragAction === 'AMBIGUOUS' ? 'banner-ambiguous' : 'banner-incorrect'
    );
    const circleEl = stg4DecisionBanner.querySelector('.decision-circle');
    if (circleEl) {
      circleEl.textContent =
        cragAction === 'CORRECT' ? '✓' :
        cragAction === 'AMBIGUOUS' ? '?' : '✗';
    }

    // Compute average confidence score from evaluator_scores
    const evalScores = stg4.evaluator_scores || [];
    let avgConf = 0.0;
    if (evalScores.length > 0) {
      const sum = evalScores.reduce((acc, curr) => acc + (Array.isArray(curr) ? curr[1] : 0), 0);
      avgConf = sum / evalScores.length;
    }
    stg4ConfidenceVal.textContent = evalScores.length > 0 ? formatNum(avgConf, 3) : '--';

    // Pointwise Evaluator Scores
    if (evalScores.length > 0) {
      stg4EvalScores.innerHTML = evalScores.map(tuple => {
        const [cId, sc] = Array.isArray(tuple) ? tuple : ['chunk', 0.0];
        const scColor = sc >= 0.7 ? 'var(--status-correct)' : (sc <= 0.3 ? 'var(--status-incorrect)' : 'var(--status-ambiguous)');
        return `
          <div class="eval-score-card">
            <div class="eval-chunk-name">${escapeHtml(cId)}</div>
            <div class="eval-chunk-score" style="color: ${scColor};">${formatNum(sc, 2)}</div>
          </div>
        `;
      }).join('');
    } else {
      stg4EvalScores.innerHTML = `<span class="empty-text">No pointwise evaluations conducted.</span>`;
    }

    // Dynamic Section: Strip Refinement OR Web Search Fallback
    if (stg4.web_searched) {
      stg4DynamicHeader.textContent = `Live Web Search Fallback (Tavily Search API)`;
      const webDocs = stg4.web_results || [];
      stg4DynamicBody.innerHTML = `
        <div style="margin-bottom: 10px;">
          <span style="font-size: 0.72rem; font-family: var(--font-mono); color: var(--amber-light); font-weight: 700;">REWRITTEN WEB QUERY:</span>
          <div style="font-size: 0.85rem; color: var(--text-main); margin-top: 4px; padding: 6px 10px; background: rgba(0,0,0,0.3); border-radius: 6px;">
            "${escapeHtml(stg4.rewritten_web_query || data.query)}"
          </div>
        </div>
        <div style="display: flex; flex-direction: column; gap: 8px;">
          ${webDocs.map(wd => `
            <div class="candidate-item" style="border-left: 3px solid var(--status-web);">
              <div class="candidate-header">
                <a href="${encodeURI(wd.url)}" target="_blank" rel="noopener" style="color: var(--status-web); text-decoration: none; font-size: 0.78rem; font-weight: 600;">
                  🌐 ${escapeHtml(wd.title || wd.url)}
                </a>
              </div>
              <div class="candidate-snippet">${escapeHtml(wd.content_snippet)}</div>
            </div>
          `).join('')}
        </div>
      `;
    } else {
      stg4DynamicHeader.textContent = `Knowledge Refinement & Internal Filtering`;
      const accepted = stg4.accepted_chunk_ids || [];
      const discarded = stg4.discarded_chunk_ids || [];
      stg4DynamicBody.innerHTML = `
        <div style="font-size: 0.78rem; color: var(--text-secondary); margin-bottom: 8px;">
          Internal decomposition & threshold filtering applied:
        </div>
        <div style="display: flex; gap: 14px; flex-wrap: wrap;">
          <div>
            <span style="font-size: 0.72rem; font-weight: 700; color: var(--status-correct); font-family: var(--font-mono);">ACCEPTED CHUNKS (${accepted.length}):</span>
            <div style="display: flex; flex-wrap: wrap; gap: 6px; margin-top: 4px;">
              ${accepted.map(cId => `<span class="rrf-tag" style="border-color: var(--status-correct); color: var(--status-correct);">${escapeHtml(cId)}</span>`).join('') || '<span class="empty-text">None</span>'}
            </div>
          </div>
          <div>
            <span style="font-size: 0.72rem; font-weight: 700; color: var(--status-incorrect); font-family: var(--font-mono);">DISCARDED CHUNKS (${discarded.length}):</span>
            <div style="display: flex; flex-wrap: wrap; gap: 6px; margin-top: 4px;">
              ${discarded.map(cId => `<span class="rrf-tag" style="border-color: var(--status-incorrect); color: var(--status-incorrect);">${escapeHtml(cId)}</span>`).join('') || '<span class="empty-text">None</span>'}
            </div>
          </div>
        </div>
      `;
    }

    // -------------------------------------------------------------
    // STAGE 5: Attribution Audit & Latency Waterfall
    // -------------------------------------------------------------
    stg5VerifiedCites.textContent = (stg6.valid_citations || []).length;
    stg5HallucinatedCites.textContent = (stg6.hallucinated_citations || []).length;
    stg5AttributionScore.textContent = formatPct(stg6.attribution_accuracy ?? 1.0, 1);

    // Latency Waterfall
    const stagesLatency = [
      { name: 'Query Processing', ms: latencies.query_processing || 0 },
      { name: 'Hybrid Retrieval', ms: latencies.retrieval || 0 },
      { name: 'RRF & Fusion', ms: latencies.fusion || 0 },
      { name: 'Cross-Encoder Rerank', ms: latencies.reranking_1 || 0 },
      { name: 'CRAG State Machine', ms: latencies.crag_midsection || 0 },
      { name: 'Gemini Generation', ms: latencies.generation || 0 },
      { name: 'Attribution Audit', ms: latencies.verification || 0 }
    ];
    const totalMs = latencies.total || stagesLatency.reduce((a, b) => a + b.ms, 0);

    stg5LatencyWaterfall.innerHTML = stagesLatency.map(st => {
      const pct = totalMs > 0 ? Math.min(100, (st.ms / totalMs) * 100) : 0;
      return `
        <div class="waterfall-row">
          <span class="waterfall-label">${st.name}</span>
          <div class="waterfall-bar-track">
            <div class="waterfall-bar-fill" style="width: ${pct.toFixed(1)}%;"></div>
          </div>
          <span class="waterfall-val">${formatNum(st.ms, 1)} ms</span>
        </div>
      `;
    }).join('');

    // Final Assembled Context (from stage_5_generation.formatted_context_preview)
    stg5FinalContext.textContent = stg5.formatted_context_preview || 'No context returned.';
  }

  // --- Citation Interactivity ---
  function attachCitationListeners() {
    document.querySelectorAll('.cite-doc').forEach(el => {
      el.addEventListener('click', () => {
        const chunkId = el.getAttribute('data-chunk');
        if (chunkId) {
          switchTab('tab-fusion');
          highlightChunk(chunkId);
        }
      });
    });
  }

  function highlightChunk(chunkId) {
    const targetEl = document.getElementById(`rerank-${chunkId}`) || document.getElementById(`dense-${chunkId}`) || document.getElementById(`sparse-${chunkId}`);
    if (targetEl) {
      targetEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
      targetEl.style.boxShadow = '0 0 24px var(--amber-glow)';
      targetEl.style.borderColor = 'var(--amber-light)';
      setTimeout(() => {
        targetEl.style.boxShadow = '';
        targetEl.style.borderColor = '';
      }, 2500);
    }
  }

  // --- Copy Context Button ---
  btnCopyContext.addEventListener('click', () => {
    if (stg5FinalContext.textContent) {
      navigator.clipboard.writeText(stg5FinalContext.textContent);
      btnCopyContext.textContent = 'Copied!';
      setTimeout(() => { btnCopyContext.textContent = 'Copy Context'; }, 2000);
    }
  });

  // --- Utility ---
  function escapeHtml(text) {
    if (!text) return '';
    return String(text)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }
});
