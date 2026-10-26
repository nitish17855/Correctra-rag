import os
import sys
import logging
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict
from langchain_core.documents import Document

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from CRAG.Retrieval_Evaulator import RetrievalEvaluator
from CRAG.actionRouter import ActionRouter, ActionType
from CRAG.knowledge_refiner import KnowledgeRefiner
from CRAG.query_rewriter import QueryRewriter
from CRAG.source_attributor import SourceAttributor, AttributionMetrics
from reranker.cross_encoder_reranker import CrossEncoderReranker

load_dotenv()
logger = logging.getLogger(__name__)


class CRAGResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    action: str                                # "CORRECT", "INCORRECT", "AMBIGUOUS"
    final_documents: list[Document]            # Ready for Context Assembly / Generation
    formatted_context: str                     # Disambiguated [DOC:...] and [WEB:...] XML context
    evaluator_scores: list[tuple[str, float]]
    accepted_chunk_ids: list[str]
    discarded_chunk_ids: list[str]
    web_searched: bool
    rewritten_web_query: Optional[str] = None


class CRAGOrchestrator:
    """
    End-to-End Corrective RAG (CRAG) Orchestrator with LCEL chains,
    intelligent query rewriting, and dual-source [DOC:...]/[WEB:...] attribution.
    """

    def __init__(
        self,
        evaluator: Optional[RetrievalEvaluator] = None,
        router: Optional[ActionRouter] = None,
        refiner: Optional[KnowledgeRefiner] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        query_rewriter: Optional[QueryRewriter] = None,
        source_attributor: Optional[SourceAttributor] = None,
        tavily_api_key: Optional[str] = None,
        web_search_max_results: int = 3,
    ):
        self.reranker = reranker or CrossEncoderReranker()
        self.evaluator = evaluator or RetrievalEvaluator()
        self.router = router or ActionRouter()
        self.refiner = refiner or KnowledgeRefiner(reranker=self.reranker)
        self.query_rewriter = query_rewriter or QueryRewriter()
        self.source_attributor = source_attributor or SourceAttributor()

        self.tavily_api_key = tavily_api_key or os.getenv("TAVILY_API_KEY")
        self.web_search_max_results = web_search_max_results

    def _search_web(self, query: str) -> tuple[list[Document], Optional[str]]:
        """
        Rewrites the query using QueryRewriter, then searches Tavily.
        Returns (list[Document], rewritten_query).
        """
        if not self.tavily_api_key:
            logger.warning("TAVILY_API_KEY is not set in environment. Skipping web search.")
            return [], None

        # 1. Rewrite user query into search engine keywords
        search_query = self.query_rewriter.rewrite(query)

        try:
            from tavily import TavilyClient

            client = TavilyClient(api_key=self.tavily_api_key)
            # Retrieve a candidate pool of web results so the Cross-Encoder can filter and rank the best snippets
            fetch_count = max(self.web_search_max_results * 2, 6)
            response = client.search(query=search_query, max_results=fetch_count)

            web_docs: list[Document] = []
            for item in response.get("results", []):
                content = item.get("content", "").strip()
                if content:
                    web_docs.append(
                        Document(
                            page_content=content,
                            metadata={
                                "source": item.get("url", "web"),
                                "title": item.get("title", ""),
                                "is_web": True,
                            },
                        )
                    )
            return web_docs, search_query
        except Exception as e:
            logger.warning(f"Tavily web search failed with error: {e}. Returning empty web results.")
            return [], search_query

    def orchestrate(
        self, query: str, documents: list[Document]
    ) -> CRAGResult:
        """
        Executes the full CRAG pipeline:
        1. Pointwise evaluate chunks via RetrievalEvaluator (Gemma).
        2. ActionRouter decides: CORRECT, INCORRECT, or AMBIGUOUS.
        3. Branching:
           - CORRECT:   Refine internal accepted documents (decompose -> recompose).
           - INCORRECT: Discard internal docs, rewrite query, search web -> Cross-Encoder rerank web snippets.
           - AMBIGUOUS: Refine internal docs + web search -> Cross-Encoder rerank joint pool.
        4. Formats context with [DOC:...] and [WEB:...] markers for source attribution.
        """
        if not documents:
            raw_web_docs, rewritten_q = self._search_web(query)
            if raw_web_docs:
                final_docs = self.reranker.rerank(
                    query=query, documents=raw_web_docs, top_k=min(self.web_search_max_results, len(raw_web_docs))
                )
            else:
                final_docs = []
            context_str = self.source_attributor.format_context(final_docs)
            return CRAGResult(
                action=ActionType.INCORRECT.value,
                final_documents=final_docs,
                formatted_context=context_str,
                evaluator_scores=[],
                accepted_chunk_ids=[],
                discarded_chunk_ids=[],
                web_searched=bool(final_docs),
                rewritten_web_query=rewritten_q,
            )

        # 1. Pointwise Evaluation: score chunks with Gemma
        eval_dict = self.evaluator.evaluate(query=query, documents=documents)
        eval_scores_list = list(eval_dict.items())

        # 2. Action Routing: dual-threshold decision
        decision = self.router.route(documents=documents, scores=eval_dict)
        accepted_ids = [d.metadata.get("chunk_id", "") for d in decision.accepted_documents]
        discarded_ids = [d.metadata.get("chunk_id", "") for d in decision.discarded_documents]

        # 3. Branching execution
        final_docs: list[Document] = []
        web_searched = False
        rewritten_q: Optional[str] = None

        if decision.action == ActionType.CORRECT:
            # Internal evidence is solid: refine accepted chunks
            final_docs = self.refiner.filter_and_recompose(
                query=query, documents=decision.accepted_documents
            )

        elif decision.action == ActionType.INCORRECT:
            # Internal evidence failed: discard and search the web with rewritten query
            raw_web_docs, rewritten_q = self._search_web(query)
            web_searched = True
            if raw_web_docs:
                # Rerank web chunks using Cross-Encoder to select the highest quality evidence
                final_docs = self.reranker.rerank(
                    query=query,
                    documents=raw_web_docs,
                    top_k=min(self.web_search_max_results, len(raw_web_docs)),
                )
            else:
                final_docs = []

        elif decision.action == ActionType.AMBIGUOUS:
            # Ambiguous: refine internal + web search, then re-rank merged pool
            refined_internal = self.refiner.filter_and_recompose(
                query=query, documents=decision.accepted_documents
            )
            raw_web_docs, rewritten_q = self._search_web(query)
            web_searched = True

            if raw_web_docs:
                # Pre-filter web docs using Cross-Encoder
                top_web = self.reranker.rerank(
                    query=query,
                    documents=raw_web_docs,
                    top_k=min(self.web_search_max_results, len(raw_web_docs)),
                )
            else:
                top_web = []

            merged_pool = refined_internal + top_web
            if merged_pool:
                final_docs = self.reranker.rerank(
                    query=query,
                    documents=merged_pool,
                    top_k=min(self.web_search_max_results, len(merged_pool)),
                )
            else:
                final_docs = []

        # 4. Format context with source attribution markers
        formatted_context = self.source_attributor.format_context(final_docs)

        return CRAGResult(
            action=decision.action.value,
            final_documents=final_docs,
            formatted_context=formatted_context,
            evaluator_scores=eval_scores_list,
            accepted_chunk_ids=accepted_ids,
            discarded_chunk_ids=discarded_ids,
            web_searched=web_searched,
            rewritten_web_query=rewritten_q,
        )

    def verify_attribution(self, answer: str, result: CRAGResult) -> AttributionMetrics:
        """
        Validates citations in the generated answer against the retrieved documents
        and computes the attribution_accuracy metric.
        """
        return self.source_attributor.attribution_accuracy(
            answer=answer, retrieved_docs=result.final_documents
        )


if __name__ == "__main__":
    print("=" * 70)
    print("TESTING FULL CRAG ORCHESTRATOR WITH QUERY REWRITING & DUAL ATTRIBUTION")
    print("=" * 70)

    orchestrator = CRAGOrchestrator()

    # Test Case 1: CORRECT Action Branch
    print("\n[TEST 1] Testing CORRECT Action Branch...")
    query_correct = "What was Apple's total net sales in fiscal year 2025?"
    candidates_correct = [
        Document(
            page_content="Total net sales were $416,161 million in 2025 compared to $391,035 million in 2024. Growth was driven by Services and iPhone.",
            metadata={"chunk_id": "chunk_sales_01", "source": "10k.html"},
        ),
        Document(
            page_content="Risk factors include macroeconomic volatility, inflation, and currency fluctuations in Asian manufacturing hubs.",
            metadata={"chunk_id": "chunk_risk_02", "source": "10k.html"},
        ),
    ]

    result_correct = orchestrator.orchestrate(query=query_correct, documents=candidates_correct)
    print(f"Action Decision: {result_correct.action}")
    print(f"Accepted Chunks: {result_correct.accepted_chunk_ids}")
    print(f"Discarded Chunks: {result_correct.discarded_chunk_ids}")
    print(f"Web Searched: {result_correct.web_searched}")
    assert result_correct.action == "CORRECT"
    assert "DOC:chunk_sales_01" in result_correct.formatted_context
    print("[TEST 1 PASSED]")

    # Test Case 2: INCORRECT Action Branch (with Query Rewriting & Web Search)
    print("\n[TEST 2] Testing INCORRECT Action Branch with Query Rewriting...")
    query_incorrect = "According to table 3 on page 14, who won the ICC Cricket World Cup in 2023?"
    candidates_incorrect = [
        Document(
            page_content="Commercial risks include inflation and currency fluctuations in Asian hubs.",
            metadata={"chunk_id": "chunk_risk_02", "source": "10k.html"},
        ),
    ]
    result_incorrect = orchestrator.orchestrate(query=query_incorrect, documents=candidates_incorrect)
    print(f"Action Decision: {result_incorrect.action}")
    print(f"Rewritten Query for Web: '{result_incorrect.rewritten_web_query}'")
    print(f"Web Searched: {result_incorrect.web_searched}")
    print(f"Web Documents Retrieved: {len(result_incorrect.final_documents)}")
    assert result_incorrect.action == "INCORRECT"
    assert result_incorrect.web_searched is True
    assert "table 3" not in result_incorrect.rewritten_web_query.lower()
    print("[TEST 2 PASSED]")

    # Test Case 3: Attribution Accuracy Verification
    print("\n[TEST 3] Testing Attribution Accuracy Metric...")
    sample_answer = (
        "In 2025, Apple recorded $416,161 million in sales [DOC:chunk_sales_01]. "
        "Australia won the World Cup [WEB:https://en.wikipedia.org/wiki/2023_Cricket_World_Cup]."
    )
    # Combine docs from Test 1 and Test 2 to verify dual attribution
    combined_docs = result_correct.final_documents + result_incorrect.final_documents
    metric = orchestrator.source_attributor.attribution_accuracy(sample_answer, combined_docs)
    print(f"Attribution Accuracy: {metric.attribution_accuracy * 100:.1f}%")
    print(f"Valid Citations: {metric.valid_citations}/{metric.total_citations}")
    print(f"Internal Citations: {metric.internal_citation_count}, Web Citations: {metric.web_citation_count}")
    assert metric.attribution_accuracy == 1.0
    print("[TEST 3 PASSED]")

    print("\n======================================================================")
    print("ALL TESTS PASSED: CRAG ORCHESTRATOR FULLY VERIFIED!")
    print("======================================================================")
