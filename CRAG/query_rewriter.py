import os
import sys
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# Ensure workspace root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

QUERY_REWRITER_TEMPLATE = """You are an expert financial and corporate intelligence search query optimizer.
Your task is to transform a user's conversational question into a high-precision, authoritative web search query optimized for search engines (Tavily, Google, SEC EDGAR).

GOAL:
Produce a targeted search query that retrieves authoritative corporate disclosures (SEC 10-K filings, annual reports, investor relations, official leadership announcements) matching the exact entity, topic, and fiscal year.

TRANSFORMATION RULES:
1. Strip conversational filler and document-internal phrasing:
   - Remove phrases like: "who is the", "tell me about", "what are their", "according to chunk 001", "in table 4", "in this filing", "mentioned on page 12".
2. Identify and specify the exact Corporate Entity:
   - Expand colloquial company names to their official corporate form (e.g., "Nvidia" -> "NVIDIA Corporation", "Apple" -> "Apple Inc.", "Microsoft" -> "Microsoft Corporation").
3. Retain and strictly anchor Temporal / Fiscal Periods:
   - If the query mentions a fiscal year or calendar year (e.g. "2025", "FY2025", "fiscal 2025"), preserve it explicitly (e.g., "FY2025" or "fiscal year 2025"). Never search for other years.
4. Target Authoritative Sources:
   - For revenue, earnings, sales, balance sheet, or financial figures: Append "Form 10-K annual report OR investor relations".
   - For leadership, CEO, executive officers, or governance: Append "CEO executive leadership official corporate governance".
   - For products / segments / technologies: Include exact product line keywords (e.g., "flagship AI GPU chips").
5. Output Format:
   - Output ONLY the clean, keyword-dense search query (no quotation marks, no preamble, no explanations).

Examples:
- User Query: "Who is the CEO of Nvidia in 2025 and what are their flagship AI products?"
  Rewritten: "NVIDIA Corporation CEO executive leadership 2025 flagship AI GPU products"
- User Query: "In table 4 on page 32, what was Apple's total revenue in fiscal year 2025?"
  Rewritten: "Apple Inc. total net sales revenue FY2025 Form 10-K annual report investor relations"
- User Query: "What was Nvidia's total revenue in fiscal 2025?"
  Rewritten: "NVIDIA Corporation total revenue FY2025 Form 10-K annual report investor relations"
- User Query: "Compare iPhone revenue between fiscal 2024 and 2025."
  Rewritten: "Apple Inc. iPhone net sales revenue FY2024 FY2025 Form 10-K annual report"

User Query: {query}

Rewritten Web Search Query:"""


class QueryRewriter:
    """
    Transforms user queries into authoritative, entity-anchored,
    and temporally-constrained web search queries using a LangChain LCEL chain.
    """

    def __init__(self, model_name: str = "google/gemma-4-31b-it"):
        self.api_key = os.getenv("REQUESTY_API_KEY")
        self.base_url = os.getenv("REQUESTY_BASE_URL", "https://router.requesty.ai/v1")

        self.llm = ChatOpenAI(
            model=model_name,
            openai_api_key=self.api_key,
            openai_api_base=self.base_url,
            temperature=0.0,
        )
        self.prompt = PromptTemplate(
            template=QUERY_REWRITER_TEMPLATE,
            input_variables=["query"],
        )

        # Declarative LangChain LCEL Chain
        self.chain = self.prompt | self.llm | StrOutputParser()

    def rewrite(self, query: str) -> str:
        """
        Rewrites the query using the LangChain chain.
        Falls back to the original query if rewriting fails.
        """
        if not query or not query.strip():
            return ""
        try:
            result = self.chain.invoke({"query": query.strip()})
            cleaned = result.strip().strip('"').strip("'")
            return cleaned if cleaned else query
        except Exception:
            return query


if __name__ == "__main__":
    print("=" * 70)
    print("TESTING ENHANCED CORPORATE QUERY REWRITER WITH LANGCHAIN LCEL")
    print("=" * 70)

    rewriter = QueryRewriter()

    test_queries = [
        "Who is the CEO of Nvidia in 2025 and what are their flagship AI products?",
        "What was Nvidia's total revenue in fiscal 2025?",
        "In table 4 on page 32, what was Apple's total revenue in fiscal year 2025?",
        "According to chunk 012, who is the CEO of Microsoft?",
    ]

    for q in test_queries:
        rewritten = rewriter.rewrite(q)
        print(f"\nOriginal : '{q}'")
        print(f"Rewritten: '{rewritten}'")
        assert len(rewritten) > 0, "Rewritten query should not be empty"

    print("\n[SUCCESS] Enhanced QueryRewriter tests completed!")
