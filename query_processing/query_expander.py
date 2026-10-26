import os
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()


class QueryExpansion(BaseModel):
    expanded_queries: list[str]   # Exactly 3 distinct variants
    hypothetical_passage: str     # Simulated 10-K excerpt (HyDE)
    reasoning: str


class QueryExpander:
    def __init__(self, model_name: str = "gemini-3.5-flash-lite", api_key: str | None = None):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.client = genai.Client(api_key=self.api_key, http_options=types.HttpOptions(timeout=30000)) if self.api_key else None

    def expand(self, query: str) -> QueryExpansion:
        if self.client is None:
            raise ValueError(
                "GEMINI_API_KEY not found. Please add GEMINI_API_KEY=your_key to a .env file or set it in your environment."
            )

        prompt = f"""You are an expert financial search engineer specializing in SEC Form 10-K filings.

Analyze the user's query and generate 3 targeted query variants and 1 HyDE passage:

1. Variant 1 (Keyword / BM25):
   - Dense with financial keywords, accounting terms, and ticker (e.g., AAPL).
2. Variant 2 (Formal SEC 10-K Terminology):
   - Rephrased into formal GAAP / SEC disclosure language.
3. Variant 3 (Semantic Paraphrase):
   - A clear alternative phrasing for dense vector similarity.
4. Hypothetical Passage (HyDE):
   - Write a 2-3 sentence simulated excerpt from an Apple 10-K report that hypothetically answers this query, matching the formal tone and structure of financial statements or Risk Factors.

User Query: "{query}"
"""

        import time
        max_retries = 4
        base_delay = 1.5
        last_exception = None

        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=QueryExpansion,
                        temperature=0.0,
                    ),
                )
                return response.parsed
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    sleep_time = base_delay * (2 ** attempt)
                    time.sleep(sleep_time)
                else:
                    raise last_exception


if __name__ == "__main__":
    test_queries = [
        "What were Apple's total revenue in fiscal year 2025?",
        "Compare iPhone net sales between 2024 and 2025 and explain how the Services margin shifted.",
        "Provide a high-level summary of the major risk factors affecting international manufacturing.",
    ]

    print("=" * 70)
    print("TESTING QUERY EXPANDER WITH GEMINI")
    print("=" * 70)

    try:
        expander = QueryExpander()
        for q in test_queries:
            print(f"\nQuery: '{q}'")
            result = expander.expand(q)
            print(f" -> Expanded Queries:")
            for i, variant in enumerate(result.expanded_queries):
                print(f"      {i + 1}. {variant}")
            print(f" -> Hypothetical Passage:\n{result.hypothetical_passage}\n")
    except ValueError as e:
        print(f"\n[!] Configuration Notice: {e}")
        print("To run live expansions with Gemini, create a .env file with:")
        print("GEMINI_API_KEY=your_api_key_here\n")
