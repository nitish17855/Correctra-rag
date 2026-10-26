import os
from typing import Literal
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

class Query_decomposition(BaseModel): 
    sub_queries:list[str]
    explanation:str

class Query_decomposer():

    def __init__(self, model_name: str = "gemini-3.5-flash-lite", api_key: str | None = None):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if self.api_key:
            self.client = genai.Client(api_key=self.api_key, http_options=types.HttpOptions(timeout=30000))
        else:
            self.client = None

    def decompose(self, query: str) -> Query_decomposition:
        if not self.client:
            raise ValueError(
                "GEMINI_API_KEY not found. Please add GEMINI_API_KEY=your_key to a .env file or set it in your environment."
            )

        prompt = f"""You are an expert financial query decomposition specialist for an SEC Form 10-K RAG system.

Your job is to analyze complex, multi-hop, comparative, or multi-part financial questions and break them down into 2 or more atomic, independent sub-queries.

CRITICAL RULES:
1. Self-Contained (Anaphora Resolution):
   - Every sub-query will be sent to search engines independently.
   - NEVER use ambiguous pronouns like "it", "its", "they", "the company", or "that period".
   - Explicitly replace pronouns with the actual company name (e.g., "Apple"), fiscal year (e.g., "fiscal 2025"), or segment (e.g., "iPhone", "Services").

2. Single-Focus:
   - Each sub-query must target ONE specific concept, financial table, or section of the 10-K filing.

3. Completeness:
   - Answering all sub-queries must provide all the necessary information to answer the original user query completely.

4. Already Atomic:
   - If the input query is already a single atomic question that cannot be decomposed, return a list containing just the original query.

User Query: "{query}"

Provide your decomposed sub-queries and a brief 1-2 sentence explanation of your decomposition logic.
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
                        response_schema=Query_decomposition,
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
        "Compare Apple's 2024 and 2025 iPhone net sales and explain how gross margins shifted between Products and Services.",
        "What are Apple's outstanding antitrust legal proceedings and what are the main supply chain risks in China?",
        "What was Apple's total cash balance at the end of fiscal 2025?",
    ]

    print("=" * 70)
    print("TESTING QUERY DECOMPOSER WITH GEMINI")
    print("=" * 70)

    try:
        decomposer = Query_decomposer()
        for q in test_queries:
            print(f"\nOriginal Query: '{q}'")
            result = decomposer.decompose(q)
            print(f"Explanation: {result.explanation}")
            print("Sub-queries:")
            for i, sub in enumerate(result.sub_queries, 1):
                print(f"  [{i}] {sub}")
    except ValueError as e:
        print(f"\n[!] Configuration Notice: {e}")
        print("To run live decomposition with Gemini, create a .env file with:")
        print("GEMINI_API_KEY=your_api_key_here\n")

