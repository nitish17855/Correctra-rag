import os
from typing import Literal
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()


class QueryClassification(BaseModel):
    """
    Data contract for query classification results.
    """
    query_type: Literal["factual", "multi_hop", "summarization"]
    explanation: str


class QueryClassifier:
    """
    Query Classifier for SEC 10-K RAG:
    Uses Gemini with Structured Outputs to categorize incoming user queries
    into:
      - 'factual': Single target fact, metric, or entity -> Direct retrieval
      - 'multi_hop': Comparison, cause-and-effect, or multi-part question -> Query Decomposer
      - 'summarization': Broad thematic overview -> Expanded retrieval window
    """

    def __init__(self, model_name: str = "gemini-3.5-flash-lite", api_key: str | None = None):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if self.api_key:
            self.client = genai.Client(api_key=self.api_key, http_options=types.HttpOptions(timeout=30000))
        else:
            self.client = None

    def classify(self, query: str) -> QueryClassification:
        """
        Classifies a user query into factual, multi_hop, or summarization
        with reasoning using Gemini structured output.
        """
        if self.client is None:
            raise ValueError(
                "GEMINI_API_KEY not found. Please add GEMINI_API_KEY=your_key to a .env file or set it in your environment."
            )

        prompt = f"""You are a specialized query analyzer for a Financial RAG system assisting users with SEC Form 10-K filings.

Analyze the user's input query and classify it into exactly ONE of the following categories:

1. "factual":
   - The query seeks a specific, localized data point, financial number, date, or single isolated fact.
   - Examples: "What were Apple's net sales in 2025?", "Who is Apple's CFO?", "What is the cash balance?"

2. "multi_hop":
   - The query requires connecting two or more distinct facts, comparing time periods, or linking cause-and-effect across different sections.
   - Examples: "Compare iPhone revenue in 2024 vs 2025 and explain why gross margins changed.", "How did supply chain challenges in China impact Services gross margin?"

3. "summarization":
   - The query asks for an overarching, broad synthesis or high-level overview across wide passages or full sections.
   - Examples: "Summarize the primary risk factors in Item 1A.", "Give an overview of Apple's business segments and strategy."

User Query: "{query}"

Provide your classification and a concise 1-2 sentence explanation justifying why the query belongs in that category.
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
                        response_schema=QueryClassification,
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
        "What was Apple's total revenue in fiscal year 2025?",
        "Compare iPhone net sales between 2024 and 2025 and explain how the Services margin shifted.",
        "Provide a high-level summary of the major risk factors affecting international manufacturing.",
    ]

    print("=" * 70)
    print("TESTING QUERY CLASSIFIER WITH GEMINI")
    print("=" * 70)

    try:
        classifier = QueryClassifier()
        for q in test_queries:
            print(f"\nQuery: '{q}'")
            result = classifier.classify(q)
            print(f" -> Classification: {result.query_type.upper()}")
            print(f" -> Explanation:    {result.explanation}")
    except ValueError as e:
        print(f"\n[!] Configuration Notice: {e}")
        print("To run live classifications with Gemini, create a .env file with:")
        print("GEMINI_API_KEY=your_api_key_here\n")
