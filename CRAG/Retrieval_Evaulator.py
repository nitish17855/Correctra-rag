import os
import json
import re
from typing import Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

load_dotenv()


class RetrievalEvaluator:
    """
    Retrieval Evaluator for Corrective RAG (CRAG).
    
    Evaluates the absolute relevance of retrieved/reranked document chunks
    with respect to a given query using a LangChain chain and an LLM hosted on Requesty.
    
    Status: Active CRAG Retrieval Evaluator (LangChain LCEL Pipeline).
    """

    def __init__(
        self,
        model_name: str = "google/gemma-4-31b-it",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
    ):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("REQUESTY_API_KEY")
        self.base_url = base_url or os.environ.get("REQUESTY_BASE_URL", "https://router.requesty.ai/v1")

        if not self.api_key:
            raise ValueError(
                "REQUESTY_API_KEY is not set. Please add REQUESTY_API_KEY to your .env file or pass it to __init__."
            )

        # Initialize LangChain ChatOpenAI configured for Requesty AI router
        self.llm = ChatOpenAI(
            model=self.model_name,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=temperature,
        )

        # Define the Batched Evaluation Prompt Template with triple-quotes
        self.prompt_template = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are an expert Information Retrieval Evaluator in a Corrective RAG (CRAG) system.
Your job is to assess the ABSOLUTE relevance of candidate document chunks with respect to a user query.

CRITICAL SCORING RULES:
1. Absolute Relevance Only: Score each chunk strictly on its own merits from 0.0 to 1.0.
   - 1.0: The chunk directly and factually contains the information needed to answer the query.
   - 0.5: The chunk contains partial information, related context, or background, but cannot fully answer the query.
   - 0.0: The chunk is completely irrelevant, off-topic, or unhelpful.
2. NO Relative Ranking: Do NOT compare chunks against each other. If all chunks are irrelevant, give all 0.0. If all are perfect, give all 1.0.
3. Strict JSON Format: You must respond ONLY with a JSON object mapping each chunk_id to its numerical score.
Do NOT include any commentary, Markdown fences, or preamble outside the JSON."""
            ),
            (
                "user",
                """User Query:
"{query}"

Candidate Chunks to Evaluate:
{formatted_chunks}

Output your evaluation as valid JSON in this exact structure:
{{
  "evaluations": {{
    "chunk_id_1": 0.95,
    "chunk_id_2": 0.10
  }}
}}"""
            )
        ])

        # Explicit LangChain LCEL Chain (Prompt -> LLM -> String Output Parser)
        self.chain = self.prompt_template | self.llm | StrOutputParser()

    def _format_chunks(self, documents: list[Document]) -> str:
        """Formats documents into a structured string for batch evaluation."""
        formatted_blocks = []
        for idx, doc in enumerate(documents):
            chunk_id = doc.metadata.get("chunk_id", f"chunk_{idx:03d}")
            text_snippet = doc.page_content.strip()
            formatted_blocks.append(f"[Chunk ID: {chunk_id}]\nContent: {text_snippet}")
        return "\n\n".join(formatted_blocks)

    def _clean_and_parse_json(self, response_text: str) -> dict[str, float]:
        """Extracts and parses JSON object from model output with regex fallback."""
        cleaned = response_text.strip()
        # Remove markdown code fences if present
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
            cleaned = cleaned.strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            # Fallback: find first JSON block via regex
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
            else:
                raise ValueError(f"Could not parse valid JSON from model response: {response_text}")

        # Extract dictionary of scores
        if "evaluations" in data and isinstance(data["evaluations"], dict):
            raw_scores = data["evaluations"]
        elif isinstance(data, dict):
            raw_scores = data
        else:
            raise ValueError(f"Unexpected JSON structure from model: {data}")

        # Ensure scores are normalized floats between 0.0 and 1.0
        scores: dict[str, float] = {}
        for cid, val in raw_scores.items():
            try:
                score_float = float(val)
                scores[str(cid)] = max(0.0, min(1.0, round(score_float, 4)))
            except (ValueError, TypeError):
                scores[str(cid)] = 0.0

        return scores

    def evaluate(self, query: str, documents: list[Document]) -> dict[str, float]:
        """
        Evaluates the relevance of each document chunk against the query using the LangChain chain.
        
        Args:
            query: The user's query string.
            documents: List of reranked candidate Document chunks.
            
        Returns:
            dict[str, float]: Mapping from chunk_id to its absolute relevance score [0.0, 1.0].
        """
        if not documents:
            return {}

        formatted_chunks = self._format_chunks(documents)

        try:
            # Invoke the LangChain LCEL chain
            raw_output = self.chain.invoke({
                "query": query,
                "formatted_chunks": formatted_chunks
            })
            scores = self._clean_and_parse_json(raw_output)

            # Ensure all input documents are accounted for in the output dict
            final_scores: dict[str, float] = {}
            for idx, doc in enumerate(documents):
                cid = doc.metadata.get("chunk_id", f"chunk_{idx:03d}")
                final_scores[cid] = scores.get(cid, 0.0)

            return final_scores

        except Exception as e:
            # Defensive fallback: if API fails, return neutral 0.0 for all chunks
            print(f"[!] RetrievalEvaluator error: {e}")
            fallback_scores = {}
            for idx, doc in enumerate(documents):
                cid = doc.metadata.get("chunk_id", f"chunk_{idx:03d}")
                fallback_scores[cid] = 0.0
            return fallback_scores


if __name__ == "__main__":
    print("=" * 70)
    print("TESTING CRAG RETRIEVAL EVALUATOR WITH LANGCHAIN CHAIN (GEMMA-4-31B-IT)")
    print("=" * 70)

    sample_query = "What was Apple's total net sales in fiscal year 2025?"
    sample_docs = [
        Document(
            page_content="Total net sales for fiscal year 2025 were $416,161 million, compared to $391,035 million in 2024.",
            metadata={"chunk_id": "chunk_152"}
        ),
        Document(
            page_content="The Company sells accessories including Apple Watch bands, iPhone cases, and wireless headphones.",
            metadata={"chunk_id": "chunk_042"}
        ),
        Document(
            page_content="iPhone remains Apple's primary revenue driver, contributing $209,562 million to 2025 net sales.",
            metadata={"chunk_id": "chunk_153"}
        )
    ]

    try:
        evaluator = RetrievalEvaluator()
        print(f"\nQuery: '{sample_query}'")
        print(f"Evaluating {len(sample_docs)} candidate chunks via LangChain chain...")
        results = evaluator.evaluate(sample_query, sample_docs)
        print("\nRelevance Scores:")
        for chunk_id, score in results.items():
            print(f" -> {chunk_id}: {score}")
    except Exception as err:
        print(f"[!] Error: {err}")
