import sys
from pathlib import Path

# Ensure workspace root is in sys.path when running script directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from reranker.base import RerankerProvider


class CrossEncoderReranker(RerankerProvider):
    """
    Local Cross-Encoder Reranker using cross-encoder/ms-marco-MiniLM-L6-v2.
    Applies joint cross-attention over [query, document] token pairs to compute
    fine-grained relevance scores that capture negation, syntax, and nuanced context.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int = 5,
    ) -> list[Document]:
        """
        Reranks a list of candidate documents against the query.
        """
        if not documents:
            return []

        # Form [query, passage] pairs for joint cross-attention
        pairs = [[query, doc.page_content] for doc in documents]
        raw_scores = self.model.predict(pairs)

        # Pair each document with its cross-encoder score and sort descending
        scored_docs = sorted(
            zip(documents, raw_scores),
            key=lambda item: item[1],
            reverse=True,
        )[:top_k]

        reranked_results = []
        for doc, score in scored_docs:
            metadata = dict(doc.metadata)
            metadata["rerank_score"] = round(float(score), 4)
            metadata["reranker"] = self.model_name
            reranked_results.append(
                Document(page_content=doc.page_content, metadata=metadata)
            )

        return reranked_results


if __name__ == "__main__":
    print("=" * 70)
    print("TESTING CROSS-ENCODER RERANKER (ms-marco-MiniLM-L6-v2)")
    print("=" * 70)

    reranker = CrossEncoderReranker()

    # Query asking about revenue GROWTH
    query = "Why did iPhone net sales increase in fiscal year 2025?"

    # Candidates: Candidate 2 has superficial keywords (iPhone, sales, 2025)
    # but Candidate 1 genuinely explains the CAUSE ("due to higher net sales of Pro models").
    candidates = [
        Document(
            page_content="Mac net sales increased during 2025 compared to 2024 primarily due to higher net sales of laptops.",
            metadata={"chunk_id": "chunk_01", "stage": "initial"},
        ),
        Document(
            page_content="iPhone net sales increased during 2025 compared to 2024 primarily due to higher net sales of iPhone Pro models.",
            metadata={"chunk_id": "chunk_02", "stage": "initial"},
        ),
        Document(
            page_content="iPhone total net sales were reported under commercial risk factors in China without specific product commentary.",
            metadata={"chunk_id": "chunk_03", "stage": "initial"},
        ),
    ]

    print(f"\nQuery: '{query}'")
    print(f"Reranking {len(candidates)} candidates with joint cross-attention...")

    results = reranker.rerank(query=query, documents=candidates, top_k=3)

    print("\nReranked Results:")
    for rank, doc in enumerate(results, 1):
        print(f" Rank {rank}: [{doc.metadata['chunk_id']}] Score: {doc.metadata['rerank_score']:+.4f}")
        print(f" Content: {doc.page_content}")

    # Assertions
    assert len(results) == 3, "Expected 3 reranked documents"
    assert results[0].metadata["chunk_id"] == "chunk_02", "chunk_02 directly answers the 'Why' question and must be Rank 1"
    assert results[0].metadata["rerank_score"] > results[1].metadata["rerank_score"], "Top result must have strictly highest score"

    print("\nAll CrossEncoderReranker tests and assertions passed successfully!")
