from sentence_transformers import SentenceTransformer
from Embedding.base import BaseEmbedding


class BGEEmbedding(BaseEmbedding):
    """
    Local BGE Embedding Provider using BAAI/bge-base-en-v1.5.
    - Generates 768-dimensional dense vectors.
    - Applies L2 normalization for optimal cosine similarity (dot product).
    - Automatically prepends the required BGE query task instruction.
    """

    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5"):
        self.model_name = model_name
        self.query_instruction = "Represent this sentence for searching relevant passages: "
        self.model = SentenceTransformer(model_name)
        self._dimension = self.model.get_sentence_embedding_dimension()

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """
        Embeds a list of document passages in batches.
        No instruction prefix is required for documents.
        """
        if not documents:
            return []
        embeddings = self.model.encode(
            documents,
            normalize_embeddings=True,
            batch_size=32,
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """
        Embeds a single user query prepended with the BGE query instruction.
        """
        prompt = self.query_instruction + query
        embedding = self.model.encode(
            prompt,
            normalize_embeddings=True,
        )
        return embedding.tolist()


if __name__ == "__main__":
    print("Loading BGE-base-en-v1.5 model...")
    embedder = BGEEmbedding()

    # 1. Verify Dimension
    print(f"Model Dimension: {embedder.dimension}")
    assert embedder.dimension == 768, f"Expected 768, got {embedder.dimension}"

    # 2. Test embed_query
    test_query = "What was Apple's total revenue in fiscal year 2025?"
    query_vector = embedder.embed_query(test_query)
    print(f"Query Vector length: {len(query_vector)}")
    assert len(query_vector) == 768, "Query vector dimension mismatch"
    assert isinstance(query_vector[0], float), "Vector components must be floats"

    # 3. Test embed_documents
    test_docs = [
        "Apple Inc. designs, manufactures, and markets smartphones, personal computers, and tablets.",
        "Total net sales were $391,035 million in 2025, compared to $383,285 million in 2024.",
    ]
    doc_vectors = embedder.embed_documents(test_docs)
    print(f"Document Vectors count: {len(doc_vectors)}")
    assert len(doc_vectors) == 2, "Expected 2 document vectors"
    assert len(doc_vectors[0]) == 768, "Document 0 dimension mismatch"
    assert len(doc_vectors[1]) == 768, "Document 1 dimension mismatch"

    print("\nAll BGE Embedding assertions passed successfully!")
