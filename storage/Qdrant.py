from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_core.documents import Document


class QdrantStore:
    """
    Qdrant Dense Vector Store:
    Supports in-memory (':memory:'), local file persistence, or remote Docker instances.
    Stores dense vectors alongside structured document payload metadata.
    """

    def __init__(
        self,
        collection_name: str = "rag_chunks",
        vector_size: int = 768,
        location: str = ":memory:",
    ):
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.client = QdrantClient(location=location)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Creates the collection if it does not already exist."""
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                ),
            )

    def upsert_documents(
        self,
        documents: list[Document],
        vectors: list[list[float]],
    ) -> None:
        """
        Uploads documents paired with their dense embedding vectors into Qdrant.
        """
        if len(documents) != len(vectors):
            raise ValueError("Number of documents must match the number of vectors")

        points = []
        for idx, (doc, vector) in enumerate(zip(documents, vectors)):
            payload = {
                "page_content": doc.page_content,
                **doc.metadata,
            }
            points.append(
                PointStruct(
                    id=idx,
                    vector=vector,
                    payload=payload,
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
    ) -> list[Document]:
        """
        Queries Qdrant for nearest neighbors and reconstructs Document objects with scores.
        """
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
        )

        documents = []
        for point in results.points:
            payload = point.payload or {}
            content = payload.get("page_content", "")
            metadata = {k: v for k, v in payload.items() if k != "page_content"}
            metadata["score"] = point.score

            documents.append(Document(page_content=content, metadata=metadata))

        return documents


if __name__ == "__main__":
    print("Testing QdrantStore in-memory...")
    store = QdrantStore(collection_name="test_collection", vector_size=4, location=":memory:")

    # 1. Prepare sample documents and normalized vectors
    docs = [
        Document(
            page_content="Apple reports strong iPhone sales in fiscal year 2025.",
            metadata={"chunk_id": "chunk_001", "source": "apple_10k.htm"},
        ),
        Document(
            page_content="Tesla announces new autonomous vehicle production timeline.",
            metadata={"chunk_id": "chunk_002", "source": "tesla_10k.htm"},
        ),
    ]
    vectors = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ]

    # 2. Upsert
    store.upsert_documents(docs, vectors)
    print("Upserted 2 test documents successfully.")

    # 3. Query (search vector strongly aligned with Document 0)
    query_vector = [0.99, 0.01, 0.0, 0.0]
    search_results = store.search(query_vector=query_vector, top_k=2)

    print(f"\nSearch returned {len(search_results)} documents:")
    print(f"Top result: {search_results[0].page_content}")
    print(f"Top result metadata: {search_results[0].metadata}")
    print(f"Similarity Score: {search_results[0].metadata['score']:.4f}")

    # 4. Assertions
    assert len(search_results) == 2, "Expected 2 search results"
    assert search_results[0].metadata["chunk_id"] == "chunk_001", "Top match should be chunk_001"
    assert search_results[0].metadata["score"] > 0.90, "Expected high cosine similarity score"

    print("\nAll QdrantStore assertions passed successfully!")
