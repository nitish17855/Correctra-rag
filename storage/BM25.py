import re
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document


class BM25Store:
    """
    BM25 Sparse Lexical Store:
    Uses Okapi BM25 for statistical term frequency / inverse document frequency (TF-IDF) matching.
    Excels at exact keyword matches, ticker symbols, financial terms, and model numbers.
    """

    def __init__(self):
        self.documents: list[Document] = []
        self.bm25: BM25Okapi | None = None

    def tokenize(self, text: str) -> list[str]:
        """
        Normalizes text by lowercasing and extracting alphanumeric word tokens,
        ignoring punctuation marks and stray symbols.
        """
        return re.findall(r"\w+", text.lower())

    def index_documents(self, documents: list[Document]) -> None:
        """
        Builds the BM25 statistical corpus from a list of Document objects.
        """
        if not documents:
            self.documents = []
            self.bm25 = None
            return

        self.documents = list(documents)
        tokenized_corpus = [self.tokenize(doc.page_content) for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, top_k: int = 5) -> list[Document]:
        """
        Scores all indexed documents against the query using BM25 and returns
        the top_k documents with their BM25 score attached in metadata['score'].
        """
        if self.bm25 is None or not self.documents:
            return []

        tokenized_query = self.tokenize(query)
        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)

        # Sort document indices by descending BM25 score
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        # Reconstruct Document objects with similarity score in metadata
        results = []
        for i in ranked_indices:
            doc = self.documents[i]
            metadata = dict(doc.metadata)
            metadata["score"] = float(scores[i])
            results.append(Document(page_content=doc.page_content, metadata=metadata))

        return results


if __name__ == "__main__":
    print("Testing BM25Store lexical search...")
    store = BM25Store()

    # 1. Sample corpus
    sample_docs = [
        Document(
            page_content="Apple Inc. reported record iPhone revenue of 96.77 billion dollars in fiscal 2025.",
            metadata={"chunk_id": "chunk_001", "source": "apple_10k.htm"},
        ),
        Document(
            page_content="Tesla announced rapid acceleration of autonomous Cybercab vehicle deliveries.",
            metadata={"chunk_id": "chunk_002", "source": "tesla_10k.htm"},
        ),
        Document(
            page_content="Microsoft Cloud and Azure infrastructure growth exceeded enterprise market expectations.",
            metadata={"chunk_id": "chunk_003", "source": "msft_10k.htm"},
        ),
    ]

    # 2. Index documents
    store.index_documents(sample_docs)
    print(f"Indexed {len(sample_docs)} documents into BM25Store.")

    # 3. Exact keyword search test
    query = "iPhone revenue 96.77 billion"
    print(f"\nSearching for exact keywords: '{query}'")
    results = store.search(query=query, top_k=2)

    print(f"Returned {len(results)} matching documents:")
    for idx, res in enumerate(results):
        print(f" Rank {idx + 1}: [{res.metadata['chunk_id']}] Score: {res.metadata['score']:.4f}")
        print(f" Content: {res.page_content}")

    # 4. Assertions
    assert len(results) == 2, "Expected 2 top results"
    assert results[0].metadata["chunk_id"] == "chunk_001", "chunk_001 should rank #1 for iPhone revenue"
    assert results[0].metadata["score"] > results[1].metadata["score"], "Top result must have strictly higher score"
    assert results[0].metadata["score"] > 0.0, "Matching document must have positive BM25 score"

    print("\nAll BM25Store tests and assertions passed successfully!")
