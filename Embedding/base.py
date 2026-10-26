from abc import ABC, abstractmethod


class BaseEmbedding(ABC):
    """
    Abstract Base Class for all Embedding Providers in the RAG pipeline.
    Enforces a consistent contract across local models (BGE) and remote APIs (Gemini/OpenAI).
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Returns the embedding vector size (e.g. 768 for BGE-base)."""
        pass

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """Converts a single user query into an embedding vector."""
        pass

    @abstractmethod
    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """Converts a list of document chunks into embedding vectors."""
        pass