from abc import ABC, abstractmethod
from langchain_core.documents import Document


class RerankerProvider(ABC):
    """
    Abstract Base Class / Provider Interface for all Rerankers.
    Enforces a uniform interface so different reranker backends (e.g., local Cross-Encoder,
    Cohere API, BGE-Reranker, FlashRank) can be swapped seamlessly without altering the call site.
    """

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int = 5,
    ) -> list[Document]:
        """
        Scores candidate documents against the query using cross-attention,
        attaches the relevance score to doc.metadata['rerank_score'],
        and returns the top_k documents sorted in descending order.
        """
        pass


# Backwards compatibility alias
BaseReranker = RerankerProvider


def get_reranker(provider: str = "local", **kwargs) -> RerankerProvider:
    """
    Factory function to instantiate rerankers without coupling call sites
    to specific implementation classes.
    """
    provider_lower = provider.lower()
    if provider_lower in ("local", "cross_encoder", "minilm"):
        from reranker.cross_encoder_reranker import CrossEncoderReranker
        model_name = kwargs.get("model_name", "cross-encoder/ms-marco-MiniLM-L6-v2")
        return CrossEncoderReranker(model_name=model_name)
    else:
        raise ValueError(
            f"Unsupported reranker provider '{provider}'. Supported: 'local' (ms-marco-MiniLM-L6-v2)"
        )
