from langchain_core.documents import Document


def normalize_scores(docs: list[Document]) -> dict[str, float]:
    """
    Min-Max normalizes raw retrieval scores to the range [0.0, 1.0].
    Returns a mapping of chunk_id -> normalized_score.
    """
    if not docs:
        return {}

    raw_scores = [doc.metadata.get("score", 0.0) for doc in docs]
    min_score = min(raw_scores)
    max_score = max(raw_scores)

    normalized = {}
    for doc in docs:
        doc_id = doc.metadata.get("chunk_id") or doc.page_content
        raw = doc.metadata.get("score", 0.0)

        if max_score == min_score:
            normalized[doc_id] = 1.0
        else:
            normalized[doc_id] = (raw - min_score) / (max_score - min_score)

    return normalized


def weighted_linear_fusion(
    dense_docs: list[Document],
    sparse_docs: list[Document],
    alpha: float = 0.6,
    top_k: int = 5,
) -> list[Document]:
    """
    Weighted Linear Fusion (WLF):
    Normalizes scores from both dense and sparse retrievers using Min-Max scaling,
    then combines them via a linear combination:
      final_score = alpha * dense_norm + (1 - alpha) * sparse_norm

    Parameters:
      dense_docs: Documents retrieved by dense semantic search (with score in metadata).
      sparse_docs: Documents retrieved by sparse keyword search (with score in metadata).
      alpha: Weight for dense retrieval (0.0 <= alpha <= 1.0). Default = 0.6.
      top_k: Number of fused documents to return.
    """
    dense_norm = normalize_scores(dense_docs)
    sparse_norm = normalize_scores(sparse_docs)

    doc_map = {}
    for doc in dense_docs + sparse_docs:
        doc_id = doc.metadata.get("chunk_id") or doc.page_content
        if doc_id not in doc_map:
            doc_map[doc_id] = doc

    wlf_scores = {}
    for doc_id in set(dense_norm.keys()) | set(sparse_norm.keys()):
        d_score = dense_norm.get(doc_id, 0.0)
        s_score = sparse_norm.get(doc_id, 0.0)
        wlf_scores[doc_id] = (alpha * d_score) + ((1.0 - alpha) * s_score)

    sorted_ids = sorted(wlf_scores.keys(), key=lambda i: wlf_scores[i], reverse=True)[:top_k]

    final_docs = []
    for doc_id in sorted_ids:
        original_doc = doc_map[doc_id]
        metadata = dict(original_doc.metadata)
        metadata["wlf_score"] = round(wlf_scores[doc_id], 4)
        metadata["fusion_method"] = "wlf"
        final_docs.append(Document(page_content=original_doc.page_content, metadata=metadata))

    return final_docs


# Alias for backward compatibility
wlf = weighted_linear_fusion


if __name__ == "__main__":
    print("=" * 70)
    print("TESTING WEIGHTED LINEAR FUSION (WLF)")
    print("=" * 70)

    # 1. Simulate Dense Retriever (e.g., Qdrant with Cosine similarity)
    dense_run = [
        Document(page_content="iPhone revenue grew in 2025.", metadata={"chunk_id": "Chunk_A", "score": 0.90}),
        Document(page_content="Services gross margin expanded.", metadata={"chunk_id": "Chunk_B", "score": 0.80}),
        Document(page_content="China supply chain risks.", metadata={"chunk_id": "Chunk_C", "score": 0.60}),
    ]

    # 2. Simulate Sparse Retriever (e.g., BM25 with unbounded TF-IDF scores)
    sparse_run = [
        Document(page_content="Services gross margin expanded.", metadata={"chunk_id": "Chunk_B", "score": 25.0}),
        Document(page_content="Antitrust legal proceedings in EU.", metadata={"chunk_id": "Chunk_D", "score": 18.0}),
        Document(page_content="iPhone revenue grew in 2025.", metadata={"chunk_id": "Chunk_A", "score": 10.0}),
    ]

    print("\nInput Raw Scores:")
    print(" Dense (Cosine): Chunk_A: 0.90, Chunk_B: 0.80, Chunk_C: 0.60")
    print(" Sparse (BM25):  Chunk_B: 25.0, Chunk_D: 18.0, Chunk_A: 10.0")

    # 3. Execute WLF with alpha = 0.6 (60% Dense, 40% Sparse)
    fused = weighted_linear_fusion(dense_docs=dense_run, sparse_docs=sparse_run, alpha=0.6, top_k=4)

    print("\nFused WLF Results (alpha=0.6):")
    for rank, doc in enumerate(fused, start=1):
        chunk_id = doc.metadata["chunk_id"]
        score = doc.metadata["wlf_score"]
        print(f" Rank {rank}: [{chunk_id}] WLF Score: {score:.4f} | Content: {doc.page_content}")

    # 4. Assertions
    # Dense norm: Chunk_A=1.0, Chunk_B=0.6667, Chunk_C=0.0
    # Sparse norm: Chunk_B=1.0, Chunk_D=0.5333, Chunk_A=0.0
    # Chunk_B final = 0.6*(0.6667) + 0.4*(1.0) = 0.4000 + 0.4000 = 0.8000
    # Chunk_A final = 0.6*(1.0) + 0.4*(0.0) = 0.6000
    # Chunk_D final = 0.6*(0.0) + 0.4*(0.5333) = 0.2133
    # Chunk_C final = 0.6*(0.0) + 0.4*(0.0) = 0.0000
    assert len(fused) == 4, "Expected 4 fused documents"
    assert fused[0].metadata["chunk_id"] == "Chunk_B", "Chunk_B must rank #1"
    assert fused[1].metadata["chunk_id"] == "Chunk_A", "Chunk_A must rank #2"
    assert fused[2].metadata["chunk_id"] == "Chunk_D", "Chunk_D must rank #3"
    assert fused[3].metadata["chunk_id"] == "Chunk_C", "Chunk_C must rank #4"

    print("\nAll WLF normalization and fusion assertions passed successfully!")