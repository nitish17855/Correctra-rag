from langchain_core.documents import Document


def reciprocal_rank_fusion(
    runs: list[list[Document]],
    k: int = 60,
    top_k: int = 5,
) -> list[Document]:
    """
    Reciprocal Rank Fusion (RRF):
    Merges and deduplicates multiple ranked document lists into a single ranked list.
    Formula: RRF_Score(d) = sum(1 / (k + rank_i(d))) for each run where d appears.

    Parameters:
      runs: A list of ranked document lists (e.g., [dense_results, sparse_results]).
      k: Smoothing parameter (default = 60).
      top_k: Number of fused results to return.
    """
    doc_map: dict[str, Document] = {}
    rrf_scores: dict[str, float] = {}

    for docs in runs:
        for rank, doc in enumerate(docs, start=1):
            doc_id = doc.metadata.get("chunk_id") or doc.page_content
            doc_map[doc_id] = doc
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank))

    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    final_docs = []
    for doc_id, score in sorted_docs[:top_k]:
        original_doc = doc_map[doc_id]
        metadata = dict(original_doc.metadata)
        metadata["rrf_score"] = round(score, 6)
        metadata["fusion_method"] = "rrf"
        final_docs.append(Document(page_content=original_doc.page_content, metadata=metadata))

    return final_docs


if __name__ == "__main__":
    print("=" * 70)
    print("TESTING RECIPROCAL RANK FUSION (RRF)")
    print("=" * 70)

    # 1. Simulate Dense Retriever (e.g., Qdrant) results
    dense_run = [
        Document(page_content="iPhone revenue grew in 2025.", metadata={"chunk_id": "Chunk_A"}),
        Document(page_content="Services gross margin expanded.", metadata={"chunk_id": "Chunk_B"}),
        Document(page_content="China supply chain risks.", metadata={"chunk_id": "Chunk_C"}),
    ]

    # 2. Simulate Sparse Retriever (e.g., BM25) results
    sparse_run = [
        Document(page_content="Services gross margin expanded.", metadata={"chunk_id": "Chunk_B"}),
        Document(page_content="Antitrust legal proceedings in EU.", metadata={"chunk_id": "Chunk_D"}),
        Document(page_content="iPhone revenue grew in 2025.", metadata={"chunk_id": "Chunk_A"}),
    ]

    print("\nInput Runs:")
    print(" Dense Run:  [Rank 1: Chunk_A, Rank 2: Chunk_B, Rank 3: Chunk_C]")
    print(" Sparse Run: [Rank 1: Chunk_B, Rank 2: Chunk_D, Rank 3: Chunk_A]")

    # 3. Execute RRF
    fused_results = reciprocal_rank_fusion(runs=[dense_run, sparse_run], k=60, top_k=4)

    print(f"\nFused RRF Results (k=60):")
    for rank, doc in enumerate(fused_results, start=1):
        chunk_id = doc.metadata["chunk_id"]
        score = doc.metadata["rrf_score"]
        print(f" Rank {rank}: [{chunk_id}] RRF Score: {score:.6f} | Content: {doc.page_content}")

    # 4. Assertions verifying the exact mathematical predictions
    assert len(fused_results) == 4, "Expected 4 deduplicated documents"
    assert fused_results[0].metadata["chunk_id"] == "Chunk_B", "Chunk_B must be #1 (appeared Rank 2 + Rank 1)"
    assert fused_results[1].metadata["chunk_id"] == "Chunk_A", "Chunk_A must be #2 (appeared Rank 1 + Rank 3)"
    assert fused_results[2].metadata["chunk_id"] == "Chunk_D", "Chunk_D must be #3 (single Rank 2)"
    assert fused_results[3].metadata["chunk_id"] == "Chunk_C", "Chunk_C must be #4 (single Rank 3)"

    print("\nAll RRF mathematical rankings and assertions passed successfully!")
