from enum import Enum
from typing import NamedTuple
from pydantic import BaseModel, Field, ConfigDict
from langchain_core.documents import Document


class ActionType(str, Enum):
    """
    Core CRAG Actions as defined in Yan et al. (2024):
    - CORRECT: Internal retrieval is confident; proceed to Knowledge Refinement.
    - INCORRECT: All retrieved documents are irrelevant; trigger Web Search.
    - AMBIGUOUS: Uncertain relevance; combine internal refinement with Web Search.
    """
    CORRECT = "CORRECT"
    AMBIGUOUS = "AMBIGUOUS"
    INCORRECT = "INCORRECT"


class RouterDecision(BaseModel):
    """
    Structured outcome of the Action Router.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    action: ActionType
    max_score: float = Field(..., description="Maximum score among all candidate chunks: max_i(s_i)")
    upper_threshold: float = Field(..., description="High confidence threshold T_H")
    lower_threshold: float = Field(..., description="Low confidence threshold T_L")
    accepted_documents: list[Document] = Field(default_factory=list, description="Documents meeting the threshold for refinement")
    discarded_documents: list[Document] = Field(default_factory=list, description="Noise documents stripped out of context")
    chunk_scores: dict[str, float] = Field(default_factory=dict, description="Relevance score per chunk_id")


class ActionRouter:
    """
    Action Router for Corrective RAG (CRAG).

    Applies dual-threshold logic on point-wise relevance scores (s_1, ..., s_k):
      - CORRECT:   max_i(s_i) >= T_H
      - INCORRECT: max_i(s_i) <  T_L
      - AMBIGUOUS: otherwise (T_L <= max_i(s_i) < T_H)
    """

    def __init__(self, upper_threshold: float = 0.7, lower_threshold: float = 0.3):
        """
        Args:
            upper_threshold: High threshold T_H (default 0.7).
            lower_threshold: Low threshold T_L (default 0.3).
        """
        if not (0.0 <= lower_threshold < upper_threshold <= 1.0):
            raise ValueError(
                f"Invalid thresholds: Must satisfy 0.0 <= T_L < T_H <= 1.0. "
                f"Got lower_threshold={lower_threshold}, upper_threshold={upper_threshold}"
            )
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold

    def route(self, documents: list[Document], scores: dict[str, float]) -> RouterDecision:
        """
        Determines the CRAG Action and filters candidate documents.

        Args:
            documents: List of candidate Document objects from reranker.
            scores: Dictionary mapping chunk_id to its absolute relevance score [0.0, 1.0].

        Returns:
            RouterDecision containing the action, max_score, and accepted/discarded docs.
        """
        if not documents:
            return RouterDecision(
                action=ActionType.INCORRECT,
                max_score=0.0,
                upper_threshold=self.upper_threshold,
                lower_threshold=self.lower_threshold,
                accepted_documents=[],
                discarded_documents=[],
                chunk_scores=scores
            )

        # 1. Calculate max score across all candidate chunks: max_i (s_i)
        score_values = []
        for idx, doc in enumerate(documents):
            cid = doc.metadata.get("chunk_id", f"chunk_{idx:03d}")
            score_values.append(scores.get(cid, 0.0))

        max_score = max(score_values) if score_values else 0.0

        # 2. Apply Dual-Threshold Logic from CRAG research paper
        if max_score >= self.upper_threshold:
            action = ActionType.CORRECT
        elif max_score < self.lower_threshold:
            action = ActionType.INCORRECT
        else:
            action = ActionType.AMBIGUOUS

        # 3. Filter chunks: In CORRECT or AMBIGUOUS state, retain chunks that meet T_L
        # Strip out noisy, completely irrelevant chunks (score < T_L)
        accepted = []
        discarded = []

        for idx, doc in enumerate(documents):
            cid = doc.metadata.get("chunk_id", f"chunk_{idx:03d}")
            score = scores.get(cid, 0.0)

            if action == ActionType.INCORRECT:
                # All documents judged unhelpful
                discarded.append(doc)
            else:
                # Retain chunks with positive signal (>= T_L)
                if score >= self.lower_threshold:
                    accepted.append(doc)
                else:
                    discarded.append(doc)

        return RouterDecision(
            action=action,
            max_score=round(max_score, 4),
            upper_threshold=self.upper_threshold,
            lower_threshold=self.lower_threshold,
            accepted_documents=accepted,
            discarded_documents=discarded,
            chunk_scores=scores
        )


if __name__ == "__main__":
    print("=" * 70)
    print("TESTING CRAG ACTION ROUTER WITH DUAL-THRESHOLD LOGIC")
    print("=" * 70)

    router = ActionRouter(upper_threshold=0.7, lower_threshold=0.3)

    docs = [
        Document(page_content="Net sales were $416B in 2025.", metadata={"chunk_id": "chunk_152"}),
        Document(page_content="Apple Watch bands in autumn colors.", metadata={"chunk_id": "chunk_042"}),
        Document(page_content="iPhone contributed $209B to revenue.", metadata={"chunk_id": "chunk_153"}),
    ]

    # Test Case 1: High confidence match -> CORRECT
    print("\n--- Test Case 1: Max Score >= 0.7 ---")
    scores_correct = {"chunk_152": 1.0, "chunk_042": 0.0, "chunk_153": 0.5}
    decision1 = router.route(docs, scores_correct)
    print(f"Action:       {decision1.action.value}")
    print(f"Max Score:    {decision1.max_score} (T_H={decision1.upper_threshold})")
    print(f"Accepted:     {[d.metadata['chunk_id'] for d in decision1.accepted_documents]}")
    print(f"Discarded:    {[d.metadata['chunk_id'] for d in decision1.discarded_documents]}")

    # Test Case 2: Only partial matches (e.g., 0.5) -> AMBIGUOUS
    print("\n--- Test Case 2: 0.3 <= Max Score < 0.7 ---")
    scores_ambiguous = {"chunk_152": 0.5, "chunk_042": 0.1, "chunk_153": 0.4}
    decision2 = router.route(docs, scores_ambiguous)
    print(f"Action:       {decision2.action.value}")
    print(f"Max Score:    {decision2.max_score}")
    print(f"Accepted:     {[d.metadata['chunk_id'] for d in decision2.accepted_documents]}")
    print(f"Discarded:    {[d.metadata['chunk_id'] for d in decision2.discarded_documents]}")

    # Test Case 3: Completely irrelevant (all < 0.3) -> INCORRECT
    print("\n--- Test Case 3: Max Score < 0.3 ---")
    scores_incorrect = {"chunk_152": 0.1, "chunk_042": 0.0, "chunk_153": 0.2}
    decision3 = router.route(docs, scores_incorrect)
    print(f"Action:       {decision3.action.value}")
    print(f"Max Score:    {decision3.max_score} (T_L={decision3.lower_threshold})")
    print(f"Accepted:     {[d.metadata['chunk_id'] for d in decision3.accepted_documents]}")
    print(f"Discarded:    {[d.metadata['chunk_id'] for d in decision3.discarded_documents]}")
