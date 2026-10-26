import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataclasses import dataclass
import numpy as np
from scipy.special import softmax
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from generation.citation_Extractor import CitationExtractor, CitationCoverageResult


@dataclass
class ClaimVerification:
    statement: str
    cited_chunk_ids: list[str]
    entailment_score: float
    contradiction_score: float
    neutral_score: float
    predicted_label: str  # "ENTAILMENT", "CONTRADICTION", "NEUTRAL"
    is_grounded: bool


@dataclass
class GroundingResult:
    status: str  # "GROUNDED", "PARTIAL", "HALLUCINATED"
    overall_grounding_score: float  # Mean entailment across claims
    total_claims: int
    grounded_claims_count: int
    hallucinated_claims_count: int
    claim_verifications: list[ClaimVerification]
    citation_coverage: CitationCoverageResult
    explanation: str


class GroundingVerifier:
    """
    Two-Tier NLI Grounding Verification Engine:
    Tier 1: Syntactic citation extraction and coverage check.
    Tier 2: Semantic Natural Language Inference (NLI) using cross-encoder/nli-deberta-v3-small.
            Evaluates (Premise=Chunk, Hypothesis=Claim) to detect semantic hallucinations.
    """

    # Label index mapping from nli-deberta-v3-small
    LABEL_CONTRADICTION = 0
    LABEL_ENTAILMENT = 1
    LABEL_NEUTRAL = 2

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-small",
        entailment_threshold: float = 0.70,
        contradiction_threshold: float = 0.50,
    ):
        self.model_name = model_name
        self.entailment_threshold = entailment_threshold
        self.contradiction_threshold = contradiction_threshold
        self.model = CrossEncoder(model_name)
        self.citation_extractor = CitationExtractor()

    def verify(
        self,
        answer: str,
        retrieved_documents: list[Document] | dict[str, str],
    ) -> GroundingResult:
        """
        Validates an answer against the retrieved documents using NLI entailment.
        """
        # Convert documents to a chunk_id -> text dictionary
        if isinstance(retrieved_documents, list):
            doc_map = {
                d.metadata.get("chunk_id", f"doc_{i}"): d.page_content
                for i, d in enumerate(retrieved_documents)
            }
        else:
            doc_map = dict(retrieved_documents)

        retrieved_ids = list(doc_map.keys())

        # 1. Tier 1: Citation Coverage Check
        coverage = self.citation_extractor.verify_coverage(answer, retrieved_ids=retrieved_ids)

        # If answer is a clean, intentional abstention
        if coverage.is_abstention and len(coverage.hallucinated_citations) == 0:
            return GroundingResult(
                status="GROUNDED",
                overall_grounding_score=1.0,
                total_claims=0,
                grounded_claims_count=0,
                hallucinated_claims_count=0,
                claim_verifications=[],
                citation_coverage=coverage,
                explanation="Model correctly abstained from answering due to lack of evidence.",
            )

        # 2. Tier 2: Sentence-Level NLI Verification
        sentences = self.citation_extractor.parse_sentences(answer)
        claim_verifications: list[ClaimVerification] = []

        for item in sentences:
            # Skip empty or pure conversational filler
            if len(item.sentence.strip()) < 10:
                continue

            # If the statement cites no source, it is ungrounded by definition
            if not item.cited_ids:
                claim_verifications.append(
                    ClaimVerification(
                        statement=item.sentence,
                        cited_chunk_ids=[],
                        entailment_score=0.0,
                        contradiction_score=0.0,
                        neutral_score=1.0,
                        predicted_label="UNCITED_CLAIM",
                        is_grounded=False,
                    )
                )
                continue

            # For each citation cited by this statement, test NLI entailment
            # We take the maximum entailment across supporting chunks cited
            best_entailment = 0.0
            best_contradiction = 0.0
            best_neutral = 1.0
            best_label = "NEUTRAL"

            for cid in item.cited_ids:
                if cid not in doc_map:
                    # Cited an ID that does not exist in retrieved set
                    best_label = "HALLUCINATED_ID"
                    best_neutral = 1.0
                    continue

                premise = doc_map[cid].strip()
                # Strip bracketed citation tags so the NLI model evaluates clean natural language semantics
                clean_hypothesis = self.citation_extractor.pattern.sub("", item.sentence).strip()

                # Predict NLI logits
                raw_logits = self.model.predict([(premise, clean_hypothesis)])
                probs = softmax(raw_logits, axis=-1)[0]

                p_contra = float(probs[self.LABEL_CONTRADICTION])
                p_entail = float(probs[self.LABEL_ENTAILMENT])
                p_neutral = float(probs[self.LABEL_NEUTRAL])

                if p_entail > best_entailment:
                    best_entailment = p_entail
                    best_contradiction = p_contra
                    best_neutral = p_neutral
                    best_label = "ENTAILMENT" if p_entail >= self.entailment_threshold else ("CONTRADICTION" if p_contra >= self.contradiction_threshold else "NEUTRAL")

            is_grounded = best_entailment >= self.entailment_threshold

            claim_verifications.append(
                ClaimVerification(
                    statement=item.sentence,
                    cited_chunk_ids=item.cited_ids,
                    entailment_score=round(best_entailment, 4),
                    contradiction_score=round(best_contradiction, 4),
                    neutral_score=round(best_neutral, 4),
                    predicted_label=best_label,
                    is_grounded=is_grounded,
                )
            )

        # 3. Aggregate Status Classification
        total_claims = len(claim_verifications)
        grounded_claims = sum(1 for c in claim_verifications if c.is_grounded)
        hallucinated_claims = total_claims - grounded_claims
        has_contradiction = any(c.predicted_label == "CONTRADICTION" for c in claim_verifications)
        has_hallucinated_ids = len(coverage.hallucinated_citations) > 0

        mean_entailment = (
            sum(c.entailment_score for c in claim_verifications) / total_claims
            if total_claims > 0
            else 0.0
        )

        # Classification rules:
        # GROUNDED: All claims entailed (>= 0.70) AND zero hallucinated IDs AND zero uncited statements
        # HALLUCINATED: Contradiction present OR zero grounded claims OR coverage == 0
        # PARTIAL: Some claims entailed, but some missing/unsupported
        if total_claims == 0:
            status = "GROUNDED" if coverage.is_abstention else "HALLUCINATED"
            explanation = "No claims extracted."
        elif has_contradiction or (grounded_claims == 0 and not coverage.is_abstention):
            status = "HALLUCINATED"
            explanation = f"Contains contradictions or 0/{total_claims} supported claims."
        elif grounded_claims == total_claims and not has_hallucinated_ids:
            status = "GROUNDED"
            explanation = f"All {total_claims} claims mathematically entailed by cited chunks (Mean Entailment: {mean_entailment:.2f})."
        elif grounded_claims > 0:
            status = "PARTIAL"
            explanation = f"{grounded_claims}/{total_claims} claims supported. {hallucinated_claims} claims require review."
        else:
            status = "HALLUCINATED"
            explanation = "Answer contains unsupported claims or invalid citations."

        return GroundingResult(
            status=status,
            overall_grounding_score=round(mean_entailment, 4),
            total_claims=total_claims,
            grounded_claims_count=grounded_claims,
            hallucinated_claims_count=hallucinated_claims,
            claim_verifications=claim_verifications,
            citation_coverage=coverage,
            explanation=explanation,
        )


if __name__ == "__main__":
    print("=" * 75)
    print("TESTING NLI-BASED GROUNDING VERIFIER (nli-deberta-v3-small)")
    print("=" * 75)

    verifier = GroundingVerifier()

    test_docs = [
        Document(
            page_content="Apple Inc. total net sales in fiscal 2025 reached $391,035 million compared to $383,285 million in 2024.",
            metadata={"chunk_id": "chunk_001"},
        ),
        Document(
            page_content="iPhone net sales were $201,183 million in fiscal year 2025 primarily driven by Pro models.",
            metadata={"chunk_id": "chunk_002"},
        ),
    ]

    # Test 1: Genuine Grounded Answer
    answer_grounded = (
        "Total net sales in 2025 were $391,035 million [chunk_001]. "
        "iPhone revenue reached $201,183 million driven by Pro models [chunk_002]."
    )
    res1 = verifier.verify(answer_grounded, test_docs)
    print("\n[Test 1: True Grounded Answer]")
    print(f"Status: {res1.status} | Score: {res1.overall_grounding_score}")
    print(f"Explanation: {res1.explanation}")
    for c in res1.claim_verifications:
        print(f" - Claim: '{c.statement[:55]}...' -> Entailment: {c.entailment_score} ({c.predicted_label})")
    assert res1.status == "GROUNDED"

    # Test 2: Semantic Hallucination (False number citing genuine chunk_001)
    answer_hallucinated = (
        "Total net sales in 2025 reached $650,000 million [chunk_001]."  # False number!
    )
    res2 = verifier.verify(answer_hallucinated, test_docs)
    print("\n[Test 2: Semantic Hallucination (Fabricated Number)]")
    print(f"Status: {res2.status} | Score: {res2.overall_grounding_score}")
    print(f"Explanation: {res2.explanation}")
    for c in res2.claim_verifications:
        print(f" - Claim: '{c.statement}' -> Entailment: {c.entailment_score} ({c.predicted_label})")
    assert res2.status == "HALLUCINATED"

    # Test 3: Partial Grounding (1 true claim, 1 uncited/unsupported claim)
    answer_partial = (
        "Total net sales were $391,035 million [chunk_001]. "
        "Apple will launch an electric car in late 2026."  # Uncited outside knowledge!
    )
    res3 = verifier.verify(answer_partial, test_docs)
    print("\n[Test 3: Partial Grounding]")
    print(f"Status: {res3.status} | Score: {res3.overall_grounding_score}")
    print(f"Explanation: {res3.explanation}")
    assert res3.status == "PARTIAL"

    print("\nAll GroundingVerifier NLI tests and assertions passed successfully!")
