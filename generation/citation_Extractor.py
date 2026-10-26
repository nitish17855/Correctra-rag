import re
from dataclasses import dataclass, field


# Matches single or comma-separated chunk citations like [chunk_001], [chunk-001], or [chunk_001, chunk_002]
CITATION_REGEX = re.compile(
    r"\[(chunk[-_\w]+(?:\s*,\s*chunk[-_\w]+)*)\]",
    re.IGNORECASE,
)

# Standard abstention keywords
ABSTENTION_MARKERS = [
    "not contain sufficient information",
    "not disclosed in the provided",
    "information is not in provided data",
    "not available in the provided",
    "no information provided",
]


@dataclass
class SentenceCitation:
    sentence: str
    cited_ids: list[str]
    is_cited: bool


@dataclass
class CitationCoverageResult:
    total_citations_found: int
    unique_cited_ids: list[str]
    valid_citations: list[str]
    hallucinated_citations: list[str]
    uncited_statements: list[str]
    citation_coverage_score: float  # valid / total citations
    is_abstention: bool
    is_fully_grounded: bool


class CitationExtractor:
    """
    Extracts, parses, and validates [chunk-id] citations from LLM output.
    Computes citation coverage, identifies hallucinated IDs, and flags uncited statements.
    """

    def __init__(self, citation_pattern: re.Pattern = CITATION_REGEX):
        self.pattern = citation_pattern

    def extract_citations(self, text: str) -> list[str]:
        """
        Extracts all unique chunk IDs cited in the text in order of appearance.
        Handles both individual '[chunk_001]' and grouped '[chunk_001, chunk_002]' formats.
        """
        if not text:
            return []

        raw_matches = self.pattern.findall(text)
        cited_ids = []
        for match in raw_matches:
            # Split comma-separated IDs inside brackets
            parts = [p.strip() for p in match.split(",") if p.strip()]
            for part in parts:
                if part not in cited_ids:
                    cited_ids.append(part)
        return cited_ids

    def parse_sentences(self, text: str) -> list[SentenceCitation]:
        """
        Splits text into sentences and associates each sentence with its cited chunk IDs.
        Flags sentences that contain substantive claims without any citation.
        """
        if not text:
            return []

        # Split on sentence terminals (. ! ?) followed by whitespace or newline
        raw_sentences = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+", text.strip())
            if len(s.strip()) > 0
        ]

        results = []
        for sentence in raw_sentences:
            cited = self.extract_citations(sentence)
            results.append(
                SentenceCitation(
                    sentence=sentence,
                    cited_ids=cited,
                    is_cited=len(cited) > 0,
                )
            )
        return results

    def verify_coverage(
        self,
        answer: str,
        retrieved_ids: list[str],
    ) -> CitationCoverageResult:
        """
        Evaluates citation validity against the retrieved document IDs.
        Identifies valid IDs, hallucinated IDs (cited but not retrieved),
        and uncited statements (hallucination candidates).
        """
        if not answer:
            return CitationCoverageResult(
                total_citations_found=0,
                unique_cited_ids=[],
                valid_citations=[],
                hallucinated_citations=[],
                uncited_statements=[],
                citation_coverage_score=0.0,
                is_abstention=False,
                is_fully_grounded=False,
            )

        # 1. Check for intentional abstention
        answer_lower = answer.lower()
        is_abstention = any(marker in answer_lower for marker in ABSTENTION_MARKERS)

        # 2. Extract citations
        cited_ids = self.extract_citations(answer)
        retrieved_set = set(retrieved_ids)

        valid_citations = [cid for cid in cited_ids if cid in retrieved_set]
        hallucinated_citations = [cid for cid in cited_ids if cid not in retrieved_set]

        # 3. Sentence-level analysis for uncited statements
        sentence_breakdown = self.parse_sentences(answer)
        uncited_statements = []
        for item in sentence_breakdown:
            # Ignore headers, disclaimers, or pure abstention sentences
            if not item.is_cited and not any(m in item.sentence.lower() for m in ABSTENTION_MARKERS):
                # If substantive (> 20 chars), flag as uncited candidate
                if len(item.sentence) > 20:
                    uncited_statements.append(item.sentence)

        # 4. Coverage Score
        if is_abstention:
            coverage_score = 1.0
            is_grounded = len(hallucinated_citations) == 0
        elif len(cited_ids) == 0:
            coverage_score = 0.0
            is_grounded = False
        else:
            coverage_score = round(len(valid_citations) / len(cited_ids), 4)
            is_grounded = len(hallucinated_citations) == 0 and len(uncited_statements) == 0

        return CitationCoverageResult(
            total_citations_found=len(cited_ids),
            unique_cited_ids=cited_ids,
            valid_citations=valid_citations,
            hallucinated_citations=hallucinated_citations,
            uncited_statements=uncited_statements,
            citation_coverage_score=coverage_score,
            is_abstention=is_abstention,
            is_fully_grounded=is_grounded,
        )


if __name__ == "__main__":
    print("=" * 70)
    print("TESTING CITATION EXTRACTOR & VALIDATION")
    print("=" * 70)

    extractor = CitationExtractor()
    retrieved_pool = ["chunk_001", "chunk_002", "chunk_003"]

    # Test Case 1: Fully grounded answer with valid citations
    valid_text = (
        "In fiscal 2025, Apple's total net sales reached $391,035 million [chunk_001]. "
        "iPhone sales accounted for $201,183 million driven by Pro models [chunk_002]."
    )
    res1 = extractor.verify_coverage(valid_text, retrieved_ids=retrieved_pool)
    print("\n[Test 1: Valid Citations]")
    print(f"Cited: {res1.unique_cited_ids}")
    print(f"Valid: {res1.valid_citations}")
    print(f"Hallucinated: {res1.hallucinated_citations}")
    print(f"Coverage Score: {res1.citation_coverage_score}")
    print(f"Is Fully Grounded: {res1.is_fully_grounded}")

    assert res1.is_fully_grounded is True
    assert len(res1.valid_citations) == 2
    assert len(res1.hallucinated_citations) == 0

    # Test Case 2: Hallucinated citation ID (chunk_099 was never retrieved)
    hallucinated_text = (
        "Mac sales were $31,659 million [chunk_001]. "
        "Apple car project was cancelled [chunk_099]."
    )
    res2 = extractor.verify_coverage(hallucinated_text, retrieved_ids=retrieved_pool)
    print("\n[Test 2: Hallucinated Citation ID]")
    print(f"Hallucinated IDs: {res2.hallucinated_citations}")
    print(f"Is Fully Grounded: {res2.is_fully_grounded}")

    assert "chunk_099" in res2.hallucinated_citations
    assert res2.is_fully_grounded is False

    # Test Case 3: Uncited claim (hallucination candidate)
    uncited_text = (
        "Total net sales were $391,035 million [chunk_001]. "
        "Apple will double its dividend next quarter."  # Uncited statement!
    )
    res3 = extractor.verify_coverage(uncited_text, retrieved_ids=retrieved_pool)
    print("\n[Test 3: Uncited Claim]")
    print(f"Uncited Statements: {res3.uncited_statements}")
    assert len(res3.uncited_statements) == 1
    assert res3.is_fully_grounded is False

    # Test Case 4: Correct Abstention
    abstention_text = "The provided documents do not contain sufficient information to answer this question."
    res4 = extractor.verify_coverage(abstention_text, retrieved_ids=retrieved_pool)
    print("\n[Test 4: Clean Abstention]")
    print(f"Is Abstention: {res4.is_abstention}")
    print(f"Coverage Score: {res4.citation_coverage_score}")
    assert res4.is_abstention is True
    assert res4.is_fully_grounded is True

    print("\nAll CitationExtractor tests and assertions passed successfully!")
