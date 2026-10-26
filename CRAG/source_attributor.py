import re
from dataclasses import dataclass, field
from typing import Optional
from langchain_core.documents import Document

# Regex to match both [DOC:chunk_id] and [WEB:url] formats, as well as comma-separated
CRAG_CITATION_REGEX = re.compile(
    r"\[((?:DOC|WEB):[^\]]+)\]",
    re.IGNORECASE,
)

CRAG_SYSTEM_PROMPT = """You are an expert financial and corporate intelligence assistant.
Your task is to answer user queries with absolute factual precision based EXCLUSIVELY on the provided <retrieved_context>.

### MANDATORY CITATION RULES:
1. PER-CLAIM CITATIONS: Every factual claim, number, date, or statement must be immediately followed by its exact bracketed source identifier.
2. DUAL-SOURCE TAGGING:
   - For facts from internal company documents, cite: [DOC:chunk_id] (e.g., "Apple net sales increased by 6% in 2025 [DOC:chunk_sales_01].")
   - For facts from external web search, cite: [WEB:url] (e.g., "Australia won the 2023 ICC Cricket World Cup [WEB:https://en.wikipedia.org/wiki/2023_Cricket_World_Cup].")
3. STRICT GROUNDING: Never speculate, extrapolate, or introduce knowledge from outside the provided context. If a fact is not in the context, it does not exist.
4. NO UNGROUNDED CITATIONS: Only cite IDs that explicitly appear in the <retrieved_context>. Never invent citation IDs.
5. EXPLICIT ABSTENTION: If the context contains insufficient information to answer the question, state:
   "The provided documents and web search do not contain sufficient information to answer this question."
"""

CRAG_USER_PROMPT_TEMPLATE = """{context_str}

User Question: {query}

Provide a direct, comprehensive answer citing every single claim with [DOC:chunk_id] for internal evidence or [WEB:url] for web evidence."""


@dataclass
class ResolvedCitation:
    raw_tag: str
    source_type: str                  # "INTERNAL" or "WEB"
    identifier: str                   # chunk_id or URL
    is_valid: bool                    # Present in retrieved documents
    matched_document: Optional[Document] = None


@dataclass
class AttributionMetrics:
    total_citations: int
    valid_citations: int
    hallucinated_citations: int
    attribution_accuracy: float       # valid / total (1.0 = perfect attribution)
    internal_citation_count: int
    web_citation_count: int
    resolved_citations: list[ResolvedCitation]
    is_abstention: bool


class SourceAttributor:
    """
    Manages dual-source attribution (INTERNAL vs WEB) for Corrective RAG.
    1. Formats context with [DOC:chunk_id] and [WEB:url] XML wrappers.
    2. Resolves citation tags in LLM answers to original source documents.
    3. Computes attribution_accuracy metric (valid citations vs hallucinated citations).
    """

    def __init__(self, citation_regex: re.Pattern = CRAG_CITATION_REGEX):
        self.pattern = citation_regex

    def format_context(self, documents: list[Document]) -> str:
        """
        Formats retrieved internal documents and web snippets into
        disambiguated XML evidence tags for the generation prompt.
        """
        if not documents:
            return "<retrieved_context>\n</retrieved_context>"

        entries = ["<retrieved_context>"]
        for rank, doc in enumerate(documents, 1):
            is_web = doc.metadata.get("is_web", False)
            if is_web:
                url = doc.metadata.get("source", "web")
                title = doc.metadata.get("title", "")
                tag = f"WEB:{url}"
                entries.append(
                    f'<evidence id="{tag}" source="WEB" rank="{rank}" title="{title}">\n'
                    f"{doc.page_content.strip()}\n"
                    f"</evidence>"
                )
            else:
                chunk_id = doc.metadata.get("chunk_id", f"chunk_{rank:03d}")
                tag = f"DOC:{chunk_id}"
                src = doc.metadata.get("source", "filing")
                entries.append(
                    f'<evidence id="{tag}" source="INTERNAL" rank="{rank}" file="{src}">\n'
                    f"{doc.page_content.strip()}\n"
                    f"</evidence>"
                )
        entries.append("</retrieved_context>")
        return "\n".join(entries)

    def extract_citations(self, text: str) -> list[str]:
        """
        Extracts all citation strings matching [DOC:...] or [WEB:...] from text.
        """
        if not text:
            return []
        matches = self.pattern.findall(text)
        unique: list[str] = []
        for m in matches:
            clean = m.strip()
            if clean and clean not in unique:
                unique.append(clean)
        return unique

    def resolve(
        self, citation_tag: str, retrieved_docs: list[Document]
    ) -> ResolvedCitation:
        """
        Resolves a single citation tag (e.g., 'DOC:chunk_01' or 'WEB:https://...')
        against the retrieved document pool to verify its origin and validity.
        """
        clean_tag = citation_tag.strip().strip("[]")
        source_type = "WEB" if clean_tag.upper().startswith("WEB:") else "INTERNAL"
        identifier = clean_tag[4:] if len(clean_tag) > 4 else clean_tag

        matched_doc = None
        for doc in retrieved_docs:
            if source_type == "WEB":
                doc_url = doc.metadata.get("source", "")
                if doc.metadata.get("is_web") and (identifier in doc_url or doc_url in identifier):
                    matched_doc = doc
                    break
            else:
                doc_cid = doc.metadata.get("chunk_id", "")
                if not doc.metadata.get("is_web") and identifier == doc_cid:
                    matched_doc = doc
                    break

        return ResolvedCitation(
            raw_tag=clean_tag,
            source_type=source_type,
            identifier=identifier,
            is_valid=matched_doc is not None,
            matched_document=matched_doc,
        )

    def attribution_accuracy(
        self, answer: str, retrieved_docs: list[Document]
    ) -> AttributionMetrics:
        """
        Calculates attribution accuracy of an answer against retrieved context.
        Metric = (Valid Citations) / (Total Citations Cited).
        Flags hallucinated citations (citations cited by LLM not present in retrieved context).
        """
        if not answer or not answer.strip():
            return AttributionMetrics(
                total_citations=0,
                valid_citations=0,
                hallucinated_citations=0,
                attribution_accuracy=0.0,
                internal_citation_count=0,
                web_citation_count=0,
                resolved_citations=[],
                is_abstention=False,
            )

        # Check for abstention
        answer_lower = answer.lower()
        is_abstention = (
            "not contain sufficient information" in answer_lower
            or "insufficient information" in answer_lower
        )

        extracted_tags = self.extract_citations(answer)
        resolved_list = [self.resolve(tag, retrieved_docs) for tag in extracted_tags]

        valid_count = sum(1 for r in resolved_list if r.is_valid)
        hallucinated_count = len(resolved_list) - valid_count
        internal_count = sum(1 for r in resolved_list if r.source_type == "INTERNAL")
        web_count = sum(1 for r in resolved_list if r.source_type == "WEB")

        if is_abstention:
            accuracy = 1.0
        elif len(resolved_list) == 0:
            accuracy = 0.0
        else:
            accuracy = round(valid_count / len(resolved_list), 4)

        return AttributionMetrics(
            total_citations=len(resolved_list),
            valid_citations=valid_count,
            hallucinated_citations=hallucinated_count,
            attribution_accuracy=accuracy,
            internal_citation_count=internal_count,
            web_citation_count=web_count,
            resolved_citations=resolved_list,
            is_abstention=is_abstention,
        )


if __name__ == "__main__":
    print("=" * 70)
    print("TESTING SOURCE ATTRIBUTOR & ATTRIBUTION ACCURACY METRIC")
    print("=" * 70)

    attributor = SourceAttributor()

    # Sample retrieved pool containing both internal and web evidence
    retrieved_pool = [
        Document(
            page_content="Apple net sales were $416,161 million in 2025.",
            metadata={"chunk_id": "chunk_sales_01", "is_web": False, "source": "apple_10k.htm"},
        ),
        Document(
            page_content="Australia defeated India by 6 wickets to win the 2023 ICC World Cup.",
            metadata={"source": "https://en.wikipedia.org/wiki/2023_World_Cup", "is_web": True, "title": "2023 World Cup"},
        ),
    ]

    # Test 1: Context formatting
    context_str = attributor.format_context(retrieved_pool)
    print("\n--- Formatted Context Preview ---")
    print(context_str[:300] + "...\n")
    assert "DOC:chunk_sales_01" in context_str
    assert "WEB:https://en.wikipedia.org/wiki/2023_World_Cup" in context_str

    # Test 2: Valid Dual-Source Citations (100% Attribution Accuracy)
    answer_valid = (
        "Apple net sales reached $416,161 million in 2025 [DOC:chunk_sales_01]. "
        "Separately, Australia won the 2023 World Cup [WEB:https://en.wikipedia.org/wiki/2023_World_Cup]."
    )
    metric1 = attributor.attribution_accuracy(answer_valid, retrieved_pool)
    print(f"[Test 2: Perfect Attribution]")
    print(f"Total: {metric1.total_citations}, Valid: {metric1.valid_citations}, Accuracy: {metric1.attribution_accuracy}")
    print(f"Internal Citations: {metric1.internal_citation_count}, Web Citations: {metric1.web_citation_count}")
    assert metric1.attribution_accuracy == 1.0
    assert metric1.valid_citations == 2
    assert metric1.hallucinated_citations == 0

    # Test 3: Hallucinated Citations (Cited ID not in retrieved pool)
    answer_hallucinated = (
        "Apple net sales were $416,161 million [DOC:chunk_sales_01]. "
        "Apple car will launch next year [DOC:chunk_phantom_99]."
    )
    metric2 = attributor.attribution_accuracy(answer_hallucinated, retrieved_pool)
    print(f"\n[Test 3: Hallucinated Citation]")
    print(f"Total: {metric2.total_citations}, Valid: {metric2.valid_citations}, Accuracy: {metric2.attribution_accuracy}")
    print(f"Hallucinated Citations: {metric2.hallucinated_citations}")
    assert metric2.attribution_accuracy == 0.5
    assert metric2.hallucinated_citations == 1

    print("\nAll SourceAttributor tests passed successfully!")
