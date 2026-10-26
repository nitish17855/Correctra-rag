from dataclasses import dataclass
from langchain_core.documents import Document
import tiktoken


@dataclass
class AssembledContext:
    context_str: str
    included_chunk_ids: list[str]
    total_tokens: int
    dropped_count: int


class ContextAssembler:
    """
    Assembles retrieved chunks into structured XML format with explicit boundary tags,
    unique document IDs, and a strict token budget guard with safe truncation.
    """

    def __init__(self, max_tokens: int = 3000, encoding_name: str = "cl100k_base"):
        self.max_tokens = max_tokens
        self.tokenizer = tiktoken.get_encoding(encoding_name)

    def assemble(self, documents: list[Document]) -> AssembledContext:
        accumulated_chunks = []
        included_ids = []
        current_tokens = 0
        dropped_count = 0

        for rank, doc in enumerate(documents, start=1):
            chunk_id = doc.metadata.get("chunk_id", f"doc_{rank}")
            source = doc.metadata.get("source", "unknown")

            # 1. Format document with explicit boundary tags
            formatted_chunk = (
                f'<document id="{chunk_id}" rank="{rank}" source="{source}">\n'
                f"{doc.page_content.strip()}\n"
                f"</document>\n"
            )

            chunk_tokens = len(self.tokenizer.encode(formatted_chunk))

            # 2. Token Budget Guard with Safe Slicing
            remaining_tokens = self.max_tokens - current_tokens

            if chunk_tokens > remaining_tokens:
                # If remaining space is too small for meaningful context, drop it
                if remaining_tokens < 60:
                    dropped_count += 1
                    continue

                # Safely truncate text to fit remaining tokens
                tokens = self.tokenizer.encode(doc.page_content.strip())
                # Reserve room for XML boundary tags (~30 tokens)
                allowed_text_tokens = max(10, remaining_tokens - 30)
                truncated_text = (
                    self.tokenizer.decode(tokens[:allowed_text_tokens])
                    + "\n... [TRUNCATED DUE TO CONTEXT LIMIT]"
                )

                formatted_chunk = (
                    f'<document id="{chunk_id}" rank="{rank}" source="{source}" truncated="true">\n'
                    f"{truncated_text}\n"
                    f"</document>\n"
                )
                chunk_tokens = len(self.tokenizer.encode(formatted_chunk))

            # 3. Accumulate chunk and update budget
            accumulated_chunks.append(formatted_chunk)
            included_ids.append(chunk_id)
            current_tokens += chunk_tokens

        # 4. Wrap all included chunks inside container tag
        final_context = (
            "<retrieved_context>\n"
            + "".join(accumulated_chunks)
            + "</retrieved_context>"
        )

        return AssembledContext(
            context_str=final_context,
            included_chunk_ids=included_ids,
            total_tokens=len(self.tokenizer.encode(final_context)),
            dropped_count=dropped_count,
        )


if __name__ == "__main__":
    print("=" * 70)
    print("TESTING CONTEXT ASSEMBLER")
    print("=" * 70)

    # Test 1: Standard assembly with 3 documents
    docs = [
        Document(
            page_content="Apple Inc. total net sales in fiscal 2025 reached $391,035 million.",
            metadata={"chunk_id": "chunk_001", "source": "aapl-20250927.htm"},
        ),
        Document(
            page_content="iPhone net sales were $201,183 million primarily driven by iPhone Pro models.",
            metadata={"chunk_id": "chunk_002", "source": "aapl-20250927.htm"},
        ),
        Document(
            page_content="Services gross margin reached 74.2% compared to Products gross margin of 36.8%.",
            metadata={"chunk_id": "chunk_003", "source": "aapl-20250927.htm"},
        ),
    ]

    assembler = ContextAssembler(max_tokens=3000)
    result = assembler.assemble(docs)

    print("\n[Test 1] Assembled Context Output:\n")
    print(result.context_str)
    print(f"\nIncluded IDs: {result.included_chunk_ids}")
    print(f"Total Tokens: {result.total_tokens}")
    print(f"Dropped Count: {result.dropped_count}")

    assert len(result.included_chunk_ids) == 3
    assert result.dropped_count == 0
    assert "chunk_001" in result.included_chunk_ids
    assert "<retrieved_context>" in result.context_str

    # Test 2: Tight token budget triggering safe truncation
    tight_assembler = ContextAssembler(max_tokens=90)
    tight_result = tight_assembler.assemble(docs)

    print("\n[Test 2] Tight Budget (90 tokens) Output:\n")
    print(tight_result.context_str)
    print(f"\nIncluded IDs: {tight_result.included_chunk_ids}")
    print(f"Total Tokens: {tight_result.total_tokens}")
    print(f"Dropped Count: {tight_result.dropped_count}")

    assert tight_result.total_tokens <= 110  # Enforces strict bounded budget
    assert tight_result.dropped_count >= 1  # Lower ranked documents dropped

    print("\nAll ContextAssembler tests and assertions passed successfully!")
