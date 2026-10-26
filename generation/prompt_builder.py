from dataclasses import dataclass


DEFAULT_SYSTEM_PROMPT = """You are an expert SEC financial analyst assistant specialized in corporate filings (Form 10-K).
Your task is to answer user queries with absolute factual precision based EXCLUSIVELY on the provided <retrieved_context>.

### CRITICAL RULES:
1. MANDATORY CITATIONS: Every single factual statement, number, percentage, or date must be immediately followed by the exact bracketed chunk identifier that supports it, e.g., "In fiscal year 2025, total net sales reached $391,035 million [chunk_001]."
2. STRICT GROUNDING: Never speculate, extrapolate, or introduce knowledge from outside the provided context. If a fact is not explicitly present in the excerpts, it does not exist for this conversation.
3. EXPLICIT ABSTENTION: If the provided documents do not contain sufficient information to answer the question, or if the question asks about data not present in the filing, you MUST state:
   "The provided documents do not contain sufficient information to answer this question."
   Do NOT attempt to invent an answer or pull from external knowledge.
4. NO UNGROUNDED CITATIONS: Only cite document IDs that actually appear in the <retrieved_context>. Never invent citation numbers.
"""

DEFAULT_USER_PROMPT_TEMPLATE = """{context_str}

User Question: {query}

Provide a direct, well-structured answer citing every claim with its supporting [chunk_id]."""


@dataclass
class PromptPayload:
    system_prompt: str
    user_prompt: str

    def to_messages(self) -> list[dict[str, str]]:
        """Converts to standard chat messages format used by major LLM providers."""
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_prompt},
        ]


class PromptBuilder:
    """
    Builds strict grounding prompts that enforce per-claim [chunk-id] citations,
    zero-hallucination constraints, and unambiguous abstention directives.
    """

    def __init__(
        self,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        user_template: str = DEFAULT_USER_PROMPT_TEMPLATE,
    ):
        self.system_prompt = system_prompt
        self.user_template = user_template

    def build(self, query: str, context_str: str) -> PromptPayload:
        """
        Builds both system and user prompts combining the user's query
        with the assembled retrieved context.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        clean_context = context_str.strip() if context_str else "<retrieved_context>\n</retrieved_context>"
        user_prompt = self.user_template.format(
            context_str=clean_context,
            query=query.strip(),
        )

        return PromptPayload(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
        )


if __name__ == "__main__":
    print("=" * 70)
    print("TESTING PROMPT BUILDER")
    print("=" * 70)

    sample_context = (
        "<retrieved_context>\n"
        '<document id="chunk_001" rank="1" source="aapl-20250927.htm">\n'
        "Total net sales were $391,035 million in 2025 compared to $383,285 million in 2024.\n"
        "</document>\n"
        '<document id="chunk_002" rank="2" source="aapl-20250927.htm">\n'
        "iPhone net sales were $201,183 million in 2025.\n"
        "</document>\n"
        "</retrieved_context>"
    )

    query = "What were Apple's net sales and iPhone sales in 2025?"

    builder = PromptBuilder()
    payload = builder.build(query=query, context_str=sample_context)

    print("\n[System Prompt Preview]:\n")
    print(payload.system_prompt[:300] + "...\n")

    print("[User Prompt Preview]:\n")
    print(payload.user_prompt)

    messages = payload.to_messages()
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "chunk_001" in payload.user_prompt
    assert "MANDATORY CITATIONS" in payload.system_prompt
    assert "EXPLICIT ABSTENTION" in payload.system_prompt

    print("\nAll PromptBuilder assertions passed successfully!")
