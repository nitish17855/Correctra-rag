import os
import sys
from pathlib import Path

# Ensure project root is in sys.path when running directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from abc import ABC, abstractmethod
from typing import Iterator
from dotenv import load_dotenv
from google import genai
from google.genai import types
from generation.prompt_builder import PromptPayload

load_dotenv()


class LLMProvider(ABC):
    """
    Abstract Base Class for all LLM providers.
    Enforces unified interfaces for both full and streaming generation.
    """

    @abstractmethod
    def generate(self, payload: PromptPayload) -> str:
        """Generates a complete response for the given prompt payload."""
        pass

    @abstractmethod
    def generate_stream(self, payload: PromptPayload) -> Iterator[str]:
        """Streams response tokens chunk-by-chunk as an iterator of strings."""
        pass


class GeminiProvider(LLMProvider):
    """
    Concrete LLM Provider connecting to Google Gemini via google-genai SDK.
    Defaults to gemini-3.5-flash-lite with temperature=0.0 for deterministic financial facts.
    """

    def __init__(self, model_name: str = "gemini-3.5-flash-lite", api_key: str | None = None):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.client = genai.Client(api_key=self.api_key, http_options=types.HttpOptions(timeout=30000)) if self.api_key else None

    def generate(self, payload: PromptPayload) -> str:
        if self.client is None:
            raise ValueError("GEMINI_API_KEY is not set. Please check your .env file.")

        import time
        max_retries = 5
        base_delay = 2.0
        last_exception = None

        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=payload.user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=payload.system_prompt,
                        temperature=0.0,
                    ),
                )
                return response.text or ""
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    sleep_time = base_delay * (2 ** attempt)
                    print(f"\n[!] Gemini API temporary error ({type(e).__name__}: {e}). Retrying in {sleep_time:.1f}s (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(sleep_time)
                else:
                    raise last_exception
        return ""

    def generate_stream(self, payload: PromptPayload) -> Iterator[str]:
        if self.client is None:
            raise ValueError("GEMINI_API_KEY is not set. Please check your .env file.")

        response_stream = self.client.models.generate_content_stream(
            model=self.model_name,
            contents=payload.user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=payload.system_prompt,
                temperature=0.0,
            ),
        )

        for chunk in response_stream:
            if chunk.text:
                yield chunk.text


def get_llm_provider(provider: str = "gemini", **kwargs) -> LLMProvider:
    """Factory function to instantiate LLM providers without coupling call sites."""
    provider_lower = provider.lower()
    if provider_lower in ("gemini", "google"):
        return GeminiProvider(**kwargs)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


if __name__ == "__main__":
    print("=" * 70)
    print("TESTING GEMINI LLM PROVIDER (gemini-3.5-flash-lite)")
    print("=" * 70)

    provider = get_llm_provider("gemini")
    test_payload = PromptPayload(
        system_prompt="You are a financial AI. Answer in 5 words or fewer.",
        user_prompt="What is Apple Inc.'s stock ticker symbol?",
    )

    print("\n1. Testing standard generate()...")
    answer = provider.generate(test_payload)
    print(f"Result: '{answer.strip()}'")
    assert "AAPL" in answer

    print("\n2. Testing generate_stream()...")
    stream_output = []
    print("Streaming: ", end="", flush=True)
    for token in provider.generate_stream(test_payload):
        print(token, end="", flush=True)
        stream_output.append(token)
    print()

    full_streamed = "".join(stream_output)
    assert len(full_streamed) > 0
    assert "AAPL" in full_streamed

    print("\nAll LLMProvider tests passed successfully!")
