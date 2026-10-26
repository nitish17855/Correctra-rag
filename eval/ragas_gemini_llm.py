import sys
import typing as t
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import eval.ragas_compat
from ragas.llms.base import BaseRagasLLM
from langchain_core.prompt_values import PromptValue, StringPromptValue
from langchain_core.outputs import LLMResult, Generation
from generation.llm_provider import GeminiProvider
from generation.prompt_builder import PromptPayload

class GeminiRagasLLM(BaseRagasLLM):
    def __init__(self, provider: t.Optional[GeminiProvider] = None):
        super().__init__()
        self.provider = provider or GeminiProvider(model_name="gemini-3.5-flash-lite")

    def generate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float = 0.01,
        stop: t.Optional[t.List[str]] = None,
        callbacks = None,
    ) -> LLMResult:
        prompt_str = prompt.to_string()
        payload = PromptPayload(system_prompt="You are an expert evaluator.", user_prompt=prompt_str)
        text = self.provider.generate(payload)
        return LLMResult(generations=[[Generation(text=text)]])

    async def agenerate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float = 0.01,
        stop: t.Optional[t.List[str]] = None,
        callbacks = None,
    ) -> LLMResult:
        return self.generate_text(prompt, n, temperature, stop, callbacks)

    def is_finished(self, *args, **kwargs) -> bool:
        return True

if __name__ == "__main__":
    llm = GeminiRagasLLM()
    res = llm.generate_text(StringPromptValue(text="Say hello in 3 words."))
    print("Result:", res.generations[0][0].text)
