"""Swappable generator backends.

Swappable-by-design so the same DeepEval suite can score different
generators against the same retriever — e.g. to compare a cheaper model
against a stronger one on Faithfulness / Legal Precision before a
model-swap decision, which is the kind of evidence a QE function should
be producing.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod

SYSTEM_PROMPT = (
    "You are a contract-review assistant. Answer the user's question ONLY "
    "using the provided contract excerpts. Cite section numbers where the "
    "excerpts include them. If the excerpts do not contain the answer, say "
    "clearly that the contracts provided do not cover that question — do "
    "not guess or invent terms."
)


class LLMGenerator(ABC):
    """Base interface every generator backend must implement."""

    @abstractmethod
    def generate(self, query: str, context: list[str]) -> str:
        ...


class OpenAIGenerator(LLMGenerator):
    def __init__(self, model: str = "gpt-5-nano"):
        from openai import OpenAI  # local import: optional dependency

        self._client = OpenAI()
        self._model = model

    def generate(self, query: str, context: list[str]) -> str:
        context_block = "\n\n".join(f"[Excerpt {i+1}] {c}" for i, c in enumerate(context))
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Contract excerpts:\n{context_block}\n\nQuestion: {query}"},
            ],
        )
        return response.choices[0].message.content or ""


class DeepSeekGenerator(LLMGenerator):
    def __init__(self, model: str = "deepseek-v4-flash"):
        from openai import OpenAI
        self._client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
        )
        self._model = model

    def generate(self, query: str, context: list[str]) -> str:
            context_block = "\n\n".join(f"[Excerpt {i+1}] {c}" for i, c in enumerate(context))
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Contract excerpts:\n{context_block}\n\nQuestion: {query}"},
                ],
            )
            return response.choices[0].message.content or ""


class MockGenerator(LLMGenerator):
    """Deterministic, offline stand-in with no API calls.

    Useful for wiring/smoke-testing the pipeline shape (retrieval,
    tracing, CI plumbing) without a key. NOT a substitute for real
    generator output when actually scoring Faithfulness / Legal
    Precision — those need a real model in the loop.
    """

    def generate(self, query: str, context: list[str]) -> str:
        if not context:
            return "The contracts provided do not cover that question."
        return "Based on the contract excerpts: " + " ".join(context[:1])


def default_generator() -> LLMGenerator:
    """Picks a generator based on whichever API key is available."""
    if os.getenv("OPENAI_API_KEY"):
        return OpenAIGenerator()
    if os.getenv("DEEPSEEK_API_KEY"):
        return DeepSeekGenerator()
    return MockGenerator()
