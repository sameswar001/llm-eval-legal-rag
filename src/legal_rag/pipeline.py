from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from legal_rag.generator import LLMGenerator, default_generator
from legal_rag.retriever import DEFAULT_CORPUS_DIR, TfidfRetriever


@dataclass
class RAGResult:
    answer: str
    contexts: list[str]
    sources: list[str]


class LegalRAGPipeline:
    """The system under test: retrieve contract context, then generate an answer."""

    def __init__(
        self,
        generator: LLMGenerator | None = None,
        corpus_dir: Path = DEFAULT_CORPUS_DIR,
        top_k: int = 3,
    ):
        self.retriever = TfidfRetriever(corpus_dir=corpus_dir)
        self.generator = generator or default_generator()
        self.top_k = top_k

    def answer(self, query: str) -> RAGResult:
        chunks = self.retriever.retrieve(query, top_k=self.top_k)
        contexts = [c.text for c in chunks]
        sources = [c.source for c in chunks]
        answer = self.generator.generate(query, contexts)
        return RAGResult(answer=answer, contexts=contexts, sources=sources)
