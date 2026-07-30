"""A simple TF-IDF(Term Frequency - Inverse Documents Frequency) retriever over a corpus of legal contracts.

Deliberately dependency-light (scikit-learn only) so the retrieval step
is fast, deterministic, and easy to reason about when debugging DeepEval
contextual-precision / contextual-recall failures — if the retriever is
simple, a low score almost always points at the generator, not noise
from an opaque vector store.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_CORPUS_DIR = Path(__file__).resolve().parents[2] / "data" / "contracts"


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str
    chunk_id: int


def _split_into_paragraphs(text: str) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return paragraphs


def load_corpus(corpus_dir: Path = DEFAULT_CORPUS_DIR) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(corpus_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        for i, paragraph in enumerate(_split_into_paragraphs(text)):
            chunks.append(Chunk(text=paragraph, source=path.name, chunk_id=i))
    if not chunks:
        raise FileNotFoundError(f"No .txt contracts found in {corpus_dir}")
    return chunks


class TfidfRetriever:
    """Retrieves the top-k most relevant contract paragraphs for a query."""

    def __init__(self, corpus_dir: Path = DEFAULT_CORPUS_DIR):
        self.chunks = load_corpus(corpus_dir)
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = self._vectorizer.fit_transform([c.text for c in self.chunks])

    def retrieve(self, query: str, top_k: int = 3) -> list[Chunk]:
        query_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._matrix).flatten()
        top_indices = scores.argsort()[::-1][:top_k]
        return [self.chunks[i] for i in top_indices if scores[i] > 0]
