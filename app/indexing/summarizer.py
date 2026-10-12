from typing import Protocol

import numpy as np

from app.indexing.embedder import Embedder, l2_normalize
from app.indexing.text import split_sentences


class Summarizer(Protocol):
    def summarize(self, title: str, content: str) -> str: ...


def truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    cut = text[: max_chars - 1].rsplit(" ", 1)[0]
    return cut.rstrip(" ,;:") + "…"


NEAR_DUPLICATE = 0.95


def select_sentences(
    vectors: np.ndarray,
    word_counts: list[int],
    k: int,
    diversity: float = 0.3,
) -> list[int]:
    """Rank sentences for an extractive summary with MMR (maximal marginal relevance).

    Relevance is the similarity to the document centroid (the "topic" of the document), with a
    small bonus for early sentences and a penalty for fragments. MMR then penalizes sentences that
    repeat what is already selected; near-duplicates are never picked twice. Returns indices in
    selection (importance) order.
    """
    n = len(vectors)
    centroid = l2_normalize(vectors.mean(axis=0))
    relevance = vectors @ centroid
    relevance += 0.05 * (1 - np.arange(n) / n)  # lead bias
    relevance -= np.array([0.1 if wc < 5 else 0.0 for wc in word_counts])  # fragments

    selected: list[int] = []
    while len(selected) < min(k, n):
        redundancy = (vectors @ vectors[selected].T).max(axis=1) if selected else np.zeros(n)
        mmr = (1 - diversity) * relevance - diversity * redundancy
        mmr[redundancy >= NEAR_DUPLICATE] = -np.inf
        mmr[selected] = -np.inf
        if np.isneginf(mmr).all():
            break  # only repeats are left
        selected.append(int(np.argmax(mmr)))
    return selected


class ExtractiveSummarizer:
    def __init__(
        self,
        embedder: Embedder,
        max_sentences: int = 3,
        max_chars: int = 500,
        max_candidates: int = 300,
    ) -> None:
        self.embedder = embedder
        self.max_sentences = max_sentences
        self.max_chars = max_chars
        # Bounds the cost for very long documents: only the first N sentences are candidates.
        self.max_candidates = max_candidates

    def summarize(self, title: str, content: str) -> str:
        sentences = [s.text for s in split_sentences(content)][: self.max_candidates]
        if not sentences:
            return ""
        if len(sentences) <= self.max_sentences:
            return truncate(" ".join(sentences), self.max_chars)

        vectors = self.embedder.embed_documents(sentences)
        ranked = select_sentences(
            vectors, [len(s.split()) for s in sentences], k=self.max_sentences
        )
        chosen: list[int] = []
        length = 0
        for i in ranked:
            if chosen and length + len(sentences[i]) + 1 > self.max_chars:
                continue
            chosen.append(i)
            length += len(sentences[i]) + 1
        return truncate(" ".join(sentences[i] for i in sorted(chosen)), self.max_chars)
