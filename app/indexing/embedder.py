"""Text embedders. All vectors are L2-normalized so cosine similarity is a dot product."""

import hashlib
import re
import threading
from collections.abc import Sequence
from typing import Protocol

import numpy as np

from app.indexing.text import normalize


class Embedder(Protocol):
    model_name: str
    dim: int

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Return an `(n, dim)` float32 matrix of normalized passage embeddings."""
        ...

    def embed_query(self, text: str) -> np.ndarray:
        """Return a `(dim,)` float32 normalized query embedding."""
        ...


def l2_normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=-1, keepdims=True)
    return (m / np.where(norms == 0, 1.0, norms)).astype(np.float32)


class FastEmbedEmbedder:
    """Small ONNX encoder (no torch, CPU-only) via `fastembed`."""

    def __init__(self, model_name: str, cache_dir: str | None = None, threads: int = 1) -> None:
        from fastembed import TextEmbedding

        self.model_name = model_name
        self._model = TextEmbedding(model_name, cache_dir=cache_dir, threads=threads)
        self.dim = int(self._model.embedding_size)
        # The worker thread and request threads share one ONNX session.
        self._lock = threading.Lock()

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        with self._lock:
            vectors = list(self._model.passage_embed(list(texts)))
        return l2_normalize(np.asarray(vectors, dtype=np.float32))

    def embed_query(self, text: str) -> np.ndarray:
        with self._lock:
            vector = next(iter(self._model.query_embed(text)))
        return l2_normalize(np.asarray(vector, dtype=np.float32))


class HashingEmbedder:
    """Deterministic bag-of-words embedder for tests and offline runs.

    It captures lexical overlap only (no semantics), which is enough to exercise the plumbing.
    """

    model_name = "hashing"

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def _embed(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        for token in re.findall(r"[a-z0-9]+", normalize(text)):
            stem = token[:5]  # crude stemming: "bills" and "bill" share buckets
            digest = hashlib.blake2b(stem.encode(), digest_size=16).digest()
            # Several signed buckets per token: a collision with another token only adds a
            # little noise instead of making two unrelated words look identical.
            for i in range(0, 16, 4):
                h = int.from_bytes(digest[i : i + 4], "little")
                vector[h % self.dim] += 1.0 if h & 0x80000000 else -1.0
        if not vector.any():
            vector[0] = 1.0  # avoid zero vectors: cosine distance would be NaN
        return vector

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return l2_normalize(np.stack([self._embed(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return l2_normalize(self._embed(text))


def create_embedder(
    model_name: str, dim: int, cache_dir: str | None = None, threads: int = 1
) -> Embedder:
    embedder: Embedder
    if model_name == "hashing":
        embedder = HashingEmbedder(dim)
    else:
        embedder = FastEmbedEmbedder(model_name, cache_dir=cache_dir, threads=threads)
    if embedder.dim != dim:
        raise ValueError(
            f"Embedding model {model_name!r} produces {embedder.dim}-d vectors, "
            f"but the database schema expects {dim}-d vectors"
        )
    return embedder
