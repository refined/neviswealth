"""Text embedders. All vectors are L2-normalized so cosine similarity is a dot product."""

import threading
from collections.abc import Sequence
from typing import Protocol

import numpy as np

from app.table.records import EMBEDDING_DIM


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
    def __init__(
        self,
        model_name: str,
        cache_dir: str | None = None,
        threads: int = 1,
        batch_size: int = 32,
    ) -> None:
        from fastembed import TextEmbedding

        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}")
        self.model_name = model_name
        self._batch_size = batch_size
        self._model = TextEmbedding(model_name, cache_dir=cache_dir, threads=threads)
        self.dim = int(self._model.embedding_size)
        # The worker thread and request threads share one ONNX session.
        self._lock = threading.Lock()

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        with self._lock:
            vectors = list(self._model.passage_embed(list(texts), batch_size=self._batch_size))
        return l2_normalize(np.asarray(vectors, dtype=np.float32))

    def embed_query(self, text: str) -> np.ndarray:
        with self._lock:
            vector = next(iter(self._model.query_embed(text)))
        return l2_normalize(np.asarray(vector, dtype=np.float32))


def create_embedder(
    model_name: str, cache_dir: str | None = None, threads: int = 1, batch_size: int = 32
) -> Embedder:
    embedder = FastEmbedEmbedder(
        model_name, cache_dir=cache_dir, threads=threads, batch_size=batch_size
    )
    if embedder.dim != EMBEDDING_DIM:
        raise ValueError(
            f"Embedding model {model_name!r} produces {embedder.dim}-d vectors, "
            f"but the database schema expects {EMBEDDING_DIM}-d vectors"
        )
    return embedder
