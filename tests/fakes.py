import hashlib
import re
from collections.abc import Sequence

import numpy as np

from app.indexing.embedder import l2_normalize
from app.indexing.text import normalize


class HashingEmbedder:
    """Deterministic bag-of-words stand-in for the real embedding model (fast, offline).

    It captures lexical overlap only (no semantics), which is enough to exercise the plumbing;
    search quality with the real model is covered by tests/eval.
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
