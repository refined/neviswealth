from collections.abc import Iterator, Sequence
from typing import ClassVar

import fastembed
import numpy as np
import pytest

from app.indexing.embedder import FastEmbedEmbedder

DIM = 384


class FakeTextEmbedding:
    """Stands in for fastembed's ONNX model and records how it was called."""

    batch_sizes: ClassVar[list[int]] = []

    def __init__(self, model_name: str, cache_dir: str | None = None, threads: int = 1) -> None:
        self.embedding_size = DIM

    def passage_embed(self, texts: Sequence[str], batch_size: int = 256) -> Iterator[np.ndarray]:
        self.batch_sizes.append(batch_size)
        for _ in texts:
            yield np.ones(DIM, dtype=np.float32)


@pytest.fixture(autouse=True)
def fake_model(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeTextEmbedding.batch_sizes = []
    monkeypatch.setattr(fastembed, "TextEmbedding", FakeTextEmbedding)


def test_documents_are_embedded_in_batches_of_the_configured_size() -> None:
    vectors = FastEmbedEmbedder("fake", batch_size=8).embed_documents([f"t{i}" for i in range(20)])

    assert FakeTextEmbedding.batch_sizes == [8]
    assert vectors.shape == (20, DIM)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)


def test_default_batch_size_is_32() -> None:
    FastEmbedEmbedder("fake").embed_documents(["text"])

    assert FakeTextEmbedding.batch_sizes == [32]


def test_empty_input_does_not_call_the_model() -> None:
    vectors = FastEmbedEmbedder("fake").embed_documents([])

    assert vectors.shape == (0, DIM)
    assert FakeTextEmbedding.batch_sizes == []


@pytest.mark.parametrize("batch_size", [0, -1])
def test_non_positive_batch_size_is_rejected(batch_size: int) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        FastEmbedEmbedder("fake", batch_size=batch_size)
