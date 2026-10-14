import dataclasses
import os
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off", ""}


@dataclass(frozen=True)
class Settings:
    database_url: str = "postgresql://advisor:advisor@localhost:5432/advisor"
    # Where clients reach the service; only used for log messages (compose sets it from APP_PORT).
    public_url: str = "http://localhost:8000"

    # Embeddings.
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    model_cache_dir: str | None = None
    # One intra-op thread keeps background indexing from starving requests on a 1 vCPU box.
    embedding_threads: int = 1

    # Background indexing worker.
    run_worker: bool = True
    worker_poll_interval_s: float = 2.0
    worker_max_attempts: int = 3
    worker_stale_after_s: int = 300

    # Indexing.
    chunk_max_words: int = 120
    summary_max_sentences: int = 3
    summary_max_chars: int = 500

    # Search relevance gates (calibrated for all-MiniLM-L6-v2, see README "Design decisions").
    semantic_min_similarity: float = 0.22
    taxonomy_min_similarity: float = 0.40
    semantic_candidate_chunks: int = 50

    # Optional API key; when set, every request except docs/health/UI must send `X-API-Key`.
    api_key: str | None = None

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "Settings":
        environ = os.environ if environ is None else environ
        values: dict[str, Any] = {}
        for field in dataclasses.fields(cls):
            raw = environ.get(field.name.upper())
            if raw is not None:
                values[field.name] = _parse(field.name, raw, field.type)
        return cls(**values)


def _parse(name: str, raw: str, annotation: object) -> Any:
    raw = raw.strip()
    if annotation is bool:
        if raw.lower() in _TRUE:
            return True
        if raw.lower() in _FALSE:
            return False
        raise ValueError(f"{name.upper()} must be a boolean, got {raw!r}")
    if annotation is int:
        return int(raw)
    if annotation is float:
        return float(raw)
    if annotation == (str | None):
        return raw or None
    return raw


@lru_cache
def get_settings() -> Settings:
    return Settings.from_env()
