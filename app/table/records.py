"""Row types returned by the database layer (the schema itself lives in db/migrations/*.sql)."""

import enum
import uuid
from dataclasses import dataclass
from datetime import datetime

import numpy as np

# Output size of the embedding model; fixed by `document_chunks.embedding vector(384)`.
EMBEDDING_DIM = 384


class DocumentStatus(enum.StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ClientRecord:
    id: uuid.UUID
    first_name: str
    last_name: str
    email: str
    description: str | None
    social_links: list[str]
    created_at: datetime


@dataclass(frozen=True, slots=True)
class DocumentRecord:
    id: uuid.UUID
    client_id: uuid.UUID
    title: str
    content: str
    created_at: datetime
    status: str
    summary: str | None
    attempts: int
    error: str | None
    processed_at: datetime | None
    processing_ms: int | None


@dataclass(frozen=True, slots=True)
class ChunkRecord:
    ordinal: int
    text: str
    char_start: int
    char_end: int


@dataclass(frozen=True, slots=True)
class NewChunk:
    ordinal: int
    text: str
    char_start: int
    char_end: int
    embedding: np.ndarray


@dataclass(frozen=True, slots=True)
class LexemeRecord:
    lexeme: str
    count: int
    weights: list[str]


@dataclass(frozen=True, slots=True)
class Job:
    """A document claimed by the indexing worker; `locked_at` identifies this claim."""

    id: uuid.UUID
    title: str
    content: str
    attempts: int
    locked_at: datetime


# -- search ----------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ClientMatch:
    id: uuid.UUID
    first_name: str
    last_name: str
    email: str
    description: str | None
    social_links: list[str]
    created_at: datetime
    word_similarity: float  # pg_trgm word_similarity(query, identity terms)
    description_rank: float | None  # ts_rank_cd when the description matches the query


@dataclass(frozen=True, slots=True)
class LexicalMatch:
    id: uuid.UUID
    rank: float
    in_title: bool


@dataclass(frozen=True, slots=True)
class NearestDocument:
    document_id: uuid.UUID
    similarity: float  # cosine similarity of the best chunk
    best_chunk: str
