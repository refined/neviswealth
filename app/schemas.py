import uuid
from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field, StringConstraints

DocumentStatusLiteral = Literal["pending", "processing", "ready", "failed"]

PersonName = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]
Description = Annotated[str, StringConstraints(strip_whitespace=True, max_length=5000)]
SocialLink = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]
Title = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=300)]
Content = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200_000)]


# -- clients ---------------------------------------------------------------------------------


class ClientCreate(BaseModel):
    first_name: PersonName
    last_name: PersonName
    email: EmailStr
    description: Description | None = None
    social_links: list[SocialLink] = Field(default_factory=list, max_length=20)

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "first_name": "John",
                    "last_name": "Doe",
                    "email": "john.doe@neviswealth.com",
                    "description": "Senior partner interested in sustainable investing.",
                    "social_links": ["https://www.linkedin.com/in/john-doe"],
                }
            ]
        }
    )


class Client(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    first_name: str
    last_name: str
    email: str
    description: str | None
    social_links: list[str]
    created_at: datetime


# -- documents -------------------------------------------------------------------------------


class DocumentCreate(BaseModel):
    title: Title
    content: Content

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "title": "Electricity bill - March 2026",
                    "content": "Monthly statement. Service address: 1450 Brickell Avenue, Miami. "
                    "Amount due: $143.27.",
                }
            ]
        }
    )


class Document(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    client_id: uuid.UUID
    title: str
    content: str
    created_at: datetime
    status: DocumentStatusLiteral = Field(
        description="Indexing state. Lexical search works immediately; semantic search and the "
        "summary become available once the status is `ready`."
    )
    summary: str | None = Field(description="Extractive summary, set when indexing finishes.")
    processed_at: datetime | None
    error: str | None


class Chunk(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    ordinal: int
    text: str
    char_start: int
    char_end: int


class Lexeme(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    lexeme: str
    count: int
    weights: list[str] = Field(description="Postgres weights: A = title, B = content.")


class DocumentIndex(BaseModel):
    """What the indexing pipeline produced for a document (used by the inspector UI)."""

    document_id: uuid.UUID
    status: DocumentStatusLiteral
    attempts: int
    error: str | None
    processed_at: datetime | None
    processing_ms: int | None
    embedding_model: str
    summary: str | None
    chunks: list[Chunk]
    lexemes: list[Lexeme] = Field(description="Most frequent lexemes in the full-text index.")


# -- search ----------------------------------------------------------------------------------


class _HitBase(BaseModel):
    score: float = Field(ge=0, le=1, description="Relevance in [0, 1]; results are sorted by it.")
    matched_on: list[str] = Field(
        description="Why the item matched: `name`, `email`, `social_links`, `description`, "
        "`title`, `content`, `semantic` or `document_type:<type>`."
    )
    highlight: str | None = Field(
        description="HTML-escaped snippet with matches wrapped in `<mark>` tags."
    )


class ClientHit(_HitBase):
    type: Literal["client"] = "client"
    client: Client


class DocumentHit(_HitBase):
    type: Literal["document"] = "document"
    document: Document


SearchHit = Annotated[ClientHit | DocumentHit, Field(discriminator="type")]


class Health(BaseModel):
    status: Literal["ok"]
    database: Literal["ok"]
    embedding_model: str
    worker_running: bool
