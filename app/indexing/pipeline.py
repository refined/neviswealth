from dataclasses import dataclass

from app.indexing.embedder import Embedder
from app.indexing.summarizer import Summarizer
from app.indexing.text import chunk_sentences, split_sentences
from app.table.records import NewChunk


@dataclass(frozen=True)
class IndexedDocument:
    chunks: list[NewChunk]
    summary: str


def chunk_embedding_input(title: str, chunk_text: str) -> str:
    # Each chunk carries the title so a chunk like "Amount due: $143" still knows it is a bill.
    return f"{title}. {chunk_text}"


def index_document(
    title: str,
    content: str,
    embedder: Embedder,
    summarizer: Summarizer,
    chunk_max_words: int,
) -> IndexedDocument:
    spans = chunk_sentences(content, split_sentences(content), chunk_max_words)
    vectors = embedder.embed_documents([chunk_embedding_input(title, s.text) for s in spans])
    chunks = [
        NewChunk(i, span.text, span.start, span.end, vector)
        for i, (span, vector) in enumerate(zip(spans, vectors, strict=True))
    ]
    return IndexedDocument(chunks=chunks, summary=summarizer.summarize(title, content))
