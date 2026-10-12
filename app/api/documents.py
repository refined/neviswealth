import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from psycopg import Connection

from app import schemas
from app.api.deps import Container, get_conn, get_container
from app.table import documents
from app.table.records import DocumentRecord

router = APIRouter(tags=["documents"])

NOT_FOUND = {404: {"description": "Document not found"}}


def get_document_or_404(conn: Connection, document_id: uuid.UUID) -> DocumentRecord:
    document = documents.get_document(conn, document_id)
    if document is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Document not found")
    return document


@router.get("/documents/{document_id}", response_model=schemas.Document, responses=NOT_FOUND)
def get_document(document_id: uuid.UUID, conn: Connection = Depends(get_conn)) -> DocumentRecord:
    return get_document_or_404(conn, document_id)


@router.get(
    "/documents/{document_id}/index", response_model=schemas.DocumentIndex, responses=NOT_FOUND
)
def get_document_index(
    document_id: uuid.UUID,
    conn: Connection = Depends(get_conn),
    container: Container = Depends(get_container),
) -> schemas.DocumentIndex:
    """Inspect what indexing produced: chunks, full-text lexemes, summary and timings."""
    document = get_document_or_404(conn, document_id)
    return schemas.DocumentIndex(
        document_id=document.id,
        status=document.status,
        attempts=document.attempts,
        error=document.error,
        processed_at=document.processed_at,
        processing_ms=document.processing_ms,
        embedding_model=container.embedder.model_name,
        summary=document.summary,
        chunks=[schemas.Chunk.model_validate(c) for c in documents.list_chunks(conn, document_id)],
        lexemes=[
            schemas.Lexeme.model_validate(lx) for lx in documents.top_lexemes(conn, document_id)
        ],
    )


@router.post(
    "/documents/{document_id}/reprocess",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=schemas.Document,
    responses=NOT_FOUND,
)
def reprocess_document(
    document_id: uuid.UUID,
    conn: Connection = Depends(get_conn),
    container: Container = Depends(get_container),
) -> DocumentRecord:
    """Queue the document for indexing again (e.g. after a failure or a model change)."""
    document = documents.requeue_document(conn, document_id)
    if document is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Document not found")
    conn.commit()
    container.worker.notify()
    return document
