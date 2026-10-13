import uuid

from fastapi import APIRouter, status

from app import schemas
from app.api.deps import DocumentServiceDep
from app.table.records import DocumentRecord

router = APIRouter(tags=["documents"])

NOT_FOUND = {404: {"description": "Document not found"}}


@router.get("/documents/{document_id}", response_model=schemas.Document, responses=NOT_FOUND)
def get_document(document_id: uuid.UUID, documents: DocumentServiceDep) -> DocumentRecord:
    return documents.get(document_id)


@router.get(
    "/documents/{document_id}/index", response_model=schemas.DocumentIndex, responses=NOT_FOUND
)
def get_document_index(
    document_id: uuid.UUID, documents: DocumentServiceDep
) -> schemas.DocumentIndex:
    """Inspect what indexing produced: chunks, full-text lexemes, summary and timings."""
    info = documents.index(document_id)
    document = info.document
    return schemas.DocumentIndex(
        document_id=document.id,
        status=document.status,
        attempts=document.attempts,
        error=document.error,
        processed_at=document.processed_at,
        processing_ms=document.processing_ms,
        embedding_model=info.embedding_model,
        summary=document.summary,
        chunks=[schemas.Chunk.model_validate(c) for c in info.chunks],
        lexemes=[schemas.Lexeme.model_validate(lx) for lx in info.lexemes],
    )


@router.post(
    "/documents/{document_id}/reprocess",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=schemas.Document,
    responses=NOT_FOUND,
)
def reprocess_document(document_id: uuid.UUID, documents: DocumentServiceDep) -> DocumentRecord:
    """Queue the document for indexing again (e.g. after a failure or a model change)."""
    return documents.reprocess(document_id)
