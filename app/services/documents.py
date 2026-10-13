import uuid
from dataclasses import dataclass

from psycopg_pool import ConnectionPool

from app.errors import NotFoundError
from app.indexing.worker import IndexingWorker
from app.table import clients, documents
from app.table.records import ChunkRecord, DocumentRecord, LexemeRecord


@dataclass(frozen=True)
class DocumentIndexInfo:
    document: DocumentRecord
    chunks: list[ChunkRecord]
    lexemes: list[LexemeRecord]
    embedding_model: str


class DocumentService:
    def __init__(
        self, pool: ConnectionPool, worker: IndexingWorker, *, embedding_model: str
    ) -> None:
        self._pool = pool
        self._worker = worker
        self._embedding_model = embedding_model

    def create(self, client_id: uuid.UUID, *, title: str, content: str) -> DocumentRecord:
        # Postgres fills the full-text index in this transaction; chunks, embeddings and the
        # summary are produced afterwards by the indexing worker.
        with self._pool.connection() as conn:
            if clients.get_client(conn, client_id) is None:
                raise NotFoundError("Client")
            document = documents.insert_document(
                conn, client_id=client_id, title=title, content=content
            )
        # Committed when the `with` block exited, so the worker can see the new row.
        self._worker.notify()
        return document

    def get(self, document_id: uuid.UUID) -> DocumentRecord:
        with self._pool.connection() as conn:
            document = documents.get_document(conn, document_id)
        if document is None:
            raise NotFoundError("Document")
        return document

    def list_for_client(self, client_id: uuid.UUID) -> list[DocumentRecord]:
        with self._pool.connection() as conn:
            if clients.get_client(conn, client_id) is None:
                raise NotFoundError("Client")
            return documents.list_client_documents(conn, client_id)

    def index(self, document_id: uuid.UUID) -> DocumentIndexInfo:
        with self._pool.connection() as conn:
            document = documents.get_document(conn, document_id)
            if document is None:
                raise NotFoundError("Document")
            return DocumentIndexInfo(
                document=document,
                chunks=documents.list_chunks(conn, document_id),
                lexemes=documents.top_lexemes(conn, document_id),
                embedding_model=self._embedding_model,
            )

    def reprocess(self, document_id: uuid.UUID) -> DocumentRecord:
        with self._pool.connection() as conn:
            document = documents.requeue_document(conn, document_id)
        if document is None:
            raise NotFoundError("Document")
        self._worker.notify()
        return document
