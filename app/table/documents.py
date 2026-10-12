import uuid
from collections.abc import Sequence

from psycopg import Connection
from psycopg.rows import class_row

from app.table.connection import vector_literal
from app.table.records import (
    ChunkRecord,
    DocumentRecord,
    DocumentStatus,
    Job,
    LexemeRecord,
    NewChunk,
)

# -- CRUD ------------------------------------------------------------------------------------


def insert_document(
    conn: Connection, *, client_id: uuid.UUID, title: str, content: str
) -> DocumentRecord:
    """Insert a document; Postgres fills its full-text index, the worker does the rest."""
    with conn.cursor(row_factory=class_row(DocumentRecord)) as cur:
        cur.execute(
            """
            INSERT INTO documents (client_id, title, content, status)
            VALUES (%s, %s, %s, 'pending')
            RETURNING id, client_id, title, content, created_at, status, summary, attempts,
                      error, processed_at, processing_ms
            """,
            (client_id, title, content),
        )
        row = cur.fetchone()
    assert row is not None
    return row


def get_document(conn: Connection, document_id: uuid.UUID) -> DocumentRecord | None:
    with conn.cursor(row_factory=class_row(DocumentRecord)) as cur:
        cur.execute(
            """
            SELECT id, client_id, title, content, created_at, status, summary, attempts, error,
                   processed_at, processing_ms
            FROM documents
            WHERE id = %s
            """,
            (document_id,),
        )
        return cur.fetchone()


def list_client_documents(conn: Connection, client_id: uuid.UUID) -> list[DocumentRecord]:
    with conn.cursor(row_factory=class_row(DocumentRecord)) as cur:
        cur.execute(
            """
            SELECT id, client_id, title, content, created_at, status, summary, attempts, error,
                   processed_at, processing_ms
            FROM documents
            WHERE client_id = %s
            ORDER BY created_at DESC, id
            """,
            (client_id,),
        )
        return cur.fetchall()


def requeue_document(conn: Connection, document_id: uuid.UUID) -> DocumentRecord | None:
    """Put the document back into the indexing queue; None if it does not exist."""
    with conn.cursor(row_factory=class_row(DocumentRecord)) as cur:
        cur.execute(
            """
            UPDATE documents
            SET status = 'pending', attempts = 0, error = NULL, locked_at = NULL
            WHERE id = %s
            RETURNING id, client_id, title, content, created_at, status, summary, attempts,
                      error, processed_at, processing_ms
            """,
            (document_id,),
        )
        return cur.fetchone()


# -- index inspection ------------------------------------------------------------------------


def list_chunks(conn: Connection, document_id: uuid.UUID) -> list[ChunkRecord]:
    with conn.cursor(row_factory=class_row(ChunkRecord)) as cur:
        cur.execute(
            """
            SELECT ordinal, text, char_start, char_end
            FROM document_chunks
            WHERE document_id = %s
            ORDER BY ordinal
            """,
            (document_id,),
        )
        return cur.fetchall()


def top_lexemes(conn: Connection, document_id: uuid.UUID, limit: int = 40) -> list[LexemeRecord]:
    """The most frequent stemmed terms of the document's full-text index."""
    with conn.cursor(row_factory=class_row(LexemeRecord)) as cur:
        cur.execute(
            """
            SELECT l.lexeme, coalesce(array_length(l.positions, 1), 0) AS count, l.weights
            FROM documents d, unnest(d.tsv) AS l
            WHERE d.id = %s
            ORDER BY count DESC, l.lexeme
            LIMIT %s
            """,
            (document_id, limit),
        )
        return cur.fetchall()


# -- indexing queue --------------------------------------------------------------------------
# The documents table is the queue. A claim stamps `locked_at`; results are written only while
# that stamp is unchanged, so a concurrent requeue wins over a stale in-flight run.


def claim_next_document(conn: Connection, *, stale_after_s: int, max_attempts: int) -> Job | None:
    """Claim the oldest pending document (or one whose worker died) for indexing."""
    # Documents stuck in `processing` that already used all attempts are given up on.
    conn.execute(
        """
        UPDATE documents
        SET status = 'failed', locked_at = NULL,
            error = 'Indexing did not finish within the time limit'
        WHERE status = 'processing'
          AND locked_at < now() - make_interval(secs => %(stale_after_s)s)
          AND attempts >= %(max_attempts)s
        """,
        {"stale_after_s": stale_after_s, "max_attempts": max_attempts},
    )
    with conn.cursor(row_factory=class_row(Job)) as cur:
        cur.execute(
            """
            UPDATE documents
            SET status = 'processing', locked_at = clock_timestamp(), attempts = attempts + 1
            WHERE id = (
                SELECT id FROM documents
                WHERE status = 'pending'
                   OR (status = 'processing'
                       AND locked_at < now() - make_interval(secs => %(stale_after_s)s))
                ORDER BY created_at
                FOR UPDATE SKIP LOCKED
                LIMIT 1
            )
            RETURNING id, title, content, attempts, locked_at
            """,
            {"stale_after_s": stale_after_s},
        )
        return cur.fetchone()


def save_index_result(
    conn: Connection, job: Job, *, summary: str, chunks: Sequence[NewChunk], processing_ms: int
) -> bool:
    """Store chunks + summary and mark the document ready. False if the claim was lost."""
    updated = conn.execute(
        """
        UPDATE documents
        SET status = 'ready', summary = %(summary)s, error = NULL, locked_at = NULL,
            processed_at = now(), processing_ms = %(processing_ms)s
        WHERE id = %(id)s AND locked_at = %(locked_at)s
        """,
        {
            "summary": summary,
            "processing_ms": processing_ms,
            "id": job.id,
            "locked_at": job.locked_at,
        },
    ).rowcount
    if not updated:
        return False
    conn.execute("DELETE FROM document_chunks WHERE document_id = %s", (job.id,))
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO document_chunks (document_id, ordinal, text, char_start, char_end,
                                         embedding)
            VALUES (%s, %s, %s, %s, %s, %s::vector)
            """,
            [
                (job.id, c.ordinal, c.text, c.char_start, c.char_end, vector_literal(c.embedding))
                for c in chunks
            ],
        )
    return True


def record_failure(conn: Connection, job: Job, *, status: DocumentStatus, error: str) -> None:
    """Release the claim after a failed attempt: back to `pending` for a retry, or `failed`."""
    conn.execute(
        """
        UPDATE documents
        SET status = %(status)s, error = %(error)s, locked_at = NULL
        WHERE id = %(id)s AND locked_at = %(locked_at)s
        """,
        {"status": status.value, "error": error, "id": job.id, "locked_at": job.locked_at},
    )
