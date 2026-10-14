import uuid
from collections.abc import Sequence

import numpy as np
from psycopg import Connection
from psycopg.rows import class_row

from app.table.connection import vector_literal
from app.table.records import (
    ChunkRecord,
    DocumentRecord,
    DocumentStatus,
    Job,
    LexemeRecord,
    LexicalMatch,
    NearestDocument,
    NewChunk,
)


def insert_document(
    conn: Connection, *, client_id: uuid.UUID, title: str, content: str
) -> DocumentRecord:
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


def get_documents(conn: Connection, document_ids: list[uuid.UUID]) -> list[DocumentRecord]:
    with conn.cursor(row_factory=class_row(DocumentRecord)) as cur:
        cur.execute(
            """
            SELECT id, client_id, title, content, created_at, status, summary, attempts, error,
                   processed_at, processing_ms
            FROM documents
            WHERE id = ANY(%s)
            """,
            (document_ids,),
        )
        return cur.fetchall()


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
    # unnest(tsvector) yields one row per stemmed lexeme with its positions and weights.
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
    # Give up on documents whose worker died after their last allowed attempt.
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
    # Oldest pending document, or one stuck in `processing` (its worker died). SKIP LOCKED makes
    # concurrent workers skip rows another worker is claiming instead of waiting for them.
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
    # Only if the claim is still ours: a requeue in the meantime resets `locked_at`.
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
    conn.execute(
        """
        UPDATE documents
        SET status = %(status)s, error = %(error)s, locked_at = NULL
        WHERE id = %(id)s AND locked_at = %(locked_at)s
        """,
        {"status": status.value, "error": error, "id": job.id, "locked_at": job.locked_at},
    )


def lexical_matches(conn: Connection, query: str, *, limit: int) -> list[LexicalMatch]:
    # `tsv @@ tsquery`: does the document's word index (tsvector) satisfy the search query
    # websearch_to_tsquery: Google-like syntax ("quotes", OR, -minus); words are stemmed, so
    # "bills" matches "bill". ts_rank_cd scores how many query terms occur and how close together;
    # title terms carry weight A (1.0) vs B (0.4) for content. Normalization 32 maps the rank to
    # [0, 1) as rank / (rank + 1).
    with conn.cursor(row_factory=class_row(LexicalMatch)) as cur:
        cur.execute(
            """
            SELECT d.id,
                   ts_rank_cd(d.tsv, tq, 32) AS rank,
                   to_tsvector('english', d.title) @@ tq AS in_title
            FROM documents d, websearch_to_tsquery('english', %(q)s) AS tq
            WHERE d.tsv @@ tq
            ORDER BY rank DESC
            LIMIT %(limit)s
            """,
            {"q": query, "limit": limit},
        )
        return cur.fetchall()


def nearest_documents(
    conn: Connection, vector: np.ndarray, *, chunk_limit: int
) -> list[NearestDocument]:
    # `<=>` is cosine distance (1 - cosine similarity), served by the HNSW index. Take the
    # `chunk_limit` nearest chunks overall, then per document the best similarity and that chunk.
    # An HNSW scan returns at most ef_search rows; keep it >= the LIMIT (transaction-local).
    conn.execute("SELECT set_config('hnsw.ef_search', %s, true)", (str(max(40, chunk_limit)),))
    with conn.cursor(row_factory=class_row(NearestDocument)) as cur:
        cur.execute(
            """
            SELECT c.document_id,
                   max(1 - c.distance) AS similarity,
                   (array_agg(c.text ORDER BY c.distance))[1] AS best_chunk
            FROM (
                SELECT document_id, text, embedding <=> %(v)s::vector AS distance
                FROM document_chunks
                ORDER BY embedding <=> %(v)s::vector
                LIMIT %(limit)s
            ) c
            GROUP BY c.document_id
            """,
            {"v": vector_literal(vector), "limit": chunk_limit},
        )
        return cur.fetchall()


def documents_with_phrases(
    conn: Connection, phrases: tuple[str, ...], *, limit: int
) -> set[uuid.UUID]:
    # phraseto_tsquery('utility bill') = stemmed words in this order, next to each other;
    # tsquery_or_agg (defined in 0001_initial.sql) ORs all phrases into one query.
    rows = conn.execute(
        """
        SELECT d.id
        FROM documents d
        WHERE d.tsv @@ (SELECT tsquery_or_agg(phraseto_tsquery('english', p))
                        FROM unnest(%(phrases)s::text[]) AS p)
        LIMIT %(limit)s
        """,
        {"phrases": list(phrases), "limit": limit},
    )
    return {row[0] for row in rows}


def headlines(
    conn: Connection,
    ids: list[uuid.UUID],
    query: str,
    phrases: list[str],
    *,
    options: str,
    strip_chars: str,
) -> dict[uuid.UUID, str]:
    # One tsquery for everything that may have matched: the user's query OR any of the phrases
    # combined with tsquery_or_agg.
    # ts_headline picks the content fragments with the most matches and wraps each match in the
    # marker characters from `options`; `strip_chars` removes those markers from the content first
    # so only real matches carry them.
    rows = conn.execute(
        """
        SELECT d.id,
               ts_headline('english', translate(d.content, %(strip)s, ''), x.tq, %(options)s)
        FROM documents d,
             (SELECT tsquery_or_agg(t) AS tq
              FROM (SELECT websearch_to_tsquery('english', %(q)s) AS t
                    UNION ALL
                    SELECT phraseto_tsquery('english', p) FROM unnest(%(phrases)s::text[]) AS p
                   ) AS parts) AS x
        WHERE d.id = ANY(%(ids)s)
        """,
        {"q": query, "phrases": phrases, "ids": ids, "options": options, "strip": strip_chars},
    )
    return {row[0]: row[1] for row in rows}
