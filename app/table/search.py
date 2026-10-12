"""Search queries: index lookups only (trigram, inverted and HNSW vector indexes)."""

import uuid

import numpy as np
from psycopg import Connection
from psycopg.rows import class_row

from app.table.connection import vector_literal
from app.table.records import ClientMatch, LexicalMatch, NearestDocument


def escape_like(value: str) -> str:
    """Make `%`, `_` and `\\` literal inside a LIKE pattern (used with ESCAPE '\\')."""
    return value.replace("\\", "\\\\").replace("%", r"\%").replace("_", r"\_")


# -- clients ---------------------------------------------------------------------------------


def client_matches(
    conn: Connection,
    query: str,
    *,
    fuzzy: bool,
    word_similarity_threshold: float,
    limit: int,
) -> list[ClientMatch]:
    """Clients whose identity terms contain `query` (or nearly do), or whose description matches.

    `query` must already be normalized (case-folded) like `clients.search_text`.
    """
    # Threshold of the `<%` operator, local to the current transaction.
    conn.execute(
        "SELECT set_config('pg_trgm.word_similarity_threshold', %s, true)",
        (str(word_similarity_threshold),),
    )
    with conn.cursor(row_factory=class_row(ClientMatch)) as cur:
        # LIKE and `<%` (word similarity) are both served by the GIN trigram index.
        cur.execute(
            r"""
            SELECT c.id, c.first_name, c.last_name, c.email, c.description, c.social_links,
                   c.created_at,
                   word_similarity(%(q)s, c.search_text) AS word_similarity,
                   CASE WHEN c.description_tsv @@ tq
                        THEN ts_rank_cd(c.description_tsv, tq, 32) END AS description_rank
            FROM clients c, websearch_to_tsquery('english', %(q)s) AS tq
            WHERE c.search_text LIKE %(like)s ESCAPE '\'
               OR (%(fuzzy)s AND %(q)s <%% c.search_text)
               OR c.description_tsv @@ tq
            LIMIT %(limit)s
            """,
            {"q": query, "like": f"%{escape_like(query)}%", "fuzzy": fuzzy, "limit": limit},
        )
        return cur.fetchall()


# -- documents -------------------------------------------------------------------------------


def lexical_matches(conn: Connection, query: str, *, limit: int) -> list[LexicalMatch]:
    """Full-text matches over title (weight A) and content (weight B), best first."""
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
    """Documents owning the `chunk_limit` chunks nearest to `vector`, with their best chunk."""
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
    """Documents containing any of the phrases (stemmed, words adjacent)."""
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


def query_headlines(
    conn: Connection, ids: list[uuid.UUID], query: str, *, options: str, strip_chars: str
) -> dict[uuid.UUID, str]:
    """`ts_headline` snippets of the documents' content for a free-text query."""
    rows = conn.execute(
        """
        SELECT d.id,
               ts_headline('english', translate(d.content, %(strip)s, ''), tq, %(options)s)
        FROM documents d, websearch_to_tsquery('english', %(q)s) AS tq
        WHERE d.id = ANY(%(ids)s)
        """,
        {"q": query, "ids": ids, "options": options, "strip": strip_chars},
    )
    return {row[0]: row[1] for row in rows}


def phrase_headlines(
    conn: Connection,
    ids: list[uuid.UUID],
    phrases: tuple[str, ...],
    *,
    options: str,
    strip_chars: str,
) -> dict[uuid.UUID, str]:
    """`ts_headline` snippets of the documents' content for any of the phrases."""
    rows = conn.execute(
        """
        SELECT d.id,
               ts_headline('english', translate(d.content, %(strip)s, ''), x.tq, %(options)s)
        FROM documents d,
             (SELECT tsquery_or_agg(phraseto_tsquery('english', p)) AS tq
              FROM unnest(%(phrases)s::text[]) AS p) AS x
        WHERE d.id = ANY(%(ids)s)
        """,
        {"phrases": list(phrases), "ids": ids, "options": options, "strip": strip_chars},
    )
    return {row[0]: row[1] for row in rows}
