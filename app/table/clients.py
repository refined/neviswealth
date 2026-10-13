import uuid

from psycopg import Connection
from psycopg.rows import class_row

from app.table.records import ClientMatch, ClientRecord


def insert_client(
    conn: Connection,
    *,
    first_name: str,
    last_name: str,
    email: str,
    description: str | None,
    social_links: list[str],
    search_text: str,
) -> ClientRecord:
    with conn.cursor(row_factory=class_row(ClientRecord)) as cur:
        cur.execute(
            """
            INSERT INTO clients (first_name, last_name, email, description, social_links,
                                 search_text)
            VALUES (%(first_name)s, %(last_name)s, %(email)s, %(description)s, %(social_links)s,
                    %(search_text)s)
            RETURNING id, first_name, last_name, email, description, social_links, created_at
            """,
            {
                "first_name": first_name,
                "last_name": last_name,
                "email": email,
                "description": description,
                "social_links": social_links,
                "search_text": search_text,
            },
        )
        row = cur.fetchone()
    assert row is not None
    return row


def get_client(conn: Connection, client_id: uuid.UUID) -> ClientRecord | None:
    with conn.cursor(row_factory=class_row(ClientRecord)) as cur:
        cur.execute(
            """
            SELECT id, first_name, last_name, email, description, social_links, created_at
            FROM clients
            WHERE id = %s
            """,
            (client_id,),
        )
        return cur.fetchone()


def list_clients(conn: Connection, *, limit: int, offset: int) -> list[ClientRecord]:
    with conn.cursor(row_factory=class_row(ClientRecord)) as cur:
        cur.execute(
            """
            SELECT id, first_name, last_name, email, description, social_links, created_at
            FROM clients
            ORDER BY created_at DESC, id
            LIMIT %s OFFSET %s
            """,
            (limit, offset),
        )
        return cur.fetchall()


def escape_like(value: str) -> str:
    """Make `%`, `_` and `\\` literal inside a LIKE pattern (used with ESCAPE '\\')."""
    return value.replace("\\", "\\\\").replace("%", r"\%").replace("_", r"\_")


def client_matches(
    conn: Connection,
    query: str,
    *,
    fuzzy: bool,
    word_similarity_threshold: float,
    limit: int,
) -> list[ClientMatch]:
    """Clients matching `query` in one of three ways; each way is served by an index.
    Two Postgres features do the work.
    1. The `pg_trgm` extension (enabled in db/migrations/0001_initial.sql;
       https://www.postgresql.org/docs/17/pgtrgm.html) compares strings by their trigrams, the
       overlapping 3-letter pieces of each word: `john` -> `  j`, ` jo`, `joh`, `ohn`, `hn `. It
       adds:
       - `word_similarity(q, text)`: the share of q's trigrams found in the most similar run of
         `text`, from 0 to 1.
       - `q <% text`: the same check as a condition, true when word_similarity >= the
         `pg_trgm.word_similarity_threshold` setting. An operator can
         use the GIN trigram index on `search_text`; so can `LIKE '%...%'`.
    2. Built-in full-text search (https://www.postgresql.org/docs/17/textsearch-controls.html)
       matches words, not substrings, for the free-text description:
       - `clients.description_tsv` holds the description's word stems, kept up to date by Postgres.
       - `websearch_to_tsquery('english', q)` turns user input into such stems and understands
         search-engine syntax
       - `tsv @@ tsquery` tests a match (GIN index); `ts_rank_cd(..., 32)` scores it in [0, 1).

    The three ways (ORed; Postgres combines the three index scans with a BitmapOr):
    - `search_text LIKE '%q%'`: `q` is part of a name, email or handle (`neviswealth`);
    - `q <% search_text`: the same with typos (`jonh doe`); off for short queries (`fuzzy`);
    - `description_tsv @@ tq`: the description contains the query's words (`retired physician`).
    `query` must already be normalized (case-folded) like `search_text`. The raw signals
    (`word_similarity`, `description_rank`) are returned for scoring in app/search/scoring.py.
    """
    conn.execute(
        "SELECT set_config('pg_trgm.word_similarity_threshold', %s, true)",
        (str(word_similarity_threshold),),
    )
    with conn.cursor(row_factory=class_row(ClientMatch)) as cur:
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
