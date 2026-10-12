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
    # Threshold of the `<%` operator, local to the current transaction.
    conn.execute(
        "SELECT set_config('pg_trgm.word_similarity_threshold', %s, true)",
        (str(word_similarity_threshold),),
    )
    # Three ways to match, all index-backed:
    #  - LIKE '%q%': `q` is a substring of the identity terms (trigram index);
    #  - `q <% search_text`: word_similarity >= threshold, i.e. most of q's 3-letter chunks
    #    (trigrams) appear in some word run of search_text: tolerates typos (trigram index);
    #  - description full-text match, ranked by ts_rank_cd scaled to [0, 1) (GIN tsvector index).
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
