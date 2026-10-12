import uuid

from psycopg import Connection
from psycopg.rows import class_row

from app.table.records import ClientRecord


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
    """Insert a client. Raises `psycopg.errors.UniqueViolation` if the email is taken."""
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
