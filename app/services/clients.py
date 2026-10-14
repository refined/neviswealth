import uuid

from psycopg.errors import UniqueViolation
from psycopg_pool import ConnectionPool

from app import table
from app.errors import DuplicateEmailError, NotFoundError
from app.indexing.text import build_client_search_text
from app.table.records import ClientRecord


class ClientService:
    def __init__(self, pool: ConnectionPool) -> None:
        self._pool = pool

    def create(
        self,
        *,
        first_name: str,
        last_name: str,
        email: str,
        description: str | None,
        social_links: list[str],
    ) -> ClientRecord:
        try:
            with self._pool.connection() as conn:
                return table.clients.insert_client(
                    conn,
                    first_name=first_name,
                    last_name=last_name,
                    email=email,
                    description=description,
                    social_links=social_links,
                    # Pre-processing at write time: identity terms for substring / fuzzy search.
                    search_text=build_client_search_text(
                        first_name, last_name, email, social_links
                    ),
                )
        except UniqueViolation as exc:
            raise DuplicateEmailError from exc

    def get(self, client_id: uuid.UUID) -> ClientRecord:
        with self._pool.connection() as conn:
            client = table.clients.get_client(conn, client_id)
        if client is None:
            raise NotFoundError("Client")
        return client

    def list(self, *, limit: int, offset: int) -> list[ClientRecord]:
        with self._pool.connection() as conn:
            return table.clients.list_clients(conn, limit=limit, offset=offset)
