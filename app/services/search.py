import enum
from dataclasses import dataclass

from psycopg_pool import ConnectionPool

from app import schemas
from app.services.search_clients import ClientSearch
from app.services.search_documents import DocumentSearch


class ResultType(enum.StrEnum):
    CLIENT = "client"
    DOCUMENT = "document"


@dataclass(frozen=True)
class SearchRequest:
    query: str  # already stripped and non-empty
    types: frozenset[ResultType]
    limit: int


class SearchService:
    """One search box over clients and documents: one transaction, one list sorted by score."""

    def __init__(
        self, pool: ConnectionPool, clients: ClientSearch, documents: DocumentSearch
    ) -> None:
        self._pool = pool
        self._clients = clients
        self._documents = documents

    def search(self, request: SearchRequest) -> list[schemas.ClientHit | schemas.DocumentHit]:
        hits: list[schemas.ClientHit | schemas.DocumentHit] = []
        with self._pool.connection() as conn:
            if ResultType.CLIENT in request.types:
                hits.extend(self._clients.search(conn, request.query, request.limit))
            if ResultType.DOCUMENT in request.types:
                hits.extend(self._documents.search(conn, request.query, request.limit))
        hits.sort(key=lambda h: (-h.score, h.type))
        return hits[: request.limit]
