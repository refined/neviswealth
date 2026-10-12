import secrets
from collections.abc import Iterator
from dataclasses import dataclass

from fastapi import Depends, HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader
from psycopg import Connection
from psycopg_pool import ConnectionPool

from app.config import Settings
from app.indexing.embedder import Embedder
from app.indexing.worker import IndexingWorker
from app.search.service import SearchService


@dataclass
class Container:
    """Long-lived objects created once at startup and shared by all requests."""

    settings: Settings
    pool: ConnectionPool
    embedder: Embedder
    search: SearchService
    worker: IndexingWorker


def get_container(request: Request) -> Container:
    return request.app.state.container


def get_conn(container: Container = Depends(get_container)) -> Iterator[Connection]:
    """A pooled connection with an open transaction: committed on success, else rolled back.

    Endpoints that write commit explicitly before responding, so the data is visible to the
    client's next request (and to the indexing worker) as soon as the response arrives.
    """
    with container.pool.connection() as conn:
        yield conn


_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(
    api_key: str | None = Security(_api_key_header),
    container: Container = Depends(get_container),
) -> None:
    expected = container.settings.api_key
    if expected and not (api_key and secrets.compare_digest(api_key, expected)):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API key",
            headers={"WWW-Authenticate": "APIKey"},
        )
