import secrets
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader
from psycopg_pool import ConnectionPool

from app.config import Settings
from app.indexing.worker import IndexingWorker
from app.services.clients import ClientService
from app.services.documents import DocumentService
from app.services.health import HealthService
from app.services.search import SearchService


@dataclass
class Container:
    """Composition root: built once at startup (app.main), read only by the getters below."""

    settings: Settings
    pool: ConnectionPool
    worker: IndexingWorker
    clients: ClientService
    documents: DocumentService
    search: SearchService
    health: HealthService


def _container(request: Request) -> Container:
    return request.app.state.container


def get_client_service(request: Request) -> ClientService:
    return _container(request).clients


def get_document_service(request: Request) -> DocumentService:
    return _container(request).documents


def get_search_service(request: Request) -> SearchService:
    return _container(request).search


def get_health_service(request: Request) -> HealthService:
    return _container(request).health


ClientServiceDep = Annotated[ClientService, Depends(get_client_service)]
DocumentServiceDep = Annotated[DocumentService, Depends(get_document_service)]
SearchServiceDep = Annotated[SearchService, Depends(get_search_service)]
HealthServiceDep = Annotated[HealthService, Depends(get_health_service)]


_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(request: Request, api_key: str | None = Security(_api_key_header)) -> None:
    expected = _container(request).settings.api_key
    if expected and not (api_key and secrets.compare_digest(api_key, expected)):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API key",
            headers={"WWW-Authenticate": "APIKey"},
        )
