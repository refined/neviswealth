import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI

from app.api import clients, documents, health, search, ui
from app.api.deps import Container, require_api_key
from app.api.errors import register_error_handlers
from app.config import Settings, get_settings
from app.indexing.embedder import Embedder, create_embedder
from app.indexing.summarizer import ExtractiveSummarizer
from app.indexing.worker import IndexingWorker
from app.search.taxonomy import TaxonomyMatcher
from app.services.clients import ClientService
from app.services.documents import DocumentService
from app.services.health import HealthService
from app.services.search import SearchService
from app.table.connection import create_pool
from app.table.records import EMBEDDING_DIM

DOCS_URL = "/api"  # Swagger UI
OPENAPI_URL = "/api/openapi.json"

log = logging.getLogger(__name__)

DESCRIPTION = """
Search across wealth-management clients and their documents.

* **Clients** are found by name, email (any part of it, e.g. the company domain), social links
  and description.
* **Documents** are found by their words *and* their meaning: `address proof` finds a utility bill.
  Each document gets an extractive summary once background indexing finishes.

Errors use FastAPI's standard shape: `{"detail": ...}`.
"""


def build_container(settings: Settings, embedder: Embedder | None = None) -> Container:
    if settings.embedding_dim != EMBEDDING_DIM:
        raise ValueError(f"EMBEDDING_DIM must be {EMBEDDING_DIM} (fixed by the database schema)")
    pool = create_pool(settings.database_url)
    embedder = embedder or create_embedder(
        settings.embedding_model,
        settings.embedding_dim,
        cache_dir=settings.model_cache_dir,
        threads=settings.embedding_threads,
    )
    summarizer = ExtractiveSummarizer(
        embedder,
        max_sentences=settings.summary_max_sentences,
        max_chars=settings.summary_max_chars,
    )
    search_service = SearchService(
        pool,
        embedder,
        TaxonomyMatcher(embedder),
        semantic_min_similarity=settings.semantic_min_similarity,
        taxonomy_min_similarity=settings.taxonomy_min_similarity,
        semantic_candidate_chunks=settings.semantic_candidate_chunks,
    )
    worker = IndexingWorker(
        pool,
        embedder,
        summarizer,
        chunk_max_words=settings.chunk_max_words,
        max_attempts=settings.worker_max_attempts,
        poll_interval_s=settings.worker_poll_interval_s,
        stale_after_s=settings.worker_stale_after_s,
    )
    return Container(
        settings=settings,
        pool=pool,
        worker=worker,
        clients=ClientService(pool),
        documents=DocumentService(pool, worker, embedding_model=embedder.model_name),
        search=search_service,
        health=HealthService(pool, embedder, worker),
    )


def create_app(settings: Settings | None = None, embedder: Embedder | None = None) -> FastAPI:
    settings = settings or get_settings()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    container = build_container(settings, embedder)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        container.pool.open(wait=True, timeout=30)
        if settings.run_worker:
            container.worker.start()
        log.info(
            "Nevis Search is up: UI %(url)s/ui  API docs %(url)s/api", {"url": settings.public_url}
        )
        yield
        container.worker.stop()
        container.pool.close()

    app = FastAPI(
        title="Nevis Search API",
        version="1.0.0",
        description=DESCRIPTION,
        lifespan=lifespan,
        docs_url=DOCS_URL,
        redoc_url=None,
        openapi_url=OPENAPI_URL,
    )
    app.state.container = container
    register_error_handlers(app)

    protected = [Depends(require_api_key)]
    app.include_router(clients.router, dependencies=protected)
    app.include_router(documents.router, dependencies=protected)
    app.include_router(search.router, dependencies=protected)

    app.include_router(health.router)
    app.include_router(ui.router)
    ui.mount_ui(app)

    return app
