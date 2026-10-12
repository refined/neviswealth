import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Response, status
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from app import schemas
from app.api import clients, documents, search
from app.api.deps import Container, get_container, require_api_key
from app.config import Settings, get_settings
from app.indexing.embedder import Embedder, create_embedder
from app.indexing.summarizer import ExtractiveSummarizer
from app.indexing.worker import IndexingWorker
from app.search.service import SearchService
from app.search.taxonomy import TaxonomyMatcher
from app.table.connection import create_pool
from app.table.records import EMBEDDING_DIM

STATIC_DIR = Path(__file__).parent / "static"

DESCRIPTION = """
Search across wealth-management clients and their documents.

* **Clients** are found by name, email (any part of it, e.g. the company domain), social links
  and description.
* **Documents** are found by their words *and* their meaning: `address proof` finds a utility bill.
  Each document gets an extractive summary once background indexing finishes.

Errors use FastAPI's standard shape: `{"detail": ...}`.
"""


class NoCacheStaticFiles(StaticFiles):
    """Static UI files that browsers revalidate (ETag) on every load, so deploys show up."""

    def file_response(self, *args: Any, **kwargs: Any) -> Response:
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


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
    return Container(settings, pool, embedder, search_service, worker)


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
        yield
        container.worker.stop()
        container.pool.close()

    app = FastAPI(
        title="Nevis Search API", version="1.0.0", description=DESCRIPTION, lifespan=lifespan
    )
    app.state.container = container

    protected = [Depends(require_api_key)]
    app.include_router(clients.router, dependencies=protected)
    app.include_router(documents.router, dependencies=protected)
    app.include_router(search.router, dependencies=protected)

    @app.get("/health", response_model=schemas.Health, tags=["health"])
    def health(container: Container = Depends(get_container)) -> schemas.Health:
        try:
            with container.pool.connection(timeout=5) as conn:
                conn.execute("SELECT 1")
        except Exception as exc:
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE, detail="Database unavailable"
            ) from exc
        return schemas.Health(
            status="ok",
            database="ok",
            embedding_model=container.embedder.model_name,
            worker_running=container.worker.is_running,
        )

    app.mount("/ui", NoCacheStaticFiles(directory=STATIC_DIR, html=True), name="ui")

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse("/ui/")

    return app
