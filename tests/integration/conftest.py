from collections.abc import Callable, Iterator
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import Settings
from app.indexing.worker import IndexingWorker
from app.main import create_app
from tests.fakes import HashingEmbedder


def make_settings(database_url: str, **overrides: Any) -> Settings:
    return Settings(database_url=database_url, run_worker=False, **overrides)


@pytest.fixture(scope="session")
def app(database_url: str) -> FastAPI:
    # Hashing embedder: deterministic and offline. It only models word overlap, which is enough
    # to test the plumbing; semantic quality is covered by tests/eval with the real model.
    return create_app(make_settings(database_url), embedder=HashingEmbedder())


@pytest.fixture(scope="session")
def session_client(app: FastAPI) -> Iterator[TestClient]:
    # One lifespan for the whole session: it opens the connection pool, which cannot be reopened.
    with TestClient(app) as client:
        yield client


@pytest.fixture
def api(app: FastAPI, session_client: TestClient) -> TestClient:
    """The API client, with an empty database for every test."""
    with app.state.container.pool.connection() as conn:
        conn.execute("TRUNCATE clients, documents, document_chunks CASCADE")
    return session_client


@pytest.fixture
def worker(app: FastAPI) -> IndexingWorker:
    return app.state.container.worker


@pytest.fixture
def make_client(api: TestClient) -> Callable[..., dict[str, Any]]:
    counter = iter(range(10_000))

    def _make(**fields: Any) -> dict[str, Any]:
        n = next(counter)
        body = {"first_name": "Test", "last_name": f"User{n}", "email": f"user{n}@example.com"}
        body.update(fields)
        response = api.post("/clients", json=body)
        assert response.status_code == 201, response.text
        return response.json()

    return _make


@pytest.fixture
def make_document(api: TestClient) -> Callable[..., dict[str, Any]]:
    def _make(client_id: str, title: str, content: str) -> dict[str, Any]:
        response = api.post(
            f"/clients/{client_id}/documents", json={"title": title, "content": content}
        )
        assert response.status_code == 201, response.text
        return response.json()

    return _make
