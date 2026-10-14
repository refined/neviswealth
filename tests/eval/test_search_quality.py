"""Search quality on the seed corpus with the real embedding model (`uv run pytest -m model`).

The corpus is db/seed/seed_corpus.json; the queries and expected documents are in queries.json.

The first run downloads the model (~90 MB) into ./models_cache.
"""

import json
from collections.abc import Iterator
from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.services.search_documents import RELATIVE_CUTOFF
from tests.conftest import ROOT

pytestmark = pytest.mark.model

CORPUS = json.loads((ROOT / "db" / "seed" / "seed_corpus.json").read_text(encoding="utf-8"))
QUERIES = json.loads((ROOT / "tests" / "eval" / "queries.json").read_text(encoding="utf-8"))


@dataclass
class Seeded:
    api: TestClient
    clients: dict[str, str]  # corpus key -> id
    documents: dict[str, str]

    def search(self, q: str, **params: object) -> list[dict]:
        response = self.api.get("/search", params={"q": q, **params})
        assert response.status_code == 200, response.text
        return response.json()

    def document_keys(self, hits: list[dict]) -> list[str]:
        by_id = {v: k for k, v in self.documents.items()}
        return [by_id[h["document"]["id"]] for h in hits if h["type"] == "document"]


@pytest.fixture(scope="module")
def seeded(database_url: str) -> Iterator[Seeded]:
    settings = Settings(
        database_url=database_url, run_worker=False, model_cache_dir=str(ROOT / "models_cache")
    )
    app = create_app(settings)
    with TestClient(app) as api:
        with app.state.container.pool.connection() as conn:
            conn.execute("TRUNCATE clients, documents, document_chunks CASCADE")
        clients, documents = {}, {}
        for c in CORPUS["clients"]:
            body = {k: v for k, v in c.items() if k != "key"}
            clients[c["key"]] = api.post("/clients", json=body).json()["id"]
        for d in CORPUS["documents"]:
            response = api.post(
                f"/clients/{clients[d['client']]}/documents",
                json={"title": d["title"], "content": d["content"]},
            )
            documents[d["key"]] = response.json()["id"]
        app.state.container.worker.process_pending()
        yield Seeded(api, clients, documents)


def test_all_documents_indexed(seeded: Seeded) -> None:
    for doc_id in seeded.documents.values():
        doc = seeded.api.get(f"/documents/{doc_id}").json()
        assert doc["status"] == "ready" and doc["summary"]


def test_company_name_finds_both_neviswealth_clients(seeded: Seeded) -> None:
    hits = seeded.search("NevisWealth")
    client_ids = [h["client"]["id"] for h in hits if h["type"] == "client"]
    assert set(client_ids[:2]) == {seeded.clients["john_doe"], seeded.clients["emma_johnson"]}


def test_address_proof_finds_utility_bill(seeded: Seeded) -> None:
    top3 = seeded.document_keys(seeded.search("address proof", type="document"))[:3]
    assert "utility_bill" in top3
    assert "lease_agreement" in top3


def _eval_params() -> list:
    return [
        pytest.param(
            item,
            id=item["q"],
            marks=[pytest.mark.xfail(reason=item["known_miss"], strict=True)]
            if "known_miss" in item
            else [],
        )
        for item in QUERIES["eval_queries"]
    ]


@pytest.mark.parametrize("q", ["Jonh Doe", "maria garcia", "chencapital"])
def test_person_and_company_queries_rank_the_client_first(seeded: Seeded, q: str) -> None:
    hits = seeded.search(q)
    assert hits and hits[0]["type"] == "client"
    assert all(h["score"] <= hits[0]["score"] for h in hits)


@pytest.mark.parametrize("item", _eval_params())
def test_expected_document_ranks_high(seeded: Seeded, item: dict) -> None:
    keys = seeded.document_keys(seeded.search(item["q"], type="document"))
    assert keys, "no results"
    assert keys[0] in item["expected"] or set(keys[:3]) & set(item["expected"]), keys[:5]


@pytest.mark.parametrize("q", QUERIES["negative_queries"])
def test_unrelated_query_returns_nothing(seeded: Seeded, q: str) -> None:
    assert seeded.search(q) == []


@pytest.mark.parametrize("q", ["pension income", "conservative investor", "stock holdings"])
def test_weak_documents_far_below_the_best_are_cut(seeded: Seeded, q: str) -> None:
    hits = seeded.search(q, type="document")
    assert hits
    assert all(h["score"] >= RELATIVE_CUTOFF * hits[0]["score"] for h in hits)
    assert len(hits) <= 4  # these returned 7-9 documents before the cut
