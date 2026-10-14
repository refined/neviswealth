import uuid

import numpy as np
from fastapi.testclient import TestClient

from app.indexing.summarizer import ExtractiveSummarizer
from app.indexing.text import split_sentences
from app.indexing.worker import IndexingWorker
from tests.fakes import HashingEmbedder

BILL = (
    "Florida Power & Light monthly statement. Service address: 1450 Brickell Avenue, Miami. "
    "Total electricity usage was 812 kWh. Amount due: $143.27, payable by April 2, 2026. "
    "Thank you for enrolling in paperless billing."
)


def test_create_document_is_pending_with_location(api: TestClient, make_client) -> None:
    client = make_client()
    response = api.post(
        f"/clients/{client['id']}/documents", json={"title": "Electricity bill", "content": BILL}
    )
    assert response.status_code == 201
    doc = response.json()
    assert response.headers["location"] == f"/documents/{doc['id']}"
    assert doc["client_id"] == client["id"]
    assert doc["status"] == "pending"
    assert doc["summary"] is None
    assert api.get(f"/documents/{doc['id']}").json() == doc


def test_document_for_unknown_client_is_404(api: TestClient) -> None:
    response = api.post(f"/clients/{uuid.uuid4()}/documents", json={"title": "t", "content": "c"})
    assert response.status_code == 404
    assert api.get(f"/clients/{uuid.uuid4()}/documents").status_code == 404


def test_document_validation(api: TestClient, make_client) -> None:
    url = f"/clients/{make_client()['id']}/documents"
    assert api.post(url, json={"title": "", "content": "x"}).status_code == 422
    assert api.post(url, json={"title": "t", "content": "  "}).status_code == 422
    assert api.post(url, json={"title": "t"}).status_code == 422
    assert api.post(url, json={"title": "t", "content": "x" * 200_001}).status_code == 422


def test_unknown_document_is_404(api: TestClient) -> None:
    assert api.get(f"/documents/{uuid.uuid4()}").status_code == 404
    assert api.get(f"/documents/{uuid.uuid4()}/index").status_code == 404
    assert api.post(f"/documents/{uuid.uuid4()}/reprocess").status_code == 404


def test_worker_indexes_document(api: TestClient, make_client, make_document, worker) -> None:
    doc = make_document(make_client()["id"], "Electricity bill", BILL)
    assert worker.process_pending() == 1

    indexed = api.get(f"/documents/{doc['id']}").json()
    assert indexed["status"] == "ready"
    assert indexed["processed_at"] is not None
    summary_sentences = split_sentences(indexed["summary"])
    assert len(summary_sentences) == 3  # extractive: 3 of the 5 original sentences
    assert all(s.text in BILL for s in summary_sentences)

    index = api.get(f"/documents/{doc['id']}/index").json()
    assert index["embedding_model"] == "hashing"
    assert index["processing_ms"] is not None
    assert len(index["chunks"]) == 1
    chunk = index["chunks"][0]
    assert BILL[chunk["char_start"] : chunk["char_end"]] == chunk["text"]
    lexemes = {lx["lexeme"]: lx for lx in index["lexemes"]}
    assert "electr" in lexemes and "A" in lexemes["electr"]["weights"]  # title words weigh A


def test_long_document_is_chunked(api: TestClient, make_client, make_document, worker) -> None:
    content = " ".join(f"Sentence number {i} talks about portfolio item {i}." for i in range(100))
    doc = make_document(make_client()["id"], "Long notes", content)
    worker.process_pending()
    chunks = api.get(f"/documents/{doc['id']}/index").json()["chunks"]
    assert len(chunks) > 1
    assert all(len(c["text"].split()) <= 120 for c in chunks)


def test_list_client_documents(api: TestClient, make_client, make_document) -> None:
    client = make_client()
    first = make_document(client["id"], "First", "One.")
    second = make_document(client["id"], "Second", "Two.")
    make_document(make_client()["id"], "Other client", "Three.")
    listed = [d["id"] for d in api.get(f"/clients/{client['id']}/documents").json()]
    assert listed == [second["id"], first["id"]]


def test_reprocess_requeues_document(api: TestClient, make_client, make_document, worker) -> None:
    doc = make_document(make_client()["id"], "Electricity bill", BILL)
    worker.process_pending()
    response = api.post(f"/documents/{doc['id']}/reprocess")
    assert response.status_code == 202
    assert response.json()["status"] == "pending"
    assert worker.process_pending() == 1
    assert api.get(f"/documents/{doc['id']}").json()["status"] == "ready"
    assert len(api.get(f"/documents/{doc['id']}/index").json()["chunks"]) == 1  # replaced


class _BrokenEmbedder(HashingEmbedder):
    def embed_documents(self, texts):  # type: ignore[no-untyped-def]
        raise RuntimeError("model crashed")

    def embed_query(self, text: str) -> np.ndarray:
        raise RuntimeError("model crashed")


def test_failing_document_is_retried_then_marked_failed(
    api: TestClient, app, make_client, make_document
) -> None:
    container = app.state.container
    broken = _BrokenEmbedder()
    worker = IndexingWorker(
        container.pool,
        broken,
        ExtractiveSummarizer(broken),
        chunk_max_words=120,
        max_attempts=3,
    )
    doc = make_document(make_client()["id"], "Bill", BILL)

    assert worker.process_pending() == 3  # three attempts, then it gives up
    failed = api.get(f"/documents/{doc['id']}").json()
    assert failed["status"] == "failed"
    assert failed["error"] == "RuntimeError: model crashed"
    assert api.get(f"/documents/{doc['id']}/index").json()["attempts"] == 3

    # A healthy worker can pick it up again after an explicit reprocess.
    api.post(f"/documents/{doc['id']}/reprocess")
    assert container.worker.process_pending() == 1
    recovered = api.get(f"/documents/{doc['id']}").json()
    assert recovered["status"] == "ready" and recovered["error"] is None


def test_stale_result_is_discarded_when_document_was_requeued(
    api: TestClient, app, make_client, make_document
) -> None:
    container = app.state.container
    worker = container.worker
    doc = make_document(make_client()["id"], "Bill", BILL)

    job = worker._claim()
    assert job is not None
    api.post(f"/documents/{doc['id']}/reprocess")  # requeued while "in flight"
    worker._process(job)  # finishes, but no longer owns the document
    assert api.get(f"/documents/{doc['id']}").json()["status"] == "pending"
    assert worker.process_pending() == 1
    assert api.get(f"/documents/{doc['id']}").json()["status"] == "ready"


def test_deleted_document_is_archived_and_hidden(
    api: TestClient, make_client, make_document, worker
) -> None:
    client = make_client()
    doc = make_document(client["id"], "Electricity bill", BILL)
    kept = make_document(client["id"], "Another bill", BILL)
    worker.process_pending()

    assert api.delete(f"/documents/{doc['id']}").status_code == 204

    assert api.get(f"/documents/{doc['id']}").status_code == 404
    assert api.get(f"/documents/{doc['id']}/index").status_code == 404
    assert api.post(f"/documents/{doc['id']}/reprocess").status_code == 404
    listed = api.get(f"/clients/{client['id']}/documents").json()
    assert [d["id"] for d in listed] == [kept["id"]]
    found = api.get("/search", params={"q": "electricity usage", "type": "document"}).json()
    assert [h["document"]["id"] for h in found] == [kept["id"]]
    assert api.delete(f"/documents/{doc['id']}").status_code == 404  # already deleted


def test_delete_unknown_document_is_404(api: TestClient) -> None:
    assert api.delete(f"/documents/{uuid.uuid4()}").status_code == 404


def test_document_deleted_before_indexing_is_never_indexed(
    api: TestClient, app, make_client, make_document, worker
) -> None:
    doc = make_document(make_client()["id"], "Electricity bill", BILL)
    assert api.delete(f"/documents/{doc['id']}").status_code == 204

    assert worker.process_pending() == 0
    with app.state.container.pool.connection() as conn:
        chunks = conn.execute(
            "SELECT count(*) FROM document_chunks WHERE document_id = %s", (doc["id"],)
        ).fetchone()
    assert chunks == (0,)


def test_document_deleted_while_indexing_keeps_no_result(
    api: TestClient, app, make_client, make_document, worker
) -> None:
    doc = make_document(make_client()["id"], "Electricity bill", BILL)
    job = worker._claim()
    assert job is not None and str(job.id) == doc["id"]
    assert api.delete(f"/documents/{doc['id']}").status_code == 204

    worker._process(job)  # finishes indexing after the delete

    with app.state.container.pool.connection() as conn:
        row = conn.execute(
            "SELECT status, (SELECT count(*) FROM document_chunks WHERE document_id = d.id) "
            "FROM documents d WHERE id = %s",
            (doc["id"],),
        ).fetchone()
    assert row == ("processing", 0)
