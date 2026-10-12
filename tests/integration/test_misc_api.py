from fastapi.testclient import TestClient


def test_health(api: TestClient) -> None:
    assert api.get("/health").json() == {
        "status": "ok",
        "database": "ok",
        "embedding_model": "hashing",
        "worker_running": False,  # tests drive the worker explicitly
    }


def test_ui_is_served_and_revalidated(api: TestClient) -> None:
    response = api.get("/", follow_redirects=True)
    assert response.status_code == 200
    assert "<title>Nevis Search</title>" in response.text
    assert response.headers["cache-control"] == "no-cache"
    assert api.get("/ui/app.js").status_code == 200


def test_openapi_documents_the_search_hit_union(api: TestClient) -> None:
    spec = api.get("/openapi.json").json()
    assert spec["openapi"].startswith("3.1")
    assert {"/clients", "/clients/{client_id}/documents", "/search"} <= set(spec["paths"])
    search_items = spec["paths"]["/search"]["get"]["responses"]["200"]["content"][
        "application/json"
    ]["schema"]["items"]
    assert search_items["discriminator"]["propertyName"] == "type"
