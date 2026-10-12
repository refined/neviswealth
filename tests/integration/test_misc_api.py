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


def test_swagger_ui_is_served_under_api(api: TestClient) -> None:
    response = api.get("/api")
    assert response.status_code == 200
    assert "swagger-ui" in response.text
    assert api.get("/docs").status_code == 404


def test_openapi_matches_the_assignment_contract(api: TestClient) -> None:
    """Every field of the assignment's spec is still there (extra fields are allowed)."""
    spec = api.get("/api/openapi.json").json()
    schemas = spec["components"]["schemas"]

    def props(name: str) -> set[str]:
        return set(schemas[name]["properties"])

    assert set(schemas["ClientCreate"]["required"]) == {"first_name", "last_name", "email"}
    assert props("ClientCreate") == {
        "first_name",
        "last_name",
        "email",
        "description",
        "social_links",
    }
    assert {"id", "first_name", "last_name", "email", "description", "social_links"} <= props(
        "Client"
    )
    assert set(schemas["DocumentCreate"]["required"]) == {"title", "content"}
    assert {"id", "client_id", "title", "content", "created_at"} <= props("Document")

    paths = spec["paths"]
    assert "201" in paths["/clients"]["post"]["responses"]
    assert "201" in paths["/clients/{client_id}/documents"]["post"]["responses"]
    q = next(p for p in paths["/search"]["get"]["parameters"] if p["name"] == "q")
    assert q["required"] is True and q["in"] == "query"


def test_openapi_documents_the_search_hit_union(api: TestClient) -> None:
    spec = api.get("/api/openapi.json").json()
    assert spec["openapi"].startswith("3.1")
    assert {"/clients", "/clients/{client_id}/documents", "/search"} <= set(spec["paths"])
    search_items = spec["paths"]["/search"]["get"]["responses"]["200"]["content"][
        "application/json"
    ]["schema"]["items"]
    assert search_items["discriminator"]["propertyName"] == "type"
