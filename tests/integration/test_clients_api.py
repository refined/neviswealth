import uuid

from fastapi.testclient import TestClient

JOHN = {
    "first_name": "John",
    "last_name": "Doe",
    "email": "john.doe@neviswealth.com",
    "description": "Senior partner interested in sustainable investing.",
    "social_links": ["https://www.linkedin.com/in/john-doe"],
}


def test_create_client_returns_201_with_location(api: TestClient) -> None:
    response = api.post("/clients", json=JOHN)
    assert response.status_code == 201
    body = response.json()
    assert response.headers["location"] == f"/clients/{body['id']}"
    assert {k: body[k] for k in JOHN} == JOHN
    assert uuid.UUID(body["id"]) and body["created_at"]


def test_optional_fields_default_to_empty(api: TestClient) -> None:
    body = api.post(
        "/clients", json={"first_name": "A", "last_name": "B", "email": "a@b.io"}
    ).json()
    assert body["description"] is None
    assert body["social_links"] == []


def test_duplicate_email_is_a_conflict_case_insensitively(api: TestClient) -> None:
    assert api.post("/clients", json=JOHN).status_code == 201
    response = api.post("/clients", json={**JOHN, "email": "John.Doe@NevisWealth.com"})
    assert response.status_code == 409
    assert response.json() == {"detail": "A client with this email already exists"}


def test_validation_errors_are_422(api: TestClient) -> None:
    assert api.post("/clients", json={**JOHN, "email": "not-an-email"}).status_code == 422
    assert api.post("/clients", json={**JOHN, "first_name": "   "}).status_code == 422
    assert api.post("/clients", json={"first_name": "John"}).status_code == 422
    assert api.post("/clients", json={**JOHN, "social_links": "not-a-list"}).status_code == 422
    assert (
        api.post(
            "/clients", content=b"{broken json", headers={"content-type": "application/json"}
        ).status_code
        == 422
    )


def test_get_client(api: TestClient) -> None:
    created = api.post("/clients", json=JOHN).json()
    assert api.get(f"/clients/{created['id']}").json() == created


def test_unknown_client_is_404_and_malformed_id_is_422(api: TestClient) -> None:
    assert api.get(f"/clients/{uuid.uuid4()}").status_code == 404
    assert api.get("/clients/not-a-uuid").status_code == 422


def test_list_clients_newest_first_with_pagination(api: TestClient, make_client) -> None:
    ids = [make_client()["id"] for _ in range(3)]
    listed = [c["id"] for c in api.get("/clients").json()]
    assert listed == ids[::-1]
    assert [c["id"] for c in api.get("/clients?limit=1&offset=1").json()] == [ids[1]]
    assert api.get("/clients?limit=0").status_code == 422
