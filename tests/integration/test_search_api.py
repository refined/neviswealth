from fastapi.testclient import TestClient

from app.main import create_app
from tests.integration.conftest import make_settings

BILL = (
    "Florida Power & Light monthly statement. Service address: 1450 Brickell Avenue, Miami. "
    "Total electricity usage was 812 kWh. Amount due: $143.27."
)
PASSPORT = "United States passport. Surname: Doe. Date of birth: 14 May 1978. Nationality: USA."


def _search(api: TestClient, q: str, **params: object) -> list[dict]:
    response = api.get("/search", params={"q": q, **params})
    assert response.status_code == 200, response.text
    return response.json()


def test_company_name_finds_client_by_email_domain(api: TestClient, make_client) -> None:
    john = make_client(first_name="John", last_name="Doe", email="john.doe@neviswealth.com")
    make_client(first_name="Maria", last_name="Garcia", email="maria@gmail.com")

    hits = _search(api, "NevisWealth")
    assert [h["client"]["id"] for h in hits] == [john["id"]]
    hit = hits[0]
    assert hit["type"] == "client"
    assert hit["score"] == 1.0
    assert hit["matched_on"] == ["email"]
    assert hit["highlight"] == "john.doe@<mark>neviswealth</mark>.com"


def test_client_search_by_name_partial_and_typo(api: TestClient, make_client) -> None:
    john = make_client(first_name="Johnathan", last_name="Doe", email="jd@example.com")
    exact = _search(api, "johnathan doe")[0]
    assert exact["client"]["id"] == john["id"] and exact["score"] == 1.0
    partial = _search(api, "Johnath")[0]
    assert (
        partial["matched_on"] == ["name"] and partial["highlight"] == "<mark>Johnath</mark>an Doe"
    )
    for typo in ("Jonathan", "john doe"):  # fuzzy matches rank below exact/substring ones
        hit = _search(api, typo)[0]
        assert hit["client"]["id"] == john["id"] and hit["score"] < 0.9
        assert hit["matched_on"] == ["name"]
    assert _search(api, "zebra") == []


def test_client_search_by_description_and_social_links(api: TestClient, make_client) -> None:
    aisha = make_client(
        description="Real estate investor with rental properties in London.",
        social_links=["https://twitter.com/aisha_invests"],
    )
    by_description = _search(api, "rental property investor")
    assert by_description[0]["client"]["id"] == aisha["id"]
    assert "description" in by_description[0]["matched_on"]
    by_handle = _search(api, "aisha_invests")
    assert by_handle[0]["matched_on"] == ["social_links"]


def test_like_wildcards_in_query_are_literal(api: TestClient, make_client) -> None:
    make_client(email="someone@example.com")
    assert _search(api, "%") == []
    assert _search(api, "_") == []


def test_pending_document_is_found_lexically(api: TestClient, make_client, make_document) -> None:
    doc = make_document(make_client()["id"], "Electricity bill", BILL)
    hits = _search(api, "electricity usage", type="document")
    assert [h["document"]["id"] for h in hits] == [doc["id"]]
    assert hits[0]["document"]["status"] == "pending"
    assert hits[0]["matched_on"][0] == "content"
    assert "semantic" not in hits[0]["matched_on"]  # no embeddings until indexing finishes
    assert "<mark>electricity usage</mark>" in hits[0]["highlight"].lower()


def test_ready_document_has_summary_and_semantic_signal(
    api: TestClient, make_client, make_document, worker
) -> None:
    client = make_client(first_name="Maria", last_name="Garcia")
    doc = make_document(client["id"], "Electricity bill", BILL)
    worker.process_pending()
    hit = _search(api, "electricity", type="document")[0]
    # Hits embed the same Document as GET /documents/{id}; the client is fetched separately.
    assert hit["document"] == api.get(f"/documents/{doc['id']}").json()
    assert "client" not in hit
    assert hit["document"]["client_id"] == client["id"]
    assert hit["document"]["summary"]
    assert hit["matched_on"][:2] == ["title", "semantic"]


def test_taxonomy_expansion_finds_utility_bill_for_address_proof(
    api: TestClient, make_client, make_document, worker
) -> None:
    client_id = make_client()["id"]
    bill = make_document(client_id, "Electricity bill - March", BILL)
    make_document(client_id, "Passport copy", PASSPORT)
    worker.process_pending()

    hits = _search(api, "address proof", type="document")
    assert hits[0]["document"]["id"] == bill["id"]
    assert "document_type:proof_of_address" in hits[0]["matched_on"]
    # The phrase "electricity bill" is only in the title, so nothing in the content is marked:
    # no stray single words such as "address" from the query or "bill" from a type phrase.
    assert hits[0]["highlight"] and "<mark>" not in hits[0]["highlight"]


def test_type_phrase_in_content_is_highlighted_as_one_phrase(
    api: TestClient, make_client, make_document, worker
) -> None:
    make_document(make_client()["id"], "March", "Attached is the utility bill for March.")
    worker.process_pending()

    hits = _search(api, "address proof", type="document")
    assert "<mark>utility bill</mark>" in hits[0]["highlight"]


def test_garbage_query_returns_nothing(api: TestClient, make_client, make_document, worker) -> None:
    make_document(make_client()["id"], "Electricity bill", BILL)
    worker.process_pending()
    assert _search(api, "qwertyzxcv") == []


def test_results_mix_types_sorted_by_score_and_respect_filters(
    api: TestClient, make_client, make_document
) -> None:
    client = make_client(first_name="Brickell", last_name="Owner", email="owner@brickell.io")
    make_document(client["id"], "Electricity bill", BILL)
    hits = _search(api, "brickell")
    assert [h["type"] for h in hits] == ["client", "document"]
    assert hits[0]["score"] >= hits[1]["score"]
    assert [h["type"] for h in _search(api, "brickell", type="document")] == ["document"]
    assert len(_search(api, "brickell", limit=1)) == 1


def test_highlight_is_html_escaped(api: TestClient, make_client, make_document) -> None:
    content = "Beware <script>alert(1)</script> invoices where fees < 5 & taxes."
    make_document(make_client()["id"], "Note", content)
    highlight = _search(api, "invoices", type="document")[0]["highlight"]
    assert "<script" not in highlight  # ts_headline drops tags; everything else is escaped
    assert "<mark>invoices</mark> where fees &lt; 5 &amp; taxes" in highlight


def test_query_validation(api: TestClient) -> None:
    assert api.get("/search").status_code == 422
    assert api.get("/search", params={"q": ""}).status_code == 422
    assert api.get("/search", params={"q": "   "}).status_code == 422
    assert api.get("/search", params={"q": "x" * 257}).status_code == 422
    assert api.get("/search", params={"q": "x", "limit": 0}).status_code == 422
    assert api.get("/search", params={"q": "x", "limit": 101}).status_code == 422
    assert api.get("/search", params={"q": "x", "type": "invoice"}).status_code == 422


def test_stopword_only_query_is_handled(api: TestClient, make_client, make_document) -> None:
    make_document(make_client()["id"], "Note", "The and of.")
    assert api.get("/search", params={"q": "the"}).status_code == 200


def test_api_key_protects_endpoints_when_configured(database_url: str) -> None:
    from tests.fakes import HashingEmbedder

    app = create_app(make_settings(database_url, api_key="s3cret"), embedder=HashingEmbedder())
    with TestClient(app) as api:
        assert api.get("/search", params={"q": "x"}).status_code == 401
        assert (
            api.get("/search", params={"q": "x"}, headers={"X-API-Key": "nope"}).status_code == 401
        )
        assert (
            api.get("/search", params={"q": "x"}, headers={"X-API-Key": "s3cret"}).status_code
            == 200
        )
        assert api.get("/health").status_code == 200  # health stays public


def test_search_service_works_without_http(app, make_client) -> None:
    # The business layer takes typed, already-validated input and opens its own connection.
    from app.services.search import ResultType, SearchRequest

    john = make_client(email="john.doe@neviswealth.com")
    request = SearchRequest(query="neviswealth", types=frozenset({ResultType.CLIENT}), limit=5)
    hits = app.state.container.search.search(request)
    assert [(h.type, str(h.client.id)) for h in hits] == [("client", john["id"])]


def test_search_type_must_be_a_known_value(api: TestClient) -> None:
    for bad in ("Client", "invoice", ""):
        assert api.get("/search", params={"q": "x", "type": bad}).status_code == 422
