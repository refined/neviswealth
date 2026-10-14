import uuid
from datetime import UTC, datetime

from app.indexing.text import client_field_terms
from app.search.taxonomy import DOCUMENT_TYPES, TaxonomyMatcher, TypeMatch
from app.services.search_clients import ClientSearch
from app.services.search_documents import DocumentSearch, DocumentSignals
from app.table.records import ClientMatch
from tests.fakes import HashingEmbedder

ADDRESS = TypeMatch(DOCUMENT_TYPES[0], 0.9)

_embedder = HashingEmbedder()
documents = DocumentSearch(
    _embedder,
    TaxonomyMatcher(_embedder),
    semantic_min_similarity=0.22,
    taxonomy_min_similarity=0.40,
)
clients = ClientSearch()


# -- documents -------------------------------------------------------------------------------


def test_document_without_strong_signal_is_dropped() -> None:
    assert documents._score(DocumentSignals()) is None
    assert documents._score(DocumentSignals(semantic=0.18)) is None


def test_each_signal_alone_can_make_a_document_relevant() -> None:
    assert documents._score(DocumentSignals(lexical_rank=0.1)) >= 0.6
    assert documents._score(DocumentSignals(semantic=0.5)) > 0.5
    assert documents._score(DocumentSignals(taxonomy=ADDRESS)) > 0.6


def test_title_match_and_agreeing_signals_rank_higher() -> None:
    content = documents._score(DocumentSignals(lexical_rank=0.1))
    title = documents._score(DocumentSignals(lexical_rank=0.1, in_title=True))
    both = documents._score(DocumentSignals(lexical_rank=0.1, semantic=0.5))
    assert content < title
    assert content < both <= 1


def test_document_matched_on_explains_signals() -> None:
    signals = DocumentSignals(lexical_rank=0.1, in_title=True, semantic=0.4, taxonomy=ADDRESS)
    assert documents._matched_on(signals) == [
        "title",
        "semantic",
        "document_type:proof_of_address",
    ]
    assert documents._matched_on(DocumentSignals(semantic=0.1)) == []


# -- clients ---------------------------------------------------------------------------------


def _client(word_similarity: float = 0.0, description_rank: float | None = None) -> ClientMatch:
    return ClientMatch(
        id=uuid.uuid4(),
        first_name="John",
        last_name="Doe",
        email="john.doe@neviswealth.com",
        description="Senior partner at a consulting firm.",
        social_links=[],
        created_at=datetime.now(UTC),
        word_similarity=word_similarity,
        description_rank=description_rank,
    )


def _score_and_reason(client: ClientMatch, q: str) -> tuple[float | None, list[str]]:
    fields = client_field_terms(
        client.first_name, client.last_name, client.email, client.social_links
    )
    fuzzy = len(q) >= 4
    return clients._score(client, q, fields, fuzzy), clients._matched_on(client, q, fields, fuzzy)


def test_whole_term_scores_highest() -> None:
    assert _score_and_reason(_client(), "neviswealth") == (1.0, ["email"])


def test_part_of_a_term_scores_slightly_lower() -> None:
    assert _score_and_reason(_client(), "nevis") == (0.9, ["email"])


def test_typo_match_scores_below_a_substring_and_names_the_closest_field() -> None:
    score, matched_on = _score_and_reason(_client(word_similarity=0.75), "jonh doe")
    assert score is not None and 0.5 < score < 0.9
    assert matched_on == ["name"]


def test_weak_typo_match_or_short_query_is_dropped() -> None:
    assert _score_and_reason(_client(word_similarity=0.4), "jonh doe") == (None, [])
    assert _score_and_reason(_client(word_similarity=0.9), "jon")[0] is None


def test_description_match_alone_is_enough() -> None:
    score, matched_on = _score_and_reason(_client(description_rank=0.1), "consulting")
    assert score is not None
    assert matched_on == ["description"]


def test_typo_hit_has_no_highlight_but_substring_hit_does() -> None:
    assert clients._highlight(_client(), "jonh doe", ["name"]) is None
    assert clients._highlight(_client(), "nevis", ["email"]) == (
        "john.doe@<mark>nevis</mark>wealth.com"
    )
