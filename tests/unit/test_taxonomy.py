from app.search.taxonomy import DOCUMENT_TYPES, TaxonomyMatcher
from tests.fakes import HashingEmbedder

embedder = HashingEmbedder()
matcher = TaxonomyMatcher(embedder)


def test_query_naming_a_phrase_matches_its_type() -> None:
    matches = matcher.match(embedder.embed_query("utility bill"), min_similarity=0.3)
    assert matches[0].type.key == "proof_of_address"
    assert matches[0].similarity > 0.99


def test_unrelated_query_matches_nothing() -> None:
    assert matcher.match(embedder.embed_query("qwertyzxcv"), min_similarity=0.3) == []


def test_at_most_top_k_types_best_first() -> None:
    matches = matcher.match(
        embedder.embed_query("statement"), min_similarity=0.0, top_k=2, margin=1.0
    )
    assert len(matches) == 2
    assert matches[0].similarity >= matches[1].similarity


def test_runner_up_type_far_behind_the_best_is_dropped() -> None:
    query = embedder.embed_query("address proof")
    assert len(matcher.match(query, min_similarity=0.0, top_k=2, margin=1.0)) == 2
    assert [m.type.key for m in matcher.match(query, min_similarity=0.0, top_k=2)] == [
        "proof_of_address"
    ]


def test_type_keys_are_unique() -> None:
    keys = [t.key for t in DOCUMENT_TYPES]
    assert len(keys) == len(set(keys))
