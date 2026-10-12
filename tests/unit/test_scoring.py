import pytest

from app.search.scoring import (
    ClientSignals,
    DocumentSignals,
    document_matched_on,
    noisy_or,
    score_client,
    score_document,
)
from app.search.taxonomy import DOCUMENT_TYPES, TypeMatch

MIN_SEMANTIC = 0.22
ADDRESS = TypeMatch(DOCUMENT_TYPES[0], 0.9)


def test_noisy_or_is_bounded_and_monotonic() -> None:
    assert noisy_or([]) == 0
    assert noisy_or([0.5, 0.5]) == pytest.approx(0.75)
    assert noisy_or([1.5]) == 1.0
    assert noisy_or([0.6, 0.3]) > noisy_or([0.6])


def test_document_without_strong_signal_is_dropped() -> None:
    assert score_document(DocumentSignals(), MIN_SEMANTIC) is None
    assert score_document(DocumentSignals(semantic=0.18), MIN_SEMANTIC) is None


def test_each_signal_alone_can_make_a_document_relevant() -> None:
    assert score_document(DocumentSignals(lexical_rank=0.1), MIN_SEMANTIC) >= 0.6
    assert score_document(DocumentSignals(semantic=0.5), MIN_SEMANTIC) > 0.5
    assert score_document(DocumentSignals(taxonomy=ADDRESS), MIN_SEMANTIC) > 0.6


def test_title_match_and_agreeing_signals_rank_higher() -> None:
    content = score_document(DocumentSignals(lexical_rank=0.1), MIN_SEMANTIC)
    title = score_document(DocumentSignals(lexical_rank=0.1, in_title=True), MIN_SEMANTIC)
    both = score_document(DocumentSignals(lexical_rank=0.1, semantic=0.5), MIN_SEMANTIC)
    assert content < title
    assert content < both <= 1


def test_document_matched_on_explains_signals() -> None:
    signals = DocumentSignals(lexical_rank=0.1, in_title=True, semantic=0.4, taxonomy=ADDRESS)
    assert document_matched_on(signals, MIN_SEMANTIC) == [
        "title",
        "semantic",
        "document_type:proof_of_address",
    ]
    assert document_matched_on(DocumentSignals(semantic=0.1), MIN_SEMANTIC) == []


def test_client_scores() -> None:
    assert score_client(ClientSignals(exact_fields=["email"])) == 1.0
    assert score_client(ClientSignals(substring_fields=["email"])) == 0.9
    fuzzy = score_client(ClientSignals(word_similarity=0.75))
    assert fuzzy is not None and 0.5 < fuzzy < 0.9
    assert score_client(ClientSignals(word_similarity=0.4)) is None
    assert score_client(ClientSignals(description_rank=0.1)) is not None
    assert score_client(ClientSignals()) is None
