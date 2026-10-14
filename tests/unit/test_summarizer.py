import numpy as np

from app.indexing.summarizer import ExtractiveSummarizer, select_sentences, truncate
from tests.fakes import HashingEmbedder

summarizer = ExtractiveSummarizer(HashingEmbedder(), max_sentences=2, max_chars=300)


def test_short_documents_are_returned_as_is() -> None:
    assert (
        summarizer.summarize("t", "One sentence. Two sentences.") == "One sentence. Two sentences."
    )


def test_empty_content_gives_empty_summary() -> None:
    assert summarizer.summarize("t", "   ") == ""


def test_summary_prefers_central_sentences_and_keeps_document_order() -> None:
    content = (
        "The portfolio holds equities and bonds. "
        "Lunch was served at noon. "
        "The portfolio equities returned eight percent. "
        "The bonds in the portfolio were stable."
    )
    summary = summarizer.summarize("Portfolio review", content)
    assert "Lunch" not in summary
    sentences = [s.strip() + "." for s in content.split(".") if s.strip()]
    picked = [s for s in sentences if s in summary]
    assert len(picked) == 2
    assert picked == sorted(picked, key=sentences.index)


def test_summary_avoids_redundant_sentences() -> None:
    repeated = "The client wants capital preservation."
    content = f"{repeated} {repeated} {repeated} The client also wants capital growth over time."
    summary = summarizer.summarize("Goals", content)
    assert summary.count(repeated) == 1


def test_summary_respects_max_chars() -> None:
    long_sentence = "This sentence is long " * 30 + "."
    content = " ".join([long_sentence] * 5)
    assert (
        len(ExtractiveSummarizer(HashingEmbedder(), max_chars=120).summarize("t", content)) <= 120
    )


def test_select_sentences_returns_unique_indices() -> None:
    vectors = np.eye(5, dtype=np.float32)
    picked = select_sentences(vectors, [10] * 5, k=3)
    assert len(picked) == len(set(picked)) == 3


def test_truncate_cuts_on_word_boundary() -> None:
    assert truncate("alpha beta gamma", 12) == "alpha beta…"
    assert truncate("short", 12) == "short"
