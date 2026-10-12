from app.indexing.text import (
    build_client_search_text,
    chunk_sentences,
    email_terms,
    normalize,
    social_link_terms,
    split_sentences,
)


def test_normalize_casefolds_and_collapses_whitespace() -> None:
    assert normalize("  Nevis\tWEALTH \n Inc ") == "nevis wealth inc"


def test_email_terms_expose_company_domain_label() -> None:
    terms = email_terms("John.Doe@NevisWealth.com")
    assert "neviswealth" in terms  # the company name without the TLD
    assert {"john.doe@neviswealth.com", "john.doe", "john", "doe", "neviswealth.com"} <= set(terms)
    assert "com" not in terms


def test_email_terms_split_plus_and_dashes_and_multi_level_domains() -> None:
    terms = email_terms("anna-maria+kyc@mail.example.co.uk")
    assert {"anna", "maria", "kyc", "mail", "example"} <= set(terms)
    assert "uk" not in terms


def test_social_link_terms_extract_site_and_handle() -> None:
    terms = social_link_terms("https://www.linkedin.com/in/john-doe-consulting/")
    assert terms[:2] == ["linkedin", "john-doe-consulting"]
    assert {"john", "doe", "consulting"} <= set(terms)
    assert "in" not in terms and "www" not in terms


def test_social_link_terms_handle_bare_handles_and_schemeless_urls() -> None:
    assert social_link_terms("@weichen_builds")[:1] == ["weichen_builds"]
    assert "weichen" in social_link_terms("@weichen_builds")
    assert "github" in social_link_terms("github.com/weichen")


def test_client_search_text_is_normalized_and_deduplicated() -> None:
    text = build_client_search_text("John", "DOE", "john.doe@neviswealth.com", [])
    assert "neviswealth" in text
    assert text == text.casefold()
    assert text.split().count("neviswealth.com") == 1


def test_split_sentences_keeps_offsets_into_original_text() -> None:
    content = "First sentence here.  Second one!\nThird? Yes."
    sentences = split_sentences(content)
    assert [s.text for s in sentences] == ["First sentence here.", "Second one!", "Third?", "Yes."]
    for s in sentences:
        assert content[s.start : s.end] == s.text


def test_split_sentences_does_not_break_on_abbreviations() -> None:
    content = "U.S. Individual Income Tax Return was filed. Acme AI Inc. Holdings paid in full."
    assert [s.text for s in split_sentences(content)] == [
        "U.S. Individual Income Tax Return was filed.",
        "Acme AI Inc. Holdings paid in full.",
    ]


def test_split_sentences_handles_empty_and_blank_text() -> None:
    assert split_sentences("") == []
    assert split_sentences("   \n  ") == []


def test_chunking_packs_sentences_with_overlap() -> None:
    sentence = " ".join(["word"] * 49) + "."
    content = " ".join(f"S{i} {sentence}" for i in range(4))  # 4 sentences x 50 words
    chunks = chunk_sentences(content, split_sentences(content), max_words=120)
    assert [c.text.split()[0] for c in chunks] == ["S0", "S1", "S2"]  # each overlaps by one
    assert all(c.word_count <= 120 for c in chunks)
    for c in chunks:
        assert content[c.start : c.end] == c.text


def test_chunking_splits_a_single_huge_sentence() -> None:
    content = " ".join(f"w{i}" for i in range(300))
    chunks = chunk_sentences(content, split_sentences(content), max_words=120)
    assert [c.word_count for c in chunks] == [120, 120, 60]
    assert " ".join(c.text for c in chunks) == content


def test_chunking_empty_input() -> None:
    assert chunk_sentences("", [], max_words=120) == []
