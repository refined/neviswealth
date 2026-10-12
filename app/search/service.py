"""Query-time search: combines index lookups (app.table.search) into one scored result list."""

import difflib
import html
import uuid
from collections import defaultdict

from psycopg import Connection

from app import schemas
from app.indexing.embedder import Embedder
from app.indexing.summarizer import truncate
from app.indexing.text import client_field_terms, normalize
from app.search.scoring import (
    FUZZY_MIN_QUERY_LENGTH,
    FUZZY_MIN_WORD_SIMILARITY,
    TYPE_CENTROID_MARGIN,
    TYPE_CENTROID_MIN_SIMILARITY,
    ClientSignals,
    DocumentSignals,
    document_matched_on,
    score_client,
    score_document,
)
from app.search.taxonomy import DocumentType, TaxonomyMatcher
from app.table import search as queries
from app.table.records import ClientMatch

# Candidates fetched per retrieval channel before fusion.
_CANDIDATES = 100

# ts_headline returns raw document text. It marks matches with private markers; the snippet is
# HTML-escaped in Python and only then are the markers turned into <mark> tags.
_MARK_START, _MARK_END = "⟦", "⟧"
_HEADLINE_OPTIONS = (
    f'MaxFragments=2, MaxWords=18, MinWords=6, StartSel="{_MARK_START}", '
    f'StopSel="{_MARK_END}", FragmentDelimiter=" … "'
)


def _mark(value: str, needle: str) -> str:
    """HTML-escape `value` and wrap the first case-insensitive occurrence of `needle`."""
    start = value.casefold().find(needle)
    if start < 0 or len(value.casefold()) != len(value):
        return html.escape(value)
    end = start + len(needle)
    return (
        f"{html.escape(value[:start])}<mark>{html.escape(value[start:end])}</mark>"
        f"{html.escape(value[end:])}"
    )


def _render_headline(snippet: str) -> str:
    return html.escape(snippet).replace(_MARK_START, "<mark>").replace(_MARK_END, "</mark>")


class SearchService:
    def __init__(
        self,
        embedder: Embedder,
        taxonomy: TaxonomyMatcher,
        *,
        semantic_min_similarity: float,
        taxonomy_min_similarity: float,
        semantic_candidate_chunks: int = 50,
    ) -> None:
        self.embedder = embedder
        self.taxonomy = taxonomy
        self.semantic_min_similarity = semantic_min_similarity
        self.taxonomy_min_similarity = taxonomy_min_similarity
        self.semantic_candidate_chunks = semantic_candidate_chunks

    def search(
        self, conn: Connection, query: str, kinds: set[str], limit: int
    ) -> list[schemas.ClientHit | schemas.DocumentHit]:
        hits: list[schemas.ClientHit | schemas.DocumentHit] = []
        if "client" in kinds:
            hits.extend(self.search_clients(conn, query, limit))
        if "document" in kinds:
            hits.extend(self.search_documents(conn, query, limit))
        hits.sort(key=lambda h: (-h.score, h.type))
        return hits[:limit]

    # -- clients ---------------------------------------------------------------------------

    def search_clients(self, conn: Connection, query: str, limit: int) -> list[schemas.ClientHit]:
        q = normalize(query)
        fuzzy = len(q) >= FUZZY_MIN_QUERY_LENGTH
        matches = queries.client_matches(
            conn,
            q,
            fuzzy=fuzzy,
            word_similarity_threshold=FUZZY_MIN_WORD_SIMILARITY,
            limit=_CANDIDATES,
        )

        hits = []
        for client in matches:
            fields = client_field_terms(
                client.first_name, client.last_name, client.email, client.social_links
            )
            signals = ClientSignals(
                exact_fields=[f for f, terms in fields.items() if q in terms],
                substring_fields=[f for f, terms in fields.items() if any(q in t for t in terms)],
                word_similarity=client.word_similarity if fuzzy else 0.0,
                description_rank=client.description_rank,
            )
            score = score_client(signals)
            if score is None:
                continue
            matched_on = signals.substring_fields or (
                [self._closest_field(q, fields)]
                if signals.word_similarity >= FUZZY_MIN_WORD_SIMILARITY
                else []
            )
            if signals.description_rank is not None:
                matched_on = [*matched_on, "description"]
            hits.append(
                schemas.ClientHit(
                    score=score,
                    matched_on=matched_on,
                    highlight=self._client_highlight(client, q, signals.substring_fields),
                    client=schemas.Client.model_validate(client),
                )
            )
        hits.sort(key=lambda h: -h.score)
        return hits[:limit]

    @staticmethod
    def _closest_field(q: str, fields: dict[str, list[str]]) -> str:
        def best_ratio(terms: list[str]) -> float:
            return max((difflib.SequenceMatcher(None, q, t).ratio() for t in terms), default=0.0)

        return max(fields, key=lambda f: best_ratio(fields[f]))

    @staticmethod
    def _client_highlight(client: ClientMatch, q: str, fields: list[str]) -> str | None:
        if "email" in fields:
            return _mark(client.email, q)
        if "name" in fields:
            return _mark(f"{client.first_name} {client.last_name}", q)
        if "social_links" in fields:
            link = next((s for s in client.social_links if q in s.casefold()), None)
            return _mark(link, q) if link else None
        return None

    # -- documents -------------------------------------------------------------------------

    def search_documents(
        self, conn: Connection, query: str, limit: int
    ) -> list[schemas.DocumentHit]:
        signals: dict[uuid.UUID, DocumentSignals] = defaultdict(DocumentSignals)

        # 1. Lexical: inverted index over title (weight A) and content (weight B).
        for match in queries.lexical_matches(conn, query, limit=_CANDIDATES):
            signals[match.id].lexical_rank = match.rank
            signals[match.id].in_title = match.in_title

        # 2. Semantic: nearest chunk embeddings (HNSW).
        query_vector = self.embedder.embed_query(query)
        for nearest in queries.nearest_documents(
            conn, query_vector, chunk_limit=self.semantic_candidate_chunks
        ):
            signals[nearest.document_id].semantic = nearest.similarity
            signals[nearest.document_id].best_chunk = nearest.best_chunk

        # 3. Taxonomy expansion: "address proof" -> documents of type "proof of address", found by
        #    the type's phrases ("utility bill", "lease agreement", ...) and by its centroid.
        for type_match in self.taxonomy.match(query_vector, self.taxonomy_min_similarity):
            for doc_id in self._documents_of_type(conn, type_match.type):
                current = signals[doc_id].taxonomy
                if current is None or current.similarity < type_match.similarity:
                    signals[doc_id].taxonomy = type_match

        scored = []
        for doc_id, sig in signals.items():
            score = score_document(sig, self.semantic_min_similarity)
            if score is not None:
                scored.append((score, doc_id, sig))
        scored.sort(key=lambda item: -item[0])
        scored = scored[:limit]
        if not scored:
            return []

        briefs = {b.id: b for b in queries.document_briefs(conn, [i for _, i, _ in scored])}
        highlights = self._document_highlights(conn, query, scored)

        hits = []
        for score, doc_id, sig in scored:
            brief = briefs.get(doc_id)
            if brief is None:  # deleted between the queries
                continue
            hits.append(
                schemas.DocumentHit(
                    score=score,
                    matched_on=document_matched_on(sig, self.semantic_min_similarity),
                    highlight=highlights.get(doc_id),
                    document=schemas.DocumentBrief.model_validate(brief),
                )
            )
        return hits

    def _documents_of_type(self, conn: Connection, doc_type: DocumentType) -> set[uuid.UUID]:
        by_phrase = queries.documents_with_phrases(conn, doc_type.all_phrases, limit=_CANDIDATES)
        # Absolute centroid similarities differ per type, so keep the documents close to the best
        # one (and above a floor) rather than applying one global threshold.
        nearest = queries.nearest_documents(
            conn, self.taxonomy.centroid(doc_type), chunk_limit=self.semantic_candidate_chunks
        )
        top = max((n.similarity for n in nearest), default=0.0)
        cutoff = max(TYPE_CENTROID_MIN_SIMILARITY, top - TYPE_CENTROID_MARGIN)
        by_centroid = {n.document_id for n in nearest if n.similarity >= cutoff}
        return by_phrase | by_centroid

    def _document_highlights(
        self,
        conn: Connection,
        query: str,
        scored: list[tuple[float, uuid.UUID, DocumentSignals]],
    ) -> dict[uuid.UUID, str]:
        """Snippet per document from the strongest evidence: query terms, type terms or chunk."""
        lexical_ids = [doc_id for _, doc_id, s in scored if s.lexical_rank is not None]
        by_type: dict[DocumentType, list[uuid.UUID]] = defaultdict(list)
        for _, doc_id, s in scored:
            if s.lexical_rank is None and s.taxonomy is not None:
                by_type[s.taxonomy.type].append(doc_id)

        strip = _MARK_START + _MARK_END
        headlines: dict[uuid.UUID, str] = {}
        if lexical_ids:
            headlines.update(
                queries.query_headlines(
                    conn, lexical_ids, query, options=_HEADLINE_OPTIONS, strip_chars=strip
                )
            )
        for doc_type, ids in by_type.items():
            headlines.update(
                queries.phrase_headlines(
                    conn, ids, doc_type.all_phrases, options=_HEADLINE_OPTIONS, strip_chars=strip
                )
            )

        # Prefer a headline that marks matches in the content; when the match was only in the
        # title (or by meaning), the best-matching chunk says more than the content's first words.
        highlights: dict[uuid.UUID, str] = {}
        for _, doc_id, s in scored:
            headline = headlines.get(doc_id)
            if headline and _MARK_START in headline:
                highlights[doc_id] = _render_headline(headline)
            elif s.best_chunk:
                highlights[doc_id] = html.escape(truncate(s.best_chunk, 240))
            elif headline:
                highlights[doc_id] = _render_headline(headline)
        return highlights
