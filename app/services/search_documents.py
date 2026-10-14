import html
import uuid
from collections import defaultdict

import numpy as np
from psycopg import Connection

from app import schemas
from app.indexing.embedder import Embedder
from app.indexing.summarizer import truncate
from app.search.scoring import (
    TYPE_CENTROID_MARGIN,
    TYPE_CENTROID_MIN_SIMILARITY,
    DocumentSignals,
    document_matched_on,
    score_document,
)
from app.search.taxonomy import TaxonomyMatcher
from app.table import documents
from app.table.records import DocumentRecord

# Candidates fetched per retrieval channel before scoring.
_CANDIDATES = 100

# ts_headline returns raw document text. It marks matches with private markers; the snippet is
# HTML-escaped in Python and only then are the markers turned into <mark> tags.
_MARK_START, _MARK_END = "⟦", "⟧"
_HEADLINE_OPTIONS = (
    f'MaxFragments=2, MaxWords=18, MinWords=6, StartSel="{_MARK_START}", '
    f'StopSel="{_MARK_END}", FragmentDelimiter=" … "'
)


def _render_headline(snippet: str) -> str:
    return html.escape(snippet).replace(_MARK_START, "<mark>").replace(_MARK_END, "</mark>")


class DocumentSearch:
    """Three searches (words, meaning, document type) feeding one score per document."""

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

    def search(self, conn: Connection, query: str, limit: int) -> list[schemas.DocumentHit]:
        signals: dict[uuid.UUID, DocumentSignals] = defaultdict(DocumentSignals)
        query_vector = self.embedder.embed_query(query)
        self._search_by_words(conn, query, signals)
        self._search_by_meaning(conn, query_vector, signals)
        self._search_by_document_type(conn, query_vector, signals)
        scored = self._score(signals, limit)
        if not scored:
            return []

        found = {d.id: d for d in documents.get_documents(conn, [i for _, i, _ in scored])}
        highlights = self._document_highlights(conn, query, scored, found)
        hits = []
        for score, doc_id, sig in scored:
            document = found.get(doc_id)
            if document is None:  # deleted between the queries
                continue
            hits.append(
                schemas.DocumentHit(
                    score=score,
                    matched_on=document_matched_on(sig, self.semantic_min_similarity),
                    highlight=highlights.get(doc_id),
                    document=schemas.Document.model_validate(document),
                )
            )
        return hits

    def _search_by_words(
        self, conn: Connection, query: str, signals: dict[uuid.UUID, DocumentSignals]
    ) -> None:
        # Full-text: the inverted index over title (weight A) and content (weight B).
        for match in documents.lexical_matches(conn, query, limit=_CANDIDATES):
            signals[match.id].lexical_rank = match.rank
            signals[match.id].in_title = match.in_title

    def _search_by_meaning(
        self,
        conn: Connection,
        query_vector: np.ndarray,
        signals: dict[uuid.UUID, DocumentSignals],
    ) -> None:
        # Embeddings: the chunks nearest to the query (HNSW index), best chunk per document.
        for nearest in documents.nearest_documents(
            conn, query_vector, chunk_limit=self.semantic_candidate_chunks
        ):
            signals[nearest.document_id].semantic = nearest.similarity
            signals[nearest.document_id].best_chunk = nearest.best_chunk

    def _search_by_document_type(
        self,
        conn: Connection,
        query_vector: np.ndarray,
        signals: dict[uuid.UUID, DocumentSignals],
    ) -> None:
        # Domain knowledge: "address proof" -> type "proof of address" -> its documents, found by
        # the type's phrases ("utility bill", "lease agreement", ...) and by its centroid.
        for type_match in self.taxonomy.match(query_vector, self.taxonomy_min_similarity):
            doc_type = type_match.type
            ids = documents.documents_with_phrases(conn, doc_type.all_phrases, limit=_CANDIDATES)
            # Absolute centroid similarities differ per type, so keep the documents close to the
            # best one (and above a floor) rather than applying one global threshold.
            nearest = documents.nearest_documents(
                conn, self.taxonomy.centroid(doc_type), chunk_limit=self.semantic_candidate_chunks
            )
            top = max((n.similarity for n in nearest), default=0.0)
            cutoff = max(TYPE_CENTROID_MIN_SIMILARITY, top - TYPE_CENTROID_MARGIN)
            ids |= {n.document_id for n in nearest if n.similarity >= cutoff}
            for doc_id in ids:
                current = signals[doc_id].taxonomy
                if current is None or current.similarity < type_match.similarity:
                    signals[doc_id].taxonomy = type_match

    def _score(
        self, signals: dict[uuid.UUID, DocumentSignals], limit: int
    ) -> list[tuple[float, uuid.UUID, DocumentSignals]]:
        # Relevant documents, best first; the formulas and thresholds live in app/search/scoring.py.
        scored = [
            (score, doc_id, sig)
            for doc_id, sig in signals.items()
            if (score := score_document(sig, self.semantic_min_similarity)) is not None
        ]
        scored.sort(key=lambda item: -item[0])
        return scored[:limit]

    def _document_highlights(
        self,
        conn: Connection,
        query: str,
        scored: list[tuple[float, uuid.UUID, DocumentSignals]],
        found: dict[uuid.UUID, DocumentRecord],
    ) -> dict[uuid.UUID, str]:
        """Snippet per document: its matched words, else the best-matching chunk, else its start."""
        word_ids = [d for _, d, s in scored if s.lexical_rank is not None or s.taxonomy is not None]
        phrases = sorted(
            {p for _, _, s in scored if s.taxonomy for p in s.taxonomy.type.all_phrases}
        )
        marked = documents.headlines(
            conn,
            word_ids,
            query,
            phrases,
            options=_HEADLINE_OPTIONS,
            strip_chars=_MARK_START + _MARK_END,
        )
        highlights: dict[uuid.UUID, str] = {}
        for _, doc_id, s in scored:
            headline = marked.get(doc_id, "")
            if _MARK_START in headline:
                highlights[doc_id] = _render_headline(headline)
            elif s.best_chunk:
                highlights[doc_id] = html.escape(truncate(s.best_chunk, 240))
            elif doc_id in found:
                highlights[doc_id] = html.escape(truncate(found[doc_id].content, 240))
        return highlights
