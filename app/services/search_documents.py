import html
import re
import uuid
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from psycopg import Connection

from app import schemas, table
from app.indexing.embedder import Embedder
from app.indexing.summarizer import truncate
from app.search.scoring import clip01, noisy_or
from app.search.taxonomy import TaxonomyMatcher, TypeMatch
from app.table.records import DocumentRecord

# Candidates fetched per retrieval channel before scoring.
_CANDIDATES = 100

# Cosine similarities of all-MiniLM-L6-v2 for relevant pairs mostly sit in 0.2..0.6 and unrelated
# queries stay below ~0.15 on the seed corpus (see scripts/spike_embeddings.py).
SEMANTIC_FLOOR = 0.15
SEMANTIC_CEILING = 0.65

# Query-to-type similarity -> confidence. A query that names a type ("address proof", 1.0) gives a
# strong signal; a vague one (~0.3) only nudges, so it cannot outrank a direct semantic match.
TAXONOMY_FLOOR = 0.25
TAXONOMY_CEILING = 0.75

# How strongly a matched type pulls the query vector towards its centroid (TaxonomyMatcher.blend).
# Measured on the eval set: 0.5 and 1.0 find the same documents, 1.0 with higher scores; at 2.0 a
# wrongly matched type starts to dominate ("retirement planning" -> estate planning -> the will).
TYPE_BLEND_WEIGHT = 1.0

# A document is shown only if it scores at least this share of the best document for the query.
# Weak "meaning only" neighbours (0.13-0.30 next to a 0.9 hit) were most of the noise: on the 42
# eval queries this cut drops 133 -> 30 unexpected documents and keeps every top-3 answer.
RELATIVE_CUTOFF = 0.5

# ts_headline returns raw document text. It marks matches with private markers; the snippet is
# HTML-escaped in Python and only then are the markers turned into <mark> tags.
_MARK_START, _MARK_END = "⟦", "⟧"
_HEADLINE_OPTIONS = (
    f'MaxFragments=2, MaxWords=18, MinWords=6, StartSel="{_MARK_START}", '
    f'StopSel="{_MARK_END}", FragmentDelimiter=" … "'
)


def _render_headline(snippet: str) -> str:
    # ts_headline marks each word separately; "⟦utility⟧ ⟦bill⟧" becomes one marked phrase.
    snippet = re.sub(rf"{_MARK_END}(\s+){_MARK_START}", r"\1", snippet)
    return html.escape(snippet).replace(_MARK_START, "<mark>").replace(_MARK_END, "</mark>")


@dataclass
class DocumentSignals:
    """What the three searches found about one document."""

    lexical_rank: float | None = None  # ts_rank_cd(..., 32), already in [0, 1)
    in_title: bool = False
    semantic: float | None = None  # best chunk cosine similarity
    best_chunk: str | None = None
    taxonomy: TypeMatch | None = None


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
        # We blend domain knowledge: "address proof" -> type "proof of address".
        # The type steers both the meaning search (blended vector) and its own phrase search.
        type_matches = self.taxonomy.match(query_vector, self.taxonomy_min_similarity)
        self._search_by_words(conn, query, signals)
        self._search_by_meaning(
            conn, self.taxonomy.blend(query_vector, type_matches, TYPE_BLEND_WEIGHT), signals
        )
        self._search_by_document_type(conn, type_matches, signals)

        scored = [
            (score, doc_id, sig)
            for doc_id, sig in signals.items()
            if (score := self._score(sig)) is not None
        ]
        scored.sort(key=lambda item: -item[0])
        if not scored:
            return []
        cutoff = RELATIVE_CUTOFF * scored[0][0]
        scored = [item for item in scored if item[0] >= cutoff][:limit]

        found = {d.id: d for d in table.documents.get_documents(conn, [i for _, i, _ in scored])}
        highlights = self._highlights(conn, query, scored, found)
        hits = []
        for score, doc_id, sig in scored:
            document = found.get(doc_id)
            if document is None:  # deleted between the queries
                continue
            hits.append(
                schemas.DocumentHit(
                    score=score,
                    matched_on=self._matched_on(sig),
                    highlight=highlights.get(doc_id),
                    document=schemas.Document.model_validate(document),
                )
            )
        return hits

    def _search_by_words(
        self, conn: Connection, query: str, signals: dict[uuid.UUID, DocumentSignals]
    ) -> None:
        # Full-text: the inverted index over title (weight A) and content (weight B).
        for match in table.documents.lexical_matches(conn, query, limit=_CANDIDATES):
            signals[match.id].lexical_rank = match.rank
            signals[match.id].in_title = match.in_title

    def _search_by_meaning(
        self,
        conn: Connection,
        query_vector: np.ndarray,
        signals: dict[uuid.UUID, DocumentSignals],
    ) -> None:
        # Embeddings: the chunks nearest to the (blended) query vector (HNSW index), best chunk
        # per document.
        for nearest in table.documents.nearest_documents(
            conn, query_vector, chunk_limit=self.semantic_candidate_chunks
        ):
            signals[nearest.document_id].semantic = nearest.similarity
            signals[nearest.document_id].best_chunk = nearest.best_chunk

    def _search_by_document_type(
        self,
        conn: Connection,
        type_matches: list[TypeMatch],
        signals: dict[uuid.UUID, DocumentSignals],
    ) -> None:
        # Documents that contain one of the matched type's phrases ("utility bill", "lease
        # agreement", ...) get the type signal.
        for type_match in type_matches:
            ids = table.documents.documents_with_phrases(
                conn, type_match.type.all_phrases, limit=_CANDIDATES
            )
            for doc_id in ids:
                current = signals[doc_id].taxonomy
                if current is None or current.similarity < type_match.similarity:
                    signals[doc_id].taxonomy = type_match

    def _score(self, sig: DocumentSignals) -> float | None:
        """Relevance in [0, 1], or None if no signal is strong enough to show the document."""
        if not (
            sig.lexical_rank is not None
            or sig.taxonomy is not None
            or (sig.semantic is not None and sig.semantic >= self.semantic_min_similarity)
        ):
            return None
        # Each signal -> a confidence in [0, 1]; the confidences are combined with noisy-OR.
        confidences = []
        if sig.lexical_rank is not None:  # words: 0.6, +0.15 in the title, +0.15 by rank
            confidences.append(
                0.6 + (0.15 if sig.in_title else 0.0) + 0.15 * clip01(sig.lexical_rank / 0.3)
            )
        if sig.semantic is not None:  # meaning: linear from SEMANTIC_FLOOR to SEMANTIC_CEILING
            confidences.append(
                0.9 * clip01((sig.semantic - SEMANTIC_FLOOR) / (SEMANTIC_CEILING - SEMANTIC_FLOOR))
            )
        if sig.taxonomy is not None:  # type phrase: by how clearly the query names the type
            confidences.append(
                0.7
                * clip01(
                    (sig.taxonomy.similarity - TAXONOMY_FLOOR) / (TAXONOMY_CEILING - TAXONOMY_FLOOR)
                )
            )
        return round(noisy_or(confidences), 4)

    def _matched_on(self, sig: DocumentSignals) -> list[str]:
        matched = []
        if sig.lexical_rank is not None:
            matched.append("title" if sig.in_title else "content")
        if sig.semantic is not None and sig.semantic >= self.semantic_min_similarity:
            matched.append("semantic")
        if sig.taxonomy is not None:
            matched.append(f"document_type:{sig.taxonomy.type.key}")
        return matched

    def _highlights(
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
        marked = table.documents.headlines(
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
