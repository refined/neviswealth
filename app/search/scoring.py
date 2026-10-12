"""Pure relevance scoring: turns raw retrieval signals into a score in [0, 1] (or "not relevant").

Each signal is mapped to a calibrated confidence and the confidences are combined with a noisy-OR,
`1 - prod(1 - c_i)`: one strong signal is enough, and agreeing signals reinforce each other while
the score stays in [0, 1], so clients and documents can share one ranked list.
"""

from dataclasses import dataclass, field

from app.search.taxonomy import TypeMatch


def noisy_or(confidences: list[float]) -> float:
    remaining = 1.0
    for c in confidences:
        remaining *= 1.0 - min(max(c, 0.0), 1.0)
    return 1.0 - remaining


def _clip01(x: float) -> float:
    return min(max(x, 0.0), 1.0)


# -- documents -------------------------------------------------------------------------------

# Cosine similarities of all-MiniLM-L6-v2 for relevant pairs mostly sit in 0.2..0.6 and unrelated
# queries stay below ~0.15 on the seed corpus (see scripts/spike_embeddings.py).
SEMANTIC_FLOOR = 0.15
SEMANTIC_CEILING = 0.65

# A document belongs to a matched type if its best chunk is close to the type's centroid: at
# least the floor, and within the margin of the closest document (centroid scales vary by type).
TYPE_CENTROID_MIN_SIMILARITY = 0.30
TYPE_CENTROID_MARGIN = 0.10

# Query-to-type similarity -> confidence. A query that names a type ("address proof", 1.0) gives a
# strong signal; a vague one (~0.3) only nudges, so it cannot outrank a direct semantic match.
TAXONOMY_FLOOR = 0.25
TAXONOMY_CEILING = 0.75


@dataclass
class DocumentSignals:
    lexical_rank: float | None = None  # ts_rank_cd(..., 32), already in [0, 1)
    in_title: bool = False
    semantic: float | None = None  # best chunk cosine similarity
    best_chunk: str | None = None
    taxonomy: TypeMatch | None = None


def document_confidences(signals: DocumentSignals) -> dict[str, float]:
    out: dict[str, float] = {}
    if signals.lexical_rank is not None:
        out["lexical"] = (
            0.6 + (0.15 if signals.in_title else 0.0) + 0.15 * _clip01(signals.lexical_rank / 0.3)
        )
    if signals.semantic is not None:
        out["semantic"] = 0.9 * _clip01(
            (signals.semantic - SEMANTIC_FLOOR) / (SEMANTIC_CEILING - SEMANTIC_FLOOR)
        )
    if signals.taxonomy is not None:
        out["taxonomy"] = 0.7 * _clip01(
            (signals.taxonomy.similarity - TAXONOMY_FLOOR) / (TAXONOMY_CEILING - TAXONOMY_FLOOR)
        )
    return out


def score_document(signals: DocumentSignals, semantic_min_similarity: float) -> float | None:
    """Return the relevance score, or None if no signal is strong enough to show the document."""
    relevant = (
        signals.lexical_rank is not None
        or signals.taxonomy is not None
        or (signals.semantic is not None and signals.semantic >= semantic_min_similarity)
    )
    if not relevant:
        return None
    return round(noisy_or(list(document_confidences(signals).values())), 4)


def document_matched_on(signals: DocumentSignals, semantic_min_similarity: float) -> list[str]:
    matched = []
    if signals.lexical_rank is not None:
        matched.append("title" if signals.in_title else "content")
    if signals.semantic is not None and signals.semantic >= semantic_min_similarity:
        matched.append("semantic")
    if signals.taxonomy is not None:
        matched.append(f"document_type:{signals.taxonomy.type.key}")
    return matched


# -- clients ---------------------------------------------------------------------------------

# pg_trgm's default (0.6) misses single typos in mid-length names: "jonathan" vs "johnathan"
# scores 0.58. Fuzzy matching is only used for queries of at least FUZZY_MIN_QUERY_LENGTH chars.
FUZZY_MIN_WORD_SIMILARITY = 0.5
FUZZY_MIN_QUERY_LENGTH = 4


@dataclass
class ClientSignals:
    exact_fields: list[str] = field(default_factory=list)  # query equals a term of these fields
    substring_fields: list[str] = field(default_factory=list)  # query is inside these fields
    word_similarity: float = 0.0  # pg_trgm word_similarity(query, identity terms)
    description_rank: float | None = None  # ts_rank_cd(..., 32) when the description matches


def score_client(signals: ClientSignals) -> float | None:
    confidences = []
    if signals.exact_fields:
        confidences.append(1.0)
    elif signals.substring_fields:
        confidences.append(0.9)
    elif signals.word_similarity >= FUZZY_MIN_WORD_SIMILARITY:  # typo-tolerant match
        confidences.append(min(signals.word_similarity, 0.85))  # always below a substring hit
    if signals.description_rank is not None:
        confidences.append(0.55 + 0.2 * _clip01(signals.description_rank / 0.3))
    if not confidences:
        return None
    return round(noisy_or(confidences), 4)
