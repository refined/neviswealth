import difflib
import html

from psycopg import Connection

from app import schemas, table
from app.indexing.text import client_field_terms, normalize
from app.search.scoring import clip01, noisy_or
from app.table.records import ClientMatch

# Candidates fetched from the database before scoring.
_CANDIDATES = 100

# pg_trgm's default (0.6) misses single typos in mid-length names: "jonathan" vs "johnathan"
# scores 0.58. Fuzzy matching is only used for queries of at least FUZZY_MIN_QUERY_LENGTH chars.
FUZZY_MIN_WORD_SIMILARITY = 0.5
FUZZY_MIN_QUERY_LENGTH = 4


def _mark(value: str, needle: str) -> str | None:
    """HTML-escape `value` and wrap the first case-insensitive occurrence of `needle`."""
    start = value.casefold().find(needle)
    if start < 0:
        return None
    if len(value.casefold()) != len(value):  # case folding shifted the offsets
        return html.escape(value)
    end = start + len(needle)
    return (
        f"{html.escape(value[:start])}<mark>{html.escape(value[start:end])}</mark>"
        f"{html.escape(value[end:])}"
    )


class ClientSearch:
    """Clients by identity terms (as a substring or with a typo) and by description words."""

    def search(self, conn: Connection, query: str, limit: int) -> list[schemas.ClientHit]:
        q = normalize(query)
        fuzzy = len(q) >= FUZZY_MIN_QUERY_LENGTH
        matches = table.clients.client_matches(
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
            score = self._score(client, q, fields, fuzzy)
            if score is None:
                continue
            matched_on = self._matched_on(client, q, fields, fuzzy)
            hits.append(
                schemas.ClientHit(
                    score=score,
                    matched_on=matched_on,
                    highlight=self._highlight(client, q, matched_on),
                    client=schemas.Client.model_validate(client),
                )
            )
        hits.sort(key=lambda h: -h.score)
        return hits[:limit]

    @staticmethod
    def _score(
        client: ClientMatch, q: str, fields: dict[str, list[str]], fuzzy: bool
    ) -> float | None:
        """Relevance in [0, 1], or None if the client should not be shown."""
        confidences = []
        if any(q in terms for terms in fields.values()):  # the query is a whole term
            confidences.append(1.0)
        elif any(q in t for terms in fields.values() for t in terms):  # part of a term
            confidences.append(0.9)
        elif fuzzy and client.word_similarity >= FUZZY_MIN_WORD_SIMILARITY:  # with a typo
            confidences.append(min(client.word_similarity, 0.85))  # always below a substring
        if client.description_rank is not None:
            confidences.append(0.55 + 0.2 * clip01(client.description_rank / 0.3))
        if not confidences:
            return None
        return round(noisy_or(confidences), 4)

    def _matched_on(
        self, client: ClientMatch, q: str, fields: dict[str, list[str]], fuzzy: bool
    ) -> list[str]:
        matched = [f for f, terms in fields.items() if any(q in t for t in terms)]
        if not matched and fuzzy and client.word_similarity >= FUZZY_MIN_WORD_SIMILARITY:
            matched = [self._closest_field(q, fields)]
        if client.description_rank is not None:
            matched.append("description")
        return matched

    @staticmethod
    def _closest_field(q: str, fields: dict[str, list[str]]) -> str:
        def best_ratio(terms: list[str]) -> float:
            return max((difflib.SequenceMatcher(None, q, t).ratio() for t in terms), default=0.0)

        return max(fields, key=lambda f: best_ratio(fields[f]))

    @staticmethod
    def _highlight(client: ClientMatch, q: str, matched_on: list[str]) -> str | None:
        """The matched identity field with the query marked; None for typo or description hits."""
        if "email" in matched_on:
            return _mark(client.email, q)
        if "name" in matched_on:
            return _mark(f"{client.first_name} {client.last_name}", q)
        if "social_links" in matched_on:
            link = next((s for s in client.social_links if q in s.casefold()), None)
            return _mark(link, q) if link else None
        return None
