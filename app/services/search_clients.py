import difflib
import html

from psycopg import Connection

from app import schemas, table
from app.indexing.text import client_field_terms, normalize
from app.search.scoring import (
    FUZZY_MIN_QUERY_LENGTH,
    FUZZY_MIN_WORD_SIMILARITY,
    ClientSignals,
    score_client,
)
from app.table.records import ClientMatch

# Candidates fetched from the database before scoring.
_CANDIDATES = 100


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


class ClientSearch:
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
                    highlight=self._highlight(client, q, signals.substring_fields),
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
    def _highlight(client: ClientMatch, q: str, fields: list[str]) -> str | None:
        if "email" in fields:
            return _mark(client.email, q)
        if "name" in fields:
            return _mark(f"{client.first_name} {client.last_name}", q)
        if "social_links" in fields:
            link = next((s for s in client.social_links if q in s.casefold()), None)
            return _mark(link, q) if link else None
        return None
