import re
import unicodedata
from dataclasses import dataclass
from urllib.parse import urlsplit

_WS_RE = re.compile(r"\s+")
_TOKEN_SPLIT_RE = re.compile(r"[._+\-]+")
_PATH_SPLIT_RE = re.compile(r"[\-_.]+")

# Path segments that carry no identity information (e.g. linkedin.com/in/<handle>).
_GENERIC_PATH_SEGMENTS = {
    "in",
    "u",
    "user",
    "users",
    "profile",
    "profiles",
    "company",
    "pub",
    "people",
}


def normalize(text: str) -> str:
    return _WS_RE.sub(" ", unicodedata.normalize("NFKC", text)).strip().casefold()


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def name_terms(first_name: str, last_name: str) -> list[str]:
    first, last = normalize(first_name), normalize(last_name)
    return _dedupe([f"{first} {last}", first, last])


def email_terms(email: str) -> list[str]:
    """`john.doe@neviswealth.com` -> full address, local part and its tokens, domain, domain labels.

    The domain label without the TLD (`neviswealth`) is what makes a company-name query match.
    """
    email = normalize(email)
    local, _, domain = email.partition("@")
    terms = [email, local, *_TOKEN_SPLIT_RE.split(local)]
    if domain:
        labels = domain.split(".")
        terms.append(domain)
        terms.extend(labels[:-1] if len(labels) > 1 else labels)
    return _dedupe(terms)


def social_link_terms(link: str) -> list[str]:
    """`https://www.linkedin.com/in/john-doe` -> `linkedin`, `john-doe`, `john`, `doe`."""
    link = normalize(link).lstrip("@")
    if "://" not in link and "/" not in link and "." not in link:
        return _dedupe([link, *_PATH_SPLIT_RE.split(link)])  # a bare handle
    parts = urlsplit(link if "://" in link else f"https://{link}")
    host = (parts.hostname or "").removeprefix("www.")
    labels = host.split(".")
    terms = labels[:-1] if len(labels) > 1 else labels
    for segment in parts.path.split("/"):
        segment = segment.lstrip("@")
        if len(segment) < 2 or segment in _GENERIC_PATH_SEGMENTS:
            continue
        terms.append(segment)
        terms.extend(p for p in _PATH_SPLIT_RE.split(segment) if len(p) >= 2)
    return _dedupe(terms)


def client_field_terms(
    first_name: str, last_name: str, email: str, social_links: list[str]
) -> dict[str, list[str]]:
    social: list[str] = []
    for link in social_links:
        social.extend(social_link_terms(link))
    return {
        "name": name_terms(first_name, last_name),
        "email": email_terms(email),
        "social_links": _dedupe(social),
    }


def build_client_search_text(
    first_name: str, last_name: str, email: str, social_links: list[str]
) -> str:
    fields = client_field_terms(first_name, last_name, email, social_links)
    return " ".join(_dedupe([t for terms in fields.values() for t in terms]))


@dataclass(frozen=True)
class Span:
    text: str
    start: int
    end: int

    @property
    def word_count(self) -> int:
        return len(self.text.split())


# A sentence ends with . ! or ? followed by whitespace and an upper-case letter, digit or quote,
# or at a blank line / bullet line break.
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[A-Z0-9])|\n\s*\n|\n(?=\s*[-*•])")
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "inc", "ltd", "llc", "llp", "co", "corp",
    "apt", "no", "vs", "etc", "e.g", "i.e", "u.s", "u.k", "jan", "feb", "mar", "apr", "jun", "jul",
    "aug", "sep", "sept", "oct", "nov", "dec", "approx", "dept", "ave", "fig",
}  # fmt: skip


def _ends_with_abbreviation(text: str) -> bool:
    last_word = text.rstrip().rsplit(maxsplit=1)[-1].rstrip(".").lower() if text.strip() else ""
    return last_word in _ABBREVIATIONS or (len(last_word) == 1 and last_word.isalpha())


def split_sentences(text: str) -> list[Span]:
    spans: list[Span] = []
    start = 0
    for match in _SENTENCE_END_RE.finditer(text):
        candidate = text[start : match.start()]
        if "\n" not in match.group() and _ends_with_abbreviation(candidate):
            continue  # "U.S. Individual" or "Acme Inc. Holdings" is not a boundary
        spans.append(_make_span(text, start, match.start()))
        start = match.end()
    spans.append(_make_span(text, start, len(text)))
    return [s for s in spans if s.text]


def _make_span(text: str, start: int, end: int) -> Span:
    raw = text[start:end]
    lead = len(raw) - len(raw.lstrip())
    trail = len(raw.rstrip())
    return Span(raw.strip(), start + lead, start + trail)


def _split_long(span: Span, max_words: int) -> list[Span]:
    words = list(re.finditer(r"\S+", span.text))
    out = []
    for i in range(0, len(words), max_words):
        window = words[i : i + max_words]
        s, e = window[0].start(), window[-1].end()
        out.append(Span(span.text[s:e], span.start + s, span.start + e))
    return out


def chunk_sentences(text: str, sentences: list[Span], max_words: int) -> list[Span]:
    """Greedy sentence packing into chunks of at most `max_words`, with one sentence of overlap.

    The embedding model (all-MiniLM-L6-v2) truncates at 256 word pieces, so chunks must stay short.
    """
    units: list[Span] = []
    for s in sentences:
        units.extend(_split_long(s, max_words) if s.word_count > max_words else [s])
    if not units:
        return []

    chunks: list[Span] = []
    current: list[Span] = []
    for unit in units:
        words = sum(u.word_count for u in current)
        if current and words + unit.word_count > max_words:
            chunks.append(_join(text, current))
            # Overlap: carry the last sentence over if it leaves room for the next one.
            last = current[-1]
            current = [last] if last.word_count + unit.word_count <= max_words else []
        current.append(unit)
    chunks.append(_join(text, current))
    return chunks


def _join(text: str, spans: list[Span]) -> Span:
    start, end = spans[0].start, spans[-1].end
    return Span(text[start:end], start, end)
