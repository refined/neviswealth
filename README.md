# Nevis Search

A service that stores clients and their documents and lets you quickly find both — by the
exact words they contain and by what they mean.

It was built as the Nevis backend home assignment: a simplified search for a WealthTech platform
where financial advisors need to find clients and client documents quickly and intelligently.

## What it does

- **Stores clients and documents.** Advisors add clients (name, email, description, social links)
  and attach text documents to them through a REST API.
- **Finds clients** by name, email or any part of it, social links and description. Small typos
  are tolerated. For example, `NevisWealth` finds `john.doe@neviswealth.com`.
- **Finds documents by words and by meaning.** A query matches documents that contain its words
  and documents that are about the same thing in other words. For example, `address proof` finds
  a utility bill and a tenancy agreement, although neither contains "address proof".
- **Summarizes documents.** Every document gets a short summary of its key sentences.
- **Explains results.** Each hit has a relevance score, the reason it matched (name, email,
  content, meaning, document type) and a highlighted snippet.

You can use it through:

- **REST API**: create clients, upload documents, search;
- **Swagger UI** at `/api`: try every endpoint from the browser;
- **Simple web UI** at `/ui`: search, browse clients, upload documents and see how each document
  was indexed.

### Under the hood

- Documents are indexed once, when they are uploaded. Full-text search works immediately. Meaning
  vectors and the summary are computed in the background and ready about 50 ms later. Search
  itself only reads indexes and takes ~20-30 ms.
- It runs on a **small CPU-only server** (1 vCPU, 4 GB RAM): one small local language model
  (22M parameters, no GPU, no paid APIs). The whole stack uses about 270 MB of RAM.
- Stack: Python 3.12 · FastAPI · PostgreSQL 17 (full-text search, `pg_trgm`, `pgvector`) · plain
  SQL via psycopg 3 · `fastembed` with `all-MiniLM-L6-v2` · Docker Compose.

## Quick start

```bash
docker compose up --build -d                    # start db + app (first build takes a few minutes)
docker compose exec app python scripts/seed.py  # load 6 demo clients and 16 KYC-style documents
```

Then open:

| URL | What |
|---|---|
| http://localhost:8000/ui | Small UI: search, clients, upload documents, **document inspector** (status, summary, chunks, index terms) |
| http://localhost:8000/api | Swagger UI (interactive API docs); spec at `/api/openapi.json` |
| http://localhost:8000/health | Health check |
| [docs/API.md](docs/API.md) | API reference (Markdown) · [docs/openapi.json](docs/openapi.json) |

The API base URL is `http://localhost:8000` (endpoints such as `/clients` and `/search` sit at the
root, as in the assignment's spec).

**Port and API key.** Copy [`.env.example`](.env.example) to `.env` and change `APP_PORT` if 8000 is
taken; all URLs above then use that port. Inside the container the app always listens on 8000, so
`docker compose exec app python scripts/seed.py` works with any `APP_PORT`. Set `API_KEY` to
require an `X-API-Key` header on every API call.

**Stop / clean up:**

```bash
docker compose stop        # stop the containers; start again with `docker compose start`
docker compose down        # stop and remove the containers; the database volume is kept
docker compose down -v     # also delete the database volume (all clients and documents)
docker compose logs -f app # follow the app logs (it prints the UI and API URLs on startup)
```

The embedding model is baked into the image at build time, so the container runs offline.

### Development

```bash
uv sync                                  # Python 3.12 + dependencies
uv run pytest                            # unit + integration tests (needs Docker: testcontainers)
uv run pytest -m model                   # search-quality eval with the real model (~90 MB download)
uv run ruff check . && uv run ruff format --check .
```

## Example queries

Responses below come from the seeded demo data; IDs and timestamps are shortened.

**Company name → clients by email domain** (`GET /search?q=NevisWealth`)

```json
[
  {
    "type": "client", "score": 1.0, "matched_on": ["email"],
    "highlight": "john.doe@<mark>neviswealth</mark>.com",
    "client": {"id": "9a5f…", "first_name": "John", "last_name": "Doe",
               "email": "john.doe@neviswealth.com", "description": "Senior partner at a consulting firm. …",
               "social_links": ["https://www.linkedin.com/in/john-doe-consulting"], "created_at": "…"}
  },
  {
    "type": "client", "score": 1.0, "matched_on": ["email"],
    "highlight": "emma.j@<mark>neviswealth</mark>.com",
    "client": {"id": "ba50…", "first_name": "Emma", "last_name": "Johnson", "email": "emma.j@neviswealth.com", "…": "…"}
  }
]
```

**Concept → documents that never mention it** (`GET /search?q=address proof&type=document`)

Document hits carry the full `Document` (the assignment's fields plus indexing status and summary).
The owner is fetched separately with `GET /clients/{client_id}`:

```json
[
  {
    "type": "document", "score": 0.727, "matched_on": ["document_type:proof_of_address"],
    "highlight": "This assured shorthold <mark>tenancy</mark> <mark>agreement</mark> is made between Kensington Lettings Ltd …",
    "document": {"id": "…", "client_id": "…", "title": "Residential tenancy agreement",
                 "content": "This assured shorthold tenancy agreement is made between …", "created_at": "…",
                 "status": "ready", "processed_at": "…", "error": null,
                 "summary": "This assured shorthold tenancy agreement is made between Kensington Lettings Ltd (the landlord) and Aisha Khan (the tenant). The tenant will occupy the flat at 18 Pembridge Gardens, London W2 4DU. Monthly rent is GBP 2,350, payable in advance on the first day of each month."}
  },
  {
    "type": "document", "score": 0.7, "matched_on": ["document_type:proof_of_address"],
    "highlight": "Florida Power &amp; Light monthly statement for account 4471-2290. Service address: 1450 Brickell Avenue, …",
    "document": {"title": "Electricity bill - March 2026", "status": "ready",
                 "summary": "Florida Power & Light monthly statement for account 4471-2290. Billing period: February 12 to March 13, 2026. Amount due: $143.27, payable by April 2, 2026.",
                 "…": "…"}
  }
]
```

**More queries:**

| Query | Top results | `matched_on` |
|---|---|---|
| `Jonh Doe` (typo) | client John Doe, score 0.67 | `name` (trigram fuzzy match) |
| `where did the money come from` | Source of wealth declaration (0.79), then the checking account statement with the $18.5M share-sale wire (0.30) | `semantic`, `document_type:source_of_wealth` |
| `proof of residence` | tenancy agreement, water bill, electricity bill | `document_type:proof_of_address` (+ `semantic`) |
| `electricity usage` right after upload | the bill, with `status: "pending"` | `content`: full-text works before async indexing finishes |
| `qwertyzxcv`, `chocolate cake recipe` | `[]` | nothing passes the relevance gates |

**Errors:**

```text
POST /clients (same email, different case)  → 409 {"detail": "A client with this email already exists"}
GET  /clients/00000000-0000-0000-0000-000000000000 → 404 {"detail": "Client not found"}
POST /clients {"email": "nope", ...}         → 422 {"detail": [{"loc": ["body", "email"], "msg": "value is not a valid email address: ...", ...}]}
GET  /search?q=%20%20                        → 422 (blank query)
```

## How it works

All expensive work happens **at write time**, split into a synchronous pre-processing step and an
asynchronous post-processing step. The query path only does index lookups plus one ~3 ms query
embedding.

```
POST /clients ─────────────► clients
  sync: identity terms (name, email parts, domain label "neviswealth", social handles)
        -> GIN trigram index; description -> tsvector (GIN)

POST /clients/{id}/documents ─► documents (status=pending) ─► 201
  sync:  Postgres generated tsvector (title weight A, content weight B) -> GIN inverted index
  async: background worker (queue = documents table, FOR UPDATE SKIP LOCKED, 3 retries)
         ├─ split sentences -> chunks of <=120 words (1-sentence overlap)
         ├─ embed "title. chunk" -> document_chunks.embedding (pgvector HNSW, cosine)
         └─ extractive summary (same model) -> status=ready

GET /search?q=
  clients:   substring (LIKE) + typo-tolerant word_similarity (pg_trgm) on identity terms,
             full-text on description
  documents: (1) full-text   websearch_to_tsquery over title+content
             (2) semantic    query embedding -> nearest chunks (HNSW) -> best chunk per document
             (3) taxonomy    query -> known document type (e.g. proof_of_address) ->
                             documents containing the type's phrases OR close to the type centroid
  scoring:   each signal -> calibrated confidence -> noisy-OR into a score in [0, 1];
             a document needs a lexical hit, a taxonomy hit, or semantic similarity >= 0.22
```

**Extractive summary.** The document's sentences are embedded with the same model. The summarizer
picks the 2-3 sentences closest to the document centroid, with a small lead bias. It uses MMR to
avoid near-duplicates and keeps the sentences in document order.

**Inspector UI.** `/ui` shows each document's indexing status, timings, summary and chunks (click a
chunk to highlight it in the content), plus the lexemes in the inverted index. It uses only the
public API.

## Design decisions and trade-offs

1. **A small local encoder instead of an LLM.**
   - The target is a 1 vCPU / 4 GB box.
   - A generative 0.5-1.5B model needs 0.5-1.5 GB of RAM and seconds per document, and it makes a
     poor retrieval embedder.
   - A 22M-parameter encoder does both jobs (semantic search and extractive summaries)
     deterministically, offline and for free.
   - The summarizer sits behind a `Summarizer` protocol, so an abstractive LLM can be plugged in
     later.

2. **The model was chosen by measurement** (`scripts/spike_embeddings.py`, 1 thread, seed corpus):

   | Model | hit@1 | MRR | query ms | doc ms | +RAM |
   |---|---|---|---|---|---|
   | **all-MiniLM-L6-v2** (chosen) | **1.00** | **1.00** | **2.9** | **26** | 156 MB |
   | bge-small-en-v1.5 | 0.91 | 0.93 | 13 | 148 | 202 MB |
   | snowflake-arctic-embed-xs | 0.82 | 0.89 | 4.2 | 26 | 150 MB |
   | granite-embedding-small-english-r2 | 0.55 | 0.71 | 14 | 159 | 275 MB |
   | bge-base-en-v1.5 | 0.82 | 0.88 | 19 | 196 | 504 MB |

3. **Embeddings alone do not solve "address proof → utility bill"; a document-type taxonomy does.**
   - The spike showed that no small model ranks a utility bill high for "address proof". bge-small
     put it 4th, after a passport and a driver license. "A utility bill *is* a proof of address" is
     KYC domain knowledge, not word similarity.
   - [`app/search/taxonomy.py`](app/search/taxonomy.py) holds 13 wealth/KYC document types with
     phrases. The query is matched against these phrases with the same model, and only above a
     noise threshold (see 6).
   - When the query matches a type, documents of that type are retrieved by its phrases
     (full-text: "utility bill", "tenancy agreement", …) and by closeness to the type's embedding
     centroid. The second path catches documents that contain none of the phrases, such as a
     risk questionnaire for "risk appetite".
   - It is explainable (`matched_on: document_type:proof_of_address`) and cheap. It generalized to
     a water bill that was not in the corpus when the taxonomy was written.

4. **Postgres does everything:**
   - full-text search (inverted index);
   - trigram fuzzy matching;
   - vector search (pgvector HNSW);
   - the job queue.

   There is one datastore and one transaction model, and Postgres adds about 40 MB of RAM. A
   dedicated engine (OpenSearch, Qdrant) would be the next step at millions of documents, not here.

   **Document text is stored in Postgres** (`documents.content`), not as files: documents arrive
   as JSON text, the full-text index is generated from that column in the same transaction, and
   one backup covers everything. Large values are compressed automatically (TOAST). Binary uploads
   such as PDFs would go to object storage, with the extracted text still stored here.

5. **Plain SQL instead of an ORM and Alembic.**
   - The schema is described in SQL:
     [`db/migrations/0001_initial.sql`](db/migrations/0001_initial.sql).
   - A small runner ([`app/table/migrate.py`](app/table/migrate.py)) applies new files in
     order, records them in `schema_migrations` and takes an advisory lock so concurrent starts
     are safe. To change the schema, add `0002_….sql`.
   - All queries live in [`app/table/`](app/table), written inline in small functions on top of
     psycopg 3 and its connection pool. Rows are mapped to dataclasses.
   - Most of the work happens in SQL anyway: full-text search, trigrams, vector search,
     `SKIP LOCKED`. An ORM would add a dependency and a layer of indirection without saving code.

6. **Relevance gates instead of "always return top-k".** Vector search always returns *something*,
   so each signal has a calibrated gate.
   - **Semantic similarity:** at least 0.22. Unrelated queries stay below 0.15 with this model.
   - **Query-to-type similarity:** at least 0.40. Short unrelated queries, such as person names,
     reach up to ~0.32 against short phrases.
   - **Scores:** confidences are combined with a noisy-OR, so one strong signal suffices,
     agreeing signals reinforce each other, and client and document scores share one [0, 1] scale.

7. **Asynchronous indexing with a database-backed queue in the app process.**
   - There is no Redis or Celery. The worker is a thread that claims documents with
     `FOR UPDATE SKIP LOCKED` (safe to run more workers), retries 3 times and marks failures.
   - A result is discarded if the document was re-queued meanwhile.
   - Full-text search works immediately, while semantic search and the summary follow ~50 ms later.
   - On 1 vCPU, ONNX runs with one thread so indexing does not starve requests.

8. **API details:**
   - Email is unique case-insensitively (`409` on duplicates).
   - IDs are UUIDs (`422` if malformed, `404` if unknown).
   - `/search` returns the flat array the spec asks for, with a `type` discriminator, `score`,
     `matched_on` and an HTML-safe `highlight`.
   - **Compatibility with the assignment's spec is additive only.** Request bodies are exactly
     the spec's. `Client` adds `created_at`; `Document` adds `status`, `summary`, `processed_at`
     and `error`. The spec leaves `/search` items untyped; ours wrap the spec's `Client` /
     `Document` objects with `type`, `score`, `matched_on` and `highlight`; a document's client is
     fetched with `GET /clients/{client_id}`. A contract test (`tests/integration/test_misc_api.py`) fails if a
     spec field disappears.
   - Extra endpoints: `GET` routes, `/documents/{id}/index`, `/reprocess`, `/health`.
   - Document hits include the full `content`, as the spec's `Document` does. For very large
     documents a `fields` parameter or snippet-only hits would be the next step.

## Evaluation

`uv run pytest -m model` seeds [db/seed/seed_corpus.json](db/seed/seed_corpus.json) through the
API, runs the real model and checks the queries in
[tests/eval/queries.json](tests/eval/queries.json). The corpus has 6 clients and 16 documents.

| Set | Result |
|---|---|
| 11 development queries ("address proof", "proof of income", "estate planning", …) | 11/11 have an expected document in the top 3 |
| 10 held-out queries ("residence verification", "rental contract", "where did the money come from", …) | 9/10. **Known miss:** "KYC identification". The small model does not know "KYC", and its similarity to the identity type (0.29) is inside the noise band, so it is tracked as a strict `xfail`. |
| 4 unrelated queries | 4/4 return `[]` |
| Client queries ("NevisWealth", "Jonh Doe", "maria garcia", "chencapital") | the client ranks first |

The held-out queries were written before any tuning. Later changes were driven by
development-set misses; for example, centroid expansion was added because of "risk appetite". The
taxonomy threshold ended *above* the noise band (0.40) instead of being lowered to rescue the one
held-out miss.

## Tests

- **Unit** (`tests/unit`)
- **Integration** (`tests/integration`) runs against a real Postgres + pgvector (testcontainers)
  with a deterministic hashing embedder
- **Config** (`tests/unit/test_config.py`): environment parsing of `Settings`.
- **Eval** (`tests/eval`): quality with the real model, as above.

## Limitations and future work

- **English only:** Postgres `english` text configuration and an English model.
  `paraphrase-multilingual-MiniLM-L12-v2` would be the drop-in option.
- **Plain-text documents only:** there is no PDF or OCR ingestion.
- **Summaries are extractive.** An abstractive summarizer (local Qwen via llama.cpp, or an API)
  can be added behind `Summarizer` on a bigger box.
- **The taxonomy is hand-curated.** Next steps would be to grow it from advisors' real document
  types and classify documents at indexing time (store `document_type`) so they can be filtered
  by type.
- **Client descriptions are matched by full-text only:** "retired doctor" will not find "retired
  physician". The same embedding pipeline could be applied to descriptions.
- **Thresholds are calibrated for this model.** Changing models means re-running the spike and the
  eval and reprocessing documents. The vector dimension is 384, fixed by the migration.
- **Not implemented:** pagination beyond `limit`, per-advisor access scoping, deletion endpoints.
  The service has not been deployed.
- Chunk offsets are Python code-point indices, so the UI highlight can shift for emoji or other
  astral characters.

## Project layout

```
app/                 application code
  api/               HTTP endpoints (FastAPI routers)
  table/             database layer: all SQL queries, row types, migration runner
  indexing/          write path: text processing, embeddings, summaries, background worker
  search/            read path: document-type taxonomy, scoring, result assembly
  static/            build-free UI (vanilla JS)
db/
  migrations/        database schema as numbered .sql files
  seed/              demo clients and documents
docs/                API reference (Markdown + OpenAPI spec)
scripts/             seeding, embedding-model spike, OpenAPI export
tests/
  unit/              pure logic, no database
  integration/       API + real Postgres (testcontainers), fake embedder
  eval/              search quality with the real model (pytest -m model)
```

## Use of LLMs

As the assignment encourages, this was built with an AI coding assistant (Claude Code), which
helped with design discussion, the embedding spike, implementation, tests and docs. Every
decision above is backed by measurements in the repository: the spike script and the eval suite.
