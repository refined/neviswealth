# Nevis Search API reference

- Interactive docs: `/docs` (Swagger UI).
- Machine-readable spec: [`openapi.json`](openapi.json) (OpenAPI 3.1). Regenerate it with
  `uv run python -m scripts.export_openapi`.

## Conventions

- **IDs are UUIDs.** A malformed ID in the path returns `422`; a well-formed but unknown ID returns
  `404`.
- **Errors use FastAPI's shape:** `{"detail": "..."}`. For validation errors (`422`), `detail` is a
  list of `{loc, msg, type}` objects.
- **Timestamps** are ISO 8601 in UTC.
- **Authentication** is optional. When the server has `API_KEY` set, every endpoint except
  `/health`, `/docs`, `/openapi.json` and `/ui` requires an `X-API-Key: <key>` header. A missing or
  wrong key returns `401`.

## Clients

### `POST /clients` → `201 Created`

```json
{
  "first_name": "John",                      // required, 1-100 chars
  "last_name": "Doe",                        // required, 1-100 chars
  "email": "john.doe@neviswealth.com",       // required, valid email, unique (case-insensitive)
  "description": "Senior partner ...",       // optional, up to 5000 chars
  "social_links": ["https://www.linkedin.com/in/john-doe"]  // optional, up to 20 strings
}
```

The response is a `Client` (the spec's fields plus `created_at`) and a `Location: /clients/{id}`
header.

| Status | When |
|---|---|
| `409` | A client with the same email already exists. |
| `422` | Invalid body: missing field, invalid email, blank name, wrong types or broken JSON. |

### `GET /clients?limit=50&offset=0` → `200`

Clients, newest first. `limit` is 1-200.

### `GET /clients/{client_id}` → `200` | `404`

## Documents

### `POST /clients/{client_id}/documents` → `201 Created`

```json
{ "title": "Electricity bill - March 2026", "content": "Monthly statement ..." }
```

- `title` is 1-300 chars and `content` is 1-200,000 chars, both after trimming.
- The response is a `Document` with `Location: /documents/{id}`. Unknown client → `404`.

Fields added beyond the spec:

| Field | Meaning |
|---|---|
| `status` | `pending` → `processing` → `ready` (or `failed`). The document is searchable by its words immediately. Semantic search and the summary become available at `ready`, usually within ~100 ms. |
| `summary` | Extractive summary (2-3 key sentences), set at `ready`. |
| `processed_at` | When indexing finished. |
| `error` | Last indexing error, if any. |

### `GET /clients/{client_id}/documents` → `200` | `404`

The client's documents, newest first.

### `GET /documents/{document_id}` → `200` | `404`

### `GET /documents/{document_id}/index` → `200` | `404`

Shows what indexing produced; the UI's document inspector uses it.

| Field | Meaning |
|---|---|
| `chunks` | `{ordinal, text, char_start, char_end}`. These are the embedded passages; offsets point into `content`. |
| `lexemes` | Top stemmed terms of the full-text index, with counts and weights (`A` = title, `B` = content). |
| `processing_ms`, `attempts`, `embedding_model`, `summary`, `status`, `error` | Indexing metadata. |

### `POST /documents/{document_id}/reprocess` → `202 Accepted` | `404`

Queues the document for indexing again, for example after a failure. Returns the document with
`status: "pending"`.

## Search

### `GET /search?q=<text>&type=<client|document>&limit=20` → `200`

| Parameter | Rules |
|---|---|
| `q` | Required, 1-256 chars, must not be blank. |
| `type` | Optional filter. |
| `limit` | 1-100. |

Invalid parameters → `422`.

The response is a single array sorted by `score` (descending). Each item has a `type`
discriminator:

```json
{
  "type": "document",                 // or "client"
  "score": 0.7,                        // relevance in [0, 1]
  "matched_on": ["document_type:proof_of_address"],
  "highlight": "Miami-Dade <mark>Water</mark> and Sewer Department",
  "document": {                        // "client": {...Client} for client hits
    "id": "…", "client_id": "…", "client_name": "Maria Garcia",
    "title": "Water bill - April 2026", "summary": "…", "status": "ready",
    "created_at": "…"
  }
}
```

`matched_on` explains why an item matched:

| Value | Meaning |
|---|---|
| `name`, `email`, `social_links` | The query is in (or, for typos, close to) the client's identity terms. For an email this includes its parts and the company domain (`neviswealth`). |
| `description` | Full-text match in the client description. |
| `title`, `content` | Full-text (stemmed) match in the document. |
| `semantic` | One of the document's chunks is close in meaning to the query (embedding similarity). |
| `document_type:<key>` | The query refers to a known document type (e.g. "address proof" → `proof_of_address`) and the document is of that type. |

`highlight` is safe HTML: all text is escaped and matches are wrapped in `<mark>` tags. It may be
`null`.

An empty array means nothing is relevant enough. Unrelated queries such as `qwertyzxcv` return `[]`
instead of the "nearest" documents.

## Health

### `GET /health` → `200` | `503`

```json
{"status": "ok", "database": "ok", "embedding_model": "sentence-transformers/all-MiniLM-L6-v2", "worker_running": true}
```
