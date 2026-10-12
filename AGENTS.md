# AGENTS.md

Working rules for anyone (human or AI agent) contributing to this repository.

## Project

Nevis Search API: a home assignment (see `Nevis - Backend Home Assignment.pdf`). It is a REST API for
wealth advisors that lets them:
- create clients and attach text documents to them;
- search across clients (name, email, description) and documents (semantic + lexical);
- get a short extractive summary of each document.

## Language

- **Every repository artifact is written in English.** This covers code, identifiers, comments,
  docstrings, README and docs, commit messages, PR descriptions, UI text, log messages and error
  messages.
- Conversation with the maintainer may happen in another language. That never leaks into files.

## Product constraints

- **English-only content.** Full-text search uses the Postgres `english` configuration and the
  embedding model is English. Other languages are out of scope and are listed as a limitation in the
  README.
- **Weak, CPU-only deployment target** (about 1 vCPU, 4 GB RAM).
  - Prefer small local models (an ONNX encoder via `fastembed`).
  - No GPU, no torch.
  - No paid external services are required to run the project.
- **Expensive work happens at indexing time**, never at query time:
  - synchronous pre-processing at write time: normalization, `tsvector`;
  - asynchronous post-processing in the background worker: chunking, embeddings, summary.
- The query path does only index lookups plus one query embedding.
- **Summaries are extractive.** They are produced by the same embedding model; there is no generative LLM.

## Stack

- Python 3.12 and `uv` for dependency management.
- FastAPI + Pydantic v2. OpenAPI/Swagger is served at `/docs`.
- PostgreSQL 17 with the `pgvector` and `pg_trgm` extensions, accessed with plain SQL through
  psycopg 3 and `psycopg_pool`. There is no ORM and no Alembic.
- Settings are a plain dataclass read from environment variables (`app/config.py`).
- `fastembed` with `sentence-transformers/all-MiniLM-L6-v2`. It was chosen by
  `scripts/spike_embeddings.py`. Search thresholds are calibrated for this model, so after
  changing it re-run the spike and `pytest -m model`.
- pytest, ruff.
- Docker + docker-compose (`db` + `app`). The background worker runs inside the app process.

## Commands

```bash
uv sync                                   # install dependencies
docker compose up --build                 # run db + app on http://localhost:8000
uv run pytest                             # unit + integration tests
uv run pytest -m model                    # quality eval with the real embedding model
uv run ruff check . && uv run ruff format --check .
```

## Conventions

- **Database:**
  - The schema is defined only in SQL: `db/migrations/NNNN_name.sql`.
  - Migrations are forward-only. To change the schema, add a new numbered file; never edit an
    applied one. `python -m app.table.migrate` applies pending files.
  - All SQL lives in `app/table/`, written inline inside small functions that take a
    `psycopg.Connection`. Rows map to dataclasses in `app/table/records.py`.
  - Other layers call these functions and never write SQL themselves.
  - Use bound parameters only; never build SQL with f-strings.
- Seed data lives in `db/seed/`; search-quality queries live in `tests/eval/queries.json`.
- Use type hints everywhere. Keep functions small and pure where possible; the indexing and search
  logic must be unit-testable without a database.
- External dependencies sit behind small protocols (`Embedder`, `Summarizer`). Tests use fakes, so
  they stay fast and offline.
- Every behavior change comes with tests. Edge cases matter: empty or huge input, unknown IDs,
  duplicates, garbage queries.
- Follow REST conventions and return precise HTTP codes (201, 202, 404, 409, 422).
- Record non-obvious design decisions and trade-offs in the README "Design decisions" section.
- Keep the README examples in sync with real API responses.
- Commits are small and focused, and their messages are written in the imperative mood.
