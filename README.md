# Advisor Search

## What it is

Financial advisors need to find their clients and the clients' documents quickly and
intelligently. This is a search API across clients and documents. It supports three cases:

1. **Find clients** by their email, name or description. For example, `NevisWealth` should find
   the client with the email `john.doe@neviswealth.com`.
2. **Find documents by similar terms**, not only exact words. For example, `address proof` should
   also find documents that contain "utility bill".
3. **Summarize a document**: a quick summary of its content.

## The solution at a glance

The API stores clients and their text documents and searches both:

| Endpoint | What it does |
|---|---|
| `POST /clients` | Create a client (name, email, description, social links). |
| `POST /clients/{id}/documents` | Attach a text document to a client. |
| `GET /search?q=…` | Search clients and documents; one list sorted by relevance. |

Supporting endpoints are `GET /clients`, `GET /clients/{id}`, `GET /clients/{id}/documents`,
`GET /documents/{id}`, `GET /documents/{id}/index`, `POST /documents/{id}/reprocess` and
`GET /health`. See them all, with request and response schemas, in the **Swagger UI at
[`/api`](http://localhost:8000/api)** or in [docs/API.md](docs/API.md).

**The main idea: do the expensive work once, when data is written, so search only reads indexes.**
- When a document is uploaded, Postgres indexes its words immediately.
- A background worker then splits it into chunks, computes a "meaning vector" for each chunk and
  writes a summary. This takes about 50 ms.
- A search query only looks things up in those indexes and takes ~20-30 ms.

Diagrams of how clients and documents flow through the system are in
[How it works](#how-it-works).

You can use it through the **REST API**, the **Swagger UI** (`/api`) or a **simple web UI**
(`/ui`): search, browse clients, upload documents and inspect how each document was indexed.

It runs on a small CPU-only server (1 vCPU, 4 GB RAM): no GPU, no paid APIs, and the whole stack
uses about 270 MB of RAM.

## How it works

`GET /search` runs the client search and the document search and merges them into one list
sorted by score. Each search has the same three steps: **find** candidates through indexes,
**score** them from 0 to 1, **highlight** why they matched.

### Part 1. Finding clients

**The problem.** People search with a fragment, not the whole field: "NevisWealth" is only a piece
of `john.doe@neviswealth.com`, and names get mistyped.

**The whole flow.** Everything happens synchronously, in the request.

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 1000}}}%%
flowchart TB
    input(["📇 POST /clients<br/>name · email · social links · description"])

    subgraph write["✍️ On write — synchronous, one transaction"]
        direction LR
        split["<b>App</b>: normalize and split into identity terms<br/><i>john.doe@neviswealth.com → john · doe · neviswealth · …</i>"]
        stem["<b>Postgres</b>: stem the description words<br/><i>retired physicians → retir · physician</i>"]
    end

    subgraph db["🐘 Postgres"]
        direction LR
        search_text[("clients.search_text<br/>GIN trigram index")]
        description_tsv[("clients.description_tsv<br/>GIN full-text index")]
    end

    input --> split & stem
    split --> search_text
    stem --> description_tsv

    subgraph search["🔍 GET /search?q=… — app/services/search_clients.py"]
        direction TB
        subgraph find["1 · Find — one SQL query, three conditions ORed"]
            direction LR
            part["<b>part of a term</b><br/>LIKE '%q%'<br/><i>NevisWealth</i>"]
            typo["<b>with a typo</b><br/>trigram similarity ≥ 0.5<br/><i>Jonh Doe</i>"]
            words["<b>description words</b><br/>full-text match<br/><i>retired physician</i>"]
        end
        score["<b>2 · Score</b> from 0 to 1<br/>whole term 1.0 · part of a term 0.9 · typo ≤ 0.85 · description 0.55–0.75<br/>combined with noisy-OR"]
        highlight["<b>3 · Highlight</b><br/>the email, name or link with the query in &lt;mark&gt;"]
        find --> score --> highlight
    end

    search_text --> part & typo
    description_tsv --> words

    classDef io fill:#ecfdf3,stroke:#3f9b5f,color:#0f3d1f
    classDef step fill:#fff8e6,stroke:#d4a12a,color:#3d2c00
    classDef store fill:#eaf2ff,stroke:#4a78c9,color:#0b2545
    class input io
    class split,stem,part,typo,words,score,highlight step
    class search_text,description_tsv store
```

**When a client is created**, the app splits their identity into searchable terms and stores them
in one column (`search_text`):

- `john.doe@neviswealth.com` → `john`, `doe`, `neviswealth.com`, `neviswealth` (the company name
  without `.com`);
- `https://www.linkedin.com/in/john-doe` → `linkedin`, `john-doe`, `john`, `doe`;
- the first and last name.

**Two ideas make this searchable fast:**

- **Trigrams.** A word is cut into overlapping 3-letter pieces: `john` → `joh`, `ohn` (plus padded
  pieces for its start and end). Two words that share most of their pieces are probably the same word with a typo, so
  `Jonh` still finds `John`. Postgres builds an index over these pieces, so a substring search
  (`…neviswealth…`) or a "similar word" search does not have to read every row.
- **Inverted index.** Like the index at the back of a book, it maps each word to the rows that
  contain it. It is used for the client description, with stemming, so "investors" also matches
  "investor".

**Scoring in plain words:**

| Match | Score |
|---|---|
| exact term | 1.0 |
| the query is part of a term | 0.9 |
| typo-tolerant match | 0.5-0.85 |
| description match | adds to the score |

**Examples** (demo data):

| Query | Result |
|---|---|
| `NevisWealth` | both clients with a `@neviswealth.com` email, score 1.0, matched on `email` |
| `Jonh Doe` | John Doe, score 0.67, typo-tolerant match on `name` |
| `retired physician` | Maria Garcia, matched on her `description` |

**Postgres:**
- the `pg_trgm` extension with a GIN trigram index on `search_text`;
- a `tsvector` column (the words of the description), generated by Postgres, with a GIN inverted
  index.

### Part 2. Finding documents by words and by meaning

**The problem.** An advisor types `address proof`; the right document says "electricity bill" and
never uses those words. Exact word search alone cannot find it, so three signals are combined.

**On upload.** Words are indexed at once; meaning and summary come from a background worker.

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 1000}}}%%
flowchart TB
    input(["📄 POST /clients/{id}/documents<br/>title · content"])

    subgraph sync["⚡ Synchronous — same transaction"]
        direction LR
        row[("documents<br/>status = pending")]
        tsv[("documents.tsv<br/>GIN full-text index<br/><i>title words weigh more</i>")]
    end

    subgraph worker["⚙️ Background worker"]
        direction LR
        sentences["split into sentences"]
        chunks["chunks of ≤ 120 words,<br/>each prefixed with the title"]
        embed["🧠 model: chunk → 384 numbers"]
        summarize["🧠 model picks up to 3<br/>most central sentences"]
        sentences --> chunks --> embed
        sentences --> summarize
    end

    subgraph stored["🐘 Postgres"]
        direction LR
        vectors[("document_chunks.embedding<br/>HNSW vector index")]
        summary[("documents.summary<br/>status = ready")]
    end

    input --> row & tsv
    row -- "queue" --> sentences
    embed --> vectors
    summarize --> summary

    classDef io fill:#ecfdf3,stroke:#3f9b5f,color:#0f3d1f
    classDef step fill:#fff8e6,stroke:#d4a12a,color:#3d2c00
    classDef store fill:#eaf2ff,stroke:#4a78c9,color:#0b2545
    class input io
    class sentences,chunks,embed,summarize step
    class row,tsv,vectors,summary store
```

**On search.**

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 1000}}}%%
flowchart TB
    input(["🔍 GET /search?q=address proof"])
    query["🧠 model: query → query vector<br/>and its document type: <i>address proof → proof of address</i>"]

    subgraph indexes["🐘 Postgres"]
        direction LR
        vectors[("document_chunks.embedding<br/>HNSW vector index")]
        tsv[("documents.tsv<br/>GIN full-text index")]
    end

    subgraph find["1 · Find — three searches"]
        direction LR
        meaning["<b>meaning</b><br/>nearest chunks to the query vector,<br/>pulled towards the document type"]
        words["<b>words</b><br/>full-text match<br/><i>electricity usage</i>"]
        type["<b>document type</b><br/>contains a phrase of the type<br/><i>utility bill, lease agreement</i>"]
    end

    score["<b>2 · Score</b> from 0 to 1<br/>each signal → a confidence · needs words, a type phrase or meaning ≥ 0.22<br/>combined with noisy-OR"]
    highlight["<b>3 · Highlight</b><br/>matched words marked by ts_headline · else the closest chunk · else the start"]

    input --> query --> find
    tsv --> words & type
    vectors --> meaning
    find --> score --> highlight

    classDef io fill:#ecfdf3,stroke:#3f9b5f,color:#0f3d1f
    classDef step fill:#fff8e6,stroke:#d4a12a,color:#3d2c00
    classDef store fill:#eaf2ff,stroke:#4a78c9,color:#0b2545
    class input io
    class query,words,meaning,type,score,highlight step
    class tsv,vectors store
```

**1. Words (full-text search).** An inverted index over the title and the content, with stemming;
words in the title count more. It is built by Postgres in the same transaction as the upload, so
a document is searchable by its words immediately. Example: `electricity usage` finds the
electricity bill right after upload, before background processing has finished.

**2. Meaning (embeddings).**

- **An embedding** is a list of 384 numbers that describes what a text is about. Texts about the
  same thing get vectors that point in a similar direction, even with no words in common.
  "Similar" is measured with cosine similarity.
- **The model.** The vectors come from **all-MiniLM-L6-v2**, a small *encoder* language model
  (22M parameters). It is not a chat LLM like GPT: it does not write text, it only turns text into
  vectors, in a few milliseconds on a CPU. It runs locally inside the container.
- **How it is used.** On upload, each document is cut into chunks of up to 120 words, and each
  chunk is embedded together with the document title. At search time only the query is embedded,
  and the nearest chunks are found.
- **Example:** `where did the money come from` → *Source of wealth declaration* (score 0.79).

**3. Domain knowledge (document types).**

- **Why it is needed.** Measured on the demo data, the small model alone does not know that "a
  utility bill *is* a proof of address": it ranked the bill 4th or lower for `address proof`. That
  is KYC domain knowledge, not word similarity.
- **What it is.** A small taxonomy of 13 wealth-management / KYC document types
  ([app/search/taxonomy.py](app/search/taxonomy.py)), each with typical phrases. For example,
  *proof of address*: utility bill, electricity bill, gas bill, tenancy agreement, lease
  agreement, …
- **How it is used.**
  1. The query is compared to these phrases with the same model.
  2. If it clearly names a type, two things happen:
     - documents that contain one of the type's phrases get a *document type* signal;
     - the meaning search runs with the query vector pulled towards the type's "average meaning"
       (the centroid of its phrases), so documents of that type come closer. See *Blended query
       vector* in the deep dive.
- **Examples:**
  - `address proof` → the *Residential tenancy agreement* and the *Electricity bill*;
  - a water bill uploaded later, not part of the demo data, was found for `proof of residence`
    the same way.

**Combining the signals.**
- Each signal becomes a confidence. They are combined into one score from 0 to 1: one strong
  signal is enough, and signals that agree add up.
- Each signal also has a minimum threshold, so a meaningless query returns nothing instead of the
  "least bad" documents: `qwertyzxcv` → `[]`.
- Every hit says why it matched (`matched_on`: `title`, `content`, `semantic`,
  `document_type:proof_of_address`) and has a highlighted snippet.

**Postgres:**
- a generated `tsvector` column with a GIN inverted index (words);
- the `pgvector` extension with an HNSW index for nearest-vector search (meaning);
- `ts_headline` for highlighted snippets;
- a tiny custom aggregate (`tsquery_or_agg`) that searches all phrases of a document type in one
  query.

### Part 3. Document summaries

**What it is.** An *extractive* summary: the 2-3 sentences that best represent the document,
quoted as they are. It uses the same model as search:
1. embed every sentence;
2. their average vector is "what the document is about";
3. take the sentences closest to it, skipping near-duplicates;
4. keep them in the original order.

**Example**, the electricity bill:

> Florida Power & Light monthly statement for account 4471-2290. Billing period: February 12 to
> March 13, 2026. Amount due: $143.27, payable by April 2, 2026.

**Why not a generative LLM.**
- On the target server, a generative model of 0.5-1.5B parameters needs 0.5-1.5 GB of RAM and
  seconds per document.
- It can also invent facts.
- An extractive summary is instant and deterministic, and it only quotes the document.

**When it is computed.** In the background, right after upload, together with the chunk vectors.
- The document's `status` goes `pending → processing → ready`. A failing document is tried up to 3
  times, then marked `failed` with the error. `POST /documents/{id}/reprocess` queues it
  again.
- The queue is the `documents` table itself. A worker thread inside the app claims rows with
  `FOR UPDATE SKIP LOCKED`, so no Redis or Celery is needed and more workers could run safely.

### Postgres is the only datastore

| Need | Postgres feature |
|---|---|
| words, stemming | `tsvector` + GIN inverted index |
| substrings and typos | `pg_trgm` + GIN trigram index |
| meaning | `pgvector` + HNSW index |
| background job queue | the `documents` table + `FOR UPDATE SKIP LOCKED` |
| schema | plain SQL migrations in [db/migrations/](db/migrations/0001_initial.sql) |

Document text is stored in Postgres too (compressed automatically), so one transaction and one
backup cover everything. Postgres adds about 40 MB of RAM. A dedicated search engine would only
pay off at millions of documents.

## Running it

```bash
docker compose up --build -d                    # start db + app (first build takes a few minutes)
docker compose exec app python scripts/seed.py  # load 6 demo clients and 16 KYC-style documents
```

| URL | What |
|---|---|
| http://localhost:8000/ui | Web UI: search, clients, uploads, document inspector |
| http://localhost:8000/api | Swagger UI; spec at `/api/openapi.json` |
| http://localhost:8000/health | Health check |

The embedding model is baked into the image at build time, so the container runs offline.

- **Port and API key.** Copy [.env.example](.env.example) to `.env`. Change `APP_PORT` if 8000 is
  taken; set `API_KEY` to require an `X-API-Key` header on every API call.
- **Database client.** Postgres is reachable from this machine only:
  `postgresql://advisor:advisor@127.0.0.1:5433/advisor`. Change the port with `DB_PORT`.
### Stopping

Run these from the project folder. They affect only this project's containers (Compose project
`neviswealth`); other Docker containers keep running.

```bash
docker compose stop        # stop the containers, keep everything
docker compose start       # start them again
docker compose down        # remove the containers, keep the database
docker compose down -v     # remove the containers and delete the database
```

From any other folder, name the project explicitly: `docker compose -p neviswealth stop`.

## API notes

- Full reference: Swagger UI at `/api`, [docs/API.md](docs/API.md),
  [docs/openapi.json](docs/openapi.json).
- **Compatible with the assignment's spec, additions only:**
  - request bodies are exactly the spec's;
  - `Client` adds `created_at`;
  - `Document` adds `status`, `summary`, `processed_at` and `error`.

  A contract test fails if a spec field disappears.
- **Search hits** wrap the spec's `Client` / `Document` with `type`, `score`, `matched_on` and
  `highlight`. A document hit carries `client_id`; the client itself is fetched with
  `GET /clients/{id}`.
- **Errors:**
  - `409` for a duplicate email (case-insensitive);
  - `404` for an unknown ID;
  - `422` for invalid input, a malformed ID or a blank query.

## Tests

```bash
uv sync                                  # Python 3.12 + dependencies
uv run pytest                            # unit + integration tests (needs Docker for testcontainers)
uv run pytest -m model                   # search-quality evaluation with the real model
uv run ruff check . && uv run ruff format --check .
```

- **Unit** (`tests/unit`): text processing, chunking, summaries, scoring, taxonomy, settings. No
  database.
- **Integration** (`tests/integration`): the whole API against a real Postgres with pgvector,
  started by testcontainers, with a fast fake embedder. Covers:
  - HTTP codes and validation;
  - the background worker, retries and failures;
  - migrations;
  - HTML escaping;
  - the API key;
  - the spec contract.
- **Search quality** (`tests/eval`): loads the demo data with the real model and checks
  [42 queries with known answers](tests/eval/queries.json):

  | Set | Result |
  |---|---|
  | 11 development queries ("address proof", "proof of income", …) | 11/11 find an expected document in the top 3 |
  | 31 held-out queries ("residence verification", "who inherits the house", …) | 29/31 |
  | 4 nonsense queries | 4/4 return nothing |
  | client queries ("NevisWealth", "Jonh Doe", …) | the client ranks first |

  The one miss is "KYC identification": the small model does not know the abbreviation "KYC". It
  is tracked as an expected failure.

## Project layout

```
app/                 application code
  api/               HTTP layer: input validation, endpoints, web UI, error mapping
  services/          business operations; each opens its own database transaction.
                     search_clients.py / search_documents.py: find, score, highlight
  table/             database layer: all SQL queries, row types, migration runner
  indexing/          write path: text processing, embeddings, summaries, background worker
  search/            document-type taxonomy and the shared noisy-OR formula
  static/            web UI (plain HTML + JavaScript, no build step)
db/
  migrations/        database schema as numbered .sql files
  seed/              demo clients and documents
docs/                API reference (Markdown + OpenAPI spec)
scripts/             demo data loading, embedding-model comparison, OpenAPI export
tests/               unit, integration and search-quality tests
```

## Limitations and future work

- **English only:** the full-text configuration and the model are English.
- **Plain-text documents only:** there is no PDF or OCR ingestion.
- **Extractive summaries only.** An abstractive LLM summarizer could replace
  `ExtractiveSummarizer` on a bigger server.
- **The document-type taxonomy is hand-written.** It should grow from advisors' real document
  types.
- **Client descriptions are matched by words only:** "retired doctor" will not find "retired
  physician". The embedding pipeline could be reused for descriptions.
- **Search thresholds are calibrated for this model.** Changing the model means re-running the
  evaluation and reprocessing the documents.
- **Not implemented:** pagination beyond `limit`, per-advisor access control, delete endpoints.

## Deep dive

Details for reviewers reading the code.

**Model choice.** Five small models were compared on the demo data (one CPU thread,
[scripts/spike_embeddings.py](scripts/spike_embeddings.py)):

| Model | hit@1 | MRR | query ms | doc ms | +RAM |
|---|---|---|---|---|---|
| **all-MiniLM-L6-v2** (chosen) | **1.00** | **1.00** | **2.9** | **26** | 156 MB |
| bge-small-en-v1.5 | 0.91 | 0.93 | 13 | 148 | 202 MB |
| snowflake-arctic-embed-xs | 0.82 | 0.89 | 4.2 | 26 | 150 MB |
| granite-embedding-small-english-r2 | 0.55 | 0.71 | 14 | 159 | 275 MB |
| bge-base-en-v1.5 | 0.82 | 0.88 | 19 | 196 | 504 MB |

How to read the columns (higher is better for hit@1 and MRR, lower for the rest):
- **hit@1** is the share of test queries where the top result is a correct document. 0.91 means
  the first result was right for 91% of queries.
- **MRR** (mean reciprocal rank) also rewards correct answers that are not first. Each query
  scores 1 / *position of the first correct result*: 1.0 if it is 1st, 0.5 if 2nd, 0.33 if 3rd.
  MRR is the average over all queries, so 1.00 means a correct document was always first.
- **query ms** is the time to embed one query.
- **doc ms** is the time to embed one document.
- **+RAM** is the extra memory used once the model is loaded.

**Calibration.**
- **Semantic threshold, 0.22:** unrelated queries stay below 0.15 with this model.
- **Document-type threshold, 0.40:** short unrelated queries such as person names reach up to
  ~0.32 against short phrases.
- **Combining signals:** confidences are combined with a noisy-OR, `1 − Π(1 − cᵢ)`, so client and
  document scores share one [0, 1] scale.
- **Held-out queries:** they were written before any tuning (21 more were added before the
  comparison below). The two misses, "KYC identification" and "ESG exclusions", were not rescued
  by lowering a threshold into the noise band.

**Blended query vector.**
- When the query names a document type, the meaning search does not use the query vector as is.
  It uses `normalize(query + α · Σ similarityᵢ · centroidᵢ)` over the matched types, with α = 1.0.
- This replaced a separate vector search by the type's centroid, which needed its own relative
  cutoff. Now there is one vector search per query instead of one plus one per matched type.
- On all 42 eval queries:

  | | Expected document in top 3 | Expected document first | Avg score of the best expected document |
  |---|---|---|---|
  | separate centroid search (before) | 40/42 | 39/42 | 0.76 |
  | blended vector (now) | 40/42 | 40/42 | 0.79 |
- **The trade-off:**
  - documents found *only* through their type score lower: `risk appetite` → the risk
    questionnaire at 0.35 (it was 0.71 as a separate signal);
  - their `matched_on` says `semantic` rather than `document_type:…`;
  - a wrongly matched type pulls harder: `retirement planning` touches *estate planning*, so the
    will comes second (0.49).
- **Why α = 1.0:** α = 0.5 finds the same documents with lower scores. At α = 2.0 the wrong type
  starts to win.

**Plain SQL instead of an ORM.**
- The schema is [SQL](db/migrations/0001_initial.sql).
- A small runner, [app/table/migrate.py](app/table/migrate.py), applies new files in order under
  an advisory lock.
- All queries live in [app/table/](app/table) on top of psycopg 3. Most of the work is SQL anyway
  (full-text, trigrams, vectors, `SKIP LOCKED`).

**Background worker details.**
- A claim stamps `locked_at`, and results are saved only while that stamp is unchanged, so a
  reprocess request wins over a stale run.
- Documents stuck in `processing` after a crash are picked up again after 5 minutes.
- On one vCPU the model runs with a single thread, so indexing does not starve search requests.

**Embedding batch size.** fastembed embeds 256 texts per batch by default. ONNX memory grows with
batch size × sequence length², so a 4,500-word document (53 chunks, 258 summary sentences) peaked
at 1.17 GB and the 1 GB container was OOM-killed. Measured peak memory of the whole process:

| Batch | That document | Worst case (300 inputs at max length) |
|---|---|---|
| 256 | 1169 MB | - |
| 64 | 492 MB | 1174 MB |
| 32 | - | 688 MB |
| 16 | 294 MB | - |

The default is 32 (`EMBEDDING_BATCH_SIZE` overrides it). Smaller batches are also faster on one
CPU: each batch is padded to its longest text, so short sentences waste less work (13 s at 16 vs
29 s at 256 for that document).

## Use of LLMs

As the assignment encourages, this was built with an AI coding assistant (Claude Code), which
helped with design discussion, the embedding-model comparison, implementation, tests and docs.
The decisions above are backed by measurements in the repository: the comparison script and the
search-quality tests.
