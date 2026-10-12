-- Initial schema: clients, documents, document chunks and the search indexes.

CREATE EXTENSION IF NOT EXISTS vector;   -- pgvector: embedding column + HNSW index
CREATE EXTENSION IF NOT EXISTS pg_trgm;  -- trigram index for substring / typo-tolerant search

-- OR-combines tsqueries, e.g. all phrases of a document type in one query:
--   SELECT tsquery_or_agg(phraseto_tsquery('english', p)) FROM unnest(ARRAY['utility bill', ...]) p
CREATE AGGREGATE tsquery_or_agg(tsquery) (SFUNC = tsquery_or, STYPE = tsquery);


CREATE TABLE clients (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    first_name      varchar(100) NOT NULL,
    last_name       varchar(100) NOT NULL,
    email           varchar(320) NOT NULL,
    description     text,
    social_links    text[] NOT NULL DEFAULT '{}',
    -- Normalized identity terms (names, email parts, social handles), built by the app on write.
    search_text     text NOT NULL,
    description_tsv tsvector NOT NULL
                    GENERATED ALWAYS AS (to_tsvector('english', coalesce(description, ''))) STORED,
    created_at      timestamptz NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX uq_clients_email_lower ON clients (lower(email));
CREATE INDEX ix_clients_search_text_trgm ON clients USING gin (search_text gin_trgm_ops);
CREATE INDEX ix_clients_description_tsv ON clients USING gin (description_tsv);


CREATE TABLE documents (
    id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    client_id     uuid NOT NULL REFERENCES clients (id) ON DELETE CASCADE,
    title         varchar(300) NOT NULL,
    content       text NOT NULL,
    created_at    timestamptz NOT NULL DEFAULT now(),

    -- Indexing state, maintained by the background worker (the table doubles as its job queue).
    status        varchar(16) NOT NULL DEFAULT 'pending'
                  CONSTRAINT ck_documents_status
                  CHECK (status IN ('pending', 'processing', 'ready', 'failed')),
    summary       text,
    attempts      integer NOT NULL DEFAULT 0,
    error         text,
    locked_at     timestamptz,
    processed_at  timestamptz,
    processing_ms integer,

    -- Source of the inverted index, kept in sync by Postgres: title (A) weighs more than content (B).
    tsv           tsvector NOT NULL GENERATED ALWAYS AS (
                      setweight(to_tsvector('english', coalesce(title, '')), 'A') ||
                      setweight(to_tsvector('english', coalesce(content, '')), 'B')
                  ) STORED
);

CREATE INDEX ix_documents_client_id ON documents (client_id);
CREATE INDEX ix_documents_tsv ON documents USING gin (tsv);
CREATE INDEX ix_documents_status_created ON documents (status, created_at);


CREATE TABLE document_chunks (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id uuid NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    ordinal     integer NOT NULL,
    text        text NOT NULL,
    char_start  integer NOT NULL,  -- offsets of the chunk inside documents.content
    char_end    integer NOT NULL,
    -- 384 = output size of all-MiniLM-L6-v2 (app.table.records.EMBEDDING_DIM).
    embedding   vector(384) NOT NULL
);

CREATE INDEX ix_document_chunks_document_id ON document_chunks (document_id);
CREATE INDEX ix_document_chunks_embedding_hnsw ON document_chunks
    USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
