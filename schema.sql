-- schema.sql
-- =====================================================================
-- RAG PIPELINE DATABASE SCHEMA
-- =====================================================================
-- RAG = Retrieval-Augmented Generation:
--   "Before asking the LLM, look up the relevant bits of OUR documents and
--    paste them into the prompt, so it answers from our data, not guesses."
--
-- Two tables, two levels:
--
--   documents  = one row per ORIGINAL source (a PDF, a web page...).
--       │        Bookkeeping only: where it came from, who owns it.
--       │ 1 document ──► many chunks
--       ▼
--   chunks     = the document cut into small pieces (a few hundred tokens)
--                + an EMBEDDING per piece: a list of 1536 numbers that
--                captures what the text MEANS. Similar meaning = nearby vectors.
--
-- At question time:
--   question ──► embed it ──► find closest chunk embeddings (vector search)
--            ──► + keyword search ──► send best chunks + question to LLM.
-- =====================================================================

-- pgvector: adds the `vector` column type and distance operators
--   <=> cosine distance   <-> euclidean distance   <#> negative inner product
-- Our docker image (pgvector/pgvector:pg16) ships it; this switches it on.
CREATE EXTENSION IF NOT EXISTS vector;


-- ---------------------------------------------------------------------
-- TABLE: documents: one row per source file/page
-- ---------------------------------------------------------------------
CREATE TABLE documents (
    -- Unique ID, generated automatically by Postgres.
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    -- Owner (customer / workspace / user). "Multi-tenancy": many customers
    -- share one DB, so every query must filter by tenant_id to keep data private.
    tenant_id       UUID NOT NULL,
    -- Where the doc came from (URL, s3://..., file path). Used to cite sources
    -- in answers ("according to handbook.pdf...").
    source_uri      TEXT NOT NULL,
    -- Optional human-readable name.
    title           TEXT,
    -- Fingerprint of the content (e.g. SHA-256 hash). If the same file is
    -- uploaded again, the hash matches and we skip re-processing it.
    content_hash    TEXT NOT NULL,
    -- Free-form extra info as JSON: author, file type, tags, page count...
    -- JSONB = binary JSON; queryable, e.g. WHERE metadata->>'type' = 'pdf'.
    metadata        JSONB DEFAULT '{}'::jsonb,
    -- TIMESTAMPTZ = timestamp with time zone. now() fills it in automatically.
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- Note: Postgres does NOT update this by itself; set it in your code
    -- (or add a trigger) whenever the row changes.
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- UNIQUE index on (tenant_id, content_hash): the same tenant can't store the
-- exact same document twice → automatic de-duplication. Different tenants
-- CAN upload the same file (each gets their own copy).
CREATE UNIQUE INDEX idx_documents_tenant_hash ON documents (tenant_id, content_hash);


-- ---------------------------------------------------------------------
-- TABLE: chunks: the heart of RAG; this is what gets SEARCHED
-- ---------------------------------------------------------------------
-- Why chunk? A whole document is too big for the LLM's context window and
-- mostly irrelevant to any one question. We fetch just the 3-10 pieces
-- that actually answer it.
CREATE TABLE chunks (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    -- Which document this piece came from.
    -- REFERENCES = foreign key: must point to a real document.
    -- ON DELETE CASCADE: delete a document → all its chunks are deleted too.
    document_id     UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    -- Copied from documents on purpose ("denormalized") so searches can
    -- filter by tenant directly, without a slower JOIN to documents.
    tenant_id       UUID NOT NULL,
    -- Position of this chunk in its document (0, 1, 2...). Lets you rebuild
    -- order or grab neighbouring chunks for extra context.
    chunk_index     INT NOT NULL,
    -- The actual text, which gets pasted into the LLM prompt.
    content         TEXT NOT NULL,
    -- Number of tokens in `content` (counted with tiktoken). Used to pack as
    -- many chunks as fit into the LLM's context limit without overflowing.
    token_count     INT NOT NULL,
    -- THE EMBEDDING: 1536 numbers = size of OpenAI text-embedding-3-small.
    -- (Using local sentence-transformers all-MiniLM-L6-v2? That's 384.)
    -- Nullable on purpose: the chunk row can be saved first and the
    -- embedding filled in later by the background (Celery) worker.
    embedding       vector(1536),
    -- Per-chunk extras: page number, section heading, etc.
    metadata        JSONB DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Fast "all chunks of document X" (re-processing, deleting, showing sources).
CREATE INDEX idx_chunks_document_id ON chunks (document_id);
-- Fast "only this tenant's chunks", used by every search.
CREATE INDEX idx_chunks_tenant_id ON chunks (tenant_id);


-- ---------------------------------------------------------------------
-- KEYWORD SEARCH (full-text search): half of "hybrid search"
-- ---------------------------------------------------------------------
-- Vector search finds MEANING ("car" matches "automobile") but can miss
-- exact terms like error codes, product names, IDs ("ERR-4021").
-- Keyword search catches those. Using both = hybrid search = better results.
--
-- tsvector = the text pre-processed for searching: lowercased, stop words
-- ("the", "is") removed, words reduced to roots ("running" → "run").
-- GENERATED ALWAYS ... STORED = Postgres computes and saves it automatically
-- whenever `content` is inserted/changed. You never write to it yourself.
ALTER TABLE chunks ADD COLUMN content_tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;
-- GIN = "inverted index" (word → list of rows containing it), like the index
-- at the back of a book. Makes keyword search fast.
-- Query example: WHERE content_tsv @@ plainto_tsquery('english', 'refund policy')
CREATE INDEX idx_chunks_content_tsv ON chunks USING GIN (content_tsv);


-- ---------------------------------------------------------------------
-- VECTOR SEARCH INDEX: the other half of hybrid search
-- ---------------------------------------------------------------------
-- Without an index, finding the nearest embeddings means comparing the
-- question against EVERY chunk (slow at millions of rows).
-- HNSW (Hierarchical Navigable Small World) builds a graph of "neighbour"
-- vectors so search hops to the closest ones quickly. It's APPROXIMATE:
-- very fast, and finds nearly all of the true nearest matches (recall ~95-99%).
-- vector_cosine_ops: compare by cosine distance (<=>), the standard choice
--   for text embeddings. Your queries must use <=> to use this index.
-- m = 16: links per node in the graph (higher = more accurate, more memory).
-- ef_construction = 64: how thoroughly the graph is built (higher = better
--   quality, slower index build). Both values are pgvector's defaults.
CREATE INDEX idx_chunks_embedding_hnsw ON chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- ---------------------------------------------------------------------
-- Example retrieval query (top 5 chunks by meaning, for one tenant):
--   SELECT content, 1 - (embedding <=> :question_embedding) AS similarity
--   FROM chunks
--   WHERE tenant_id = :tenant_id
--   ORDER BY embedding <=> :question_embedding
--   LIMIT 5;
-- ---------------------------------------------------------------------
