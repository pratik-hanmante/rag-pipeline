> 📘 **Interactive version:** open `rag-guide.html` in a browser, or view it online at https://claude.ai/artifact/QTHmf8YrvcaoDbySJicB4v

# RAG Pipeline: Package Notes

## What is a RAG pipeline?

**RAG = Retrieval-Augmented Generation.** Instead of asking an LLM a question cold, you:

1. **Ingest:** take documents (PDFs, text, and so on), split them into small **chunks**, and turn each chunk into an **embedding**. An embedding is a list of numbers that captures the chunk's meaning.
2. **Store:** save the chunks and their embeddings in a database (here, **Postgres + pgvector**).
3. **Retrieve:** when a user asks a question, embed the question and find the chunks whose embeddings are closest to it (vector similarity search).
4. **Generate:** send the question plus those retrieved chunks to an LLM. The LLM answers using *your* data rather than only what it learned in training.

```
                    ┌──────────── INGEST (background, Celery) ────────────┐
 upload file ──► FastAPI ──► Redis queue ──► Celery worker: chunk ──► embed ──► Postgres/pgvector
                                                                                     │
 user question ──► FastAPI ──► embed question ──► similarity search ◄────────────────┘
                                                        │
                                       top-k chunks + question ──► LLM (OpenAI) ──► answer
```

---

## The packages, grouped by job

### 1. Web API layer

| Package | What it does | Role in this project |
|---|---|---|
| **fastapi** | A modern Python web framework for building APIs. It uses type hints to validate requests automatically and generates interactive docs at `/docs`. | Defines your endpoints, e.g. `POST /documents` (upload) and `POST /query` (ask a question). |
| **uvicorn[standard]** | An ASGI **server**, the program that actually runs your FastAPI app and listens on a port. `[standard]` adds faster extras (uvloop, httptools, websockets, auto-reload). | You start the app with `uvicorn app.main:app --reload`. |
| **python-multipart** | Parses `multipart/form-data` requests, which is how browsers send file uploads. | FastAPI **requires** it for `UploadFile` / `Form(...)`. Without it, file-upload endpoints raise an error. |

### 2. Database layer (Postgres + vectors)

| Package | What it does | Role in this project |
|---|---|---|
| **sqlalchemy[asyncio]** | Python's main ORM/SQL toolkit. It lets you define tables as Python classes and build queries in Python. `[asyncio]` enables `async`/`await` usage. | Defines tables such as `documents` and `chunks`, and runs the queries. |
| **asyncpg** | A very fast **async** PostgreSQL driver: the low-level code that talks to Postgres over the network. | SQLAlchemy uses it under the hood. Connection URL looks like `postgresql+asyncpg://rag_user:rag_password@localhost:5432/rag_db`. |
| **pgvector** | Python helpers for the **pgvector** Postgres extension. It adds a `Vector` column type for SQLAlchemy and distance operators. | Stores embeddings (e.g. `Vector(1536)`) and lets you run "find the nearest chunks" queries using cosine distance (`<=>`). |

> The `docker-compose.yml` runs the `pgvector/pgvector:pg16` image, which is Postgres 16 with the vector extension preinstalled. You still need to run `CREATE EXTENSION IF NOT EXISTS vector;` once in the database.

### 3. Data validation and config

| Package | What it does | Role in this project |
|---|---|---|
| **pydantic** | Validates data and converts types using Python type hints. If the data doesn't match a model, Pydantic raises a clear error. | Request/response schemas, e.g. `QueryRequest(question: str, top_k: int = 5)`. FastAPI is built on it. |
| **pydantic-settings** | Loads configuration from **environment variables / `.env` files** into a typed Pydantic class. | A `Settings` class holding `DATABASE_URL`, `REDIS_URL`, `OPENAI_API_KEY`, `CHUNK_SIZE`, and so on. This keeps secrets out of your code. |

### 4. AI / LLM layer

| Package | What it does | Role in this project |
|---|---|---|
| **openai** | Official OpenAI SDK. Calls chat models (e.g. GPT) and embedding models (e.g. `text-embedding-3-small`). | (a) Create embeddings for chunks and questions. (b) Generate the final answer from the retrieved context. |
| **sentence-transformers** | Runs embedding models **locally** on your machine (e.g. `all-MiniLM-L6-v2`), so no API key or per-call cost. Also supports **cross-encoder rerankers**. | Either a free/offline alternative to OpenAI embeddings, or a **reranker** that re-scores the top results for better accuracy. It pulls in PyTorch, which is a large install. |
| **tiktoken** | OpenAI's fast **tokenizer**. Counts how many tokens a piece of text uses. | Splits documents into chunks by *token* count (e.g. 500 tokens with 50 overlap) and makes sure prompts fit inside the model's context limit. |

> ⚠️ Embedding dimensions must match your DB column. OpenAI `text-embedding-3-small` = **1536** dims; `all-MiniLM-L6-v2` = **384** dims. Don't mix them in the same column.

### 5. Background jobs (async ingestion)

Embedding a big PDF can take a long time, so you don't want the HTTP request to sit waiting for it. Instead, the request hands the work to a background worker.

| Package | What it does | Role in this project |
|---|---|---|
| **celery** | A distributed **task queue**. You define functions as tasks, and separate worker processes run them in the background. | `process_document.delay(doc_id)`: the worker chunks, embeds and stores the document while the API responds immediately. |
| **redis** | Python client for **Redis**, an in-memory key-value store. | (a) **Broker** for Celery, the queue that tasks pass through. (b) Optional **cache** for repeated queries or embeddings. Redis itself runs in Docker (`redis:7`). |
| **flower** | A web dashboard for watching Celery: active/failed tasks, workers, retries. | Run `celery -A app.worker flower` and open `http://localhost:5555`. |

### 6. Networking and reliability

| Package | What it does | Role in this project |
|---|---|---|
| **httpx** | A modern HTTP client (sync **and** async), similar to `requests`. | The OpenAI SDK uses it internally. It is also used in tests to call your FastAPI app (`httpx.AsyncClient`) and for any other external HTTP calls. |
| **tenacity** | **Retry** library. Wraps a function so that if it fails, it retries with backoff. | Retries OpenAI calls when you hit rate limits (429) or timeouts. Example: `@retry(stop=stop_after_attempt(5), wait=wait_exponential())`. |

### 7. Testing

| Package | What it does | Role in this project |
|---|---|---|
| **pytest** | The standard Python testing framework. Write `test_*.py` files and run `pytest`. | Unit tests for chunking, retrieval and the API endpoints. |
| **pytest-asyncio** | Lets pytest run `async def` test functions. | Needed because the DB layer and API are async. Mark tests with `@pytest.mark.asyncio`. |

---

## How they fit together (one request's journey)

**Uploading a document**
1. `uvicorn` receives the request, and `fastapi` + `python-multipart` read the uploaded file.
2. `sqlalchemy` + `asyncpg` save a `documents` row with status `pending`.
3. `celery` queues a task through `redis` and returns `202 Accepted` right away.
4. A Celery worker loads the file and splits it into chunks with `tiktoken`.
5. Each chunk is embedded with `openai` (wrapped in `tenacity` retries) or `sentence-transformers`.
6. The chunks and embeddings go into Postgres using the `pgvector` column type.
7. You can watch progress in `flower`.

**Asking a question**
1. `fastapi` validates the body with a `pydantic` model.
2. The question is embedded the same way as the chunks.
3. A pgvector similarity query (`ORDER BY embedding <=> :q LIMIT k`) returns the closest chunks.
4. (Optional) A `sentence-transformers` cross-encoder reranks them.
5. `openai` chat completion gets a prompt like "Answer using only this context: …".
6. The answer and its sources are returned as JSON.

---

## Setup cheat-sheet

```zsh
# 1. Start Postgres + Redis
docker compose up -d

# 2. Activate the virtualenv (folder is "venv", not ".venv")
conda deactivate          # optional: avoid conda/venv mixing
source venv/bin/activate

# 3. Install packages
pip install -r requirements.txt

# 4. Run the API
uvicorn app.main:app --reload            # http://localhost:8000/docs

# 5. Run the background worker (separate terminal)
celery -A app.worker worker --loglevel=info

# 6. Celery dashboard (optional)
celery -A app.worker flower              # http://localhost:5555

# 7. Tests
pytest
```

Example `.env` (read by pydantic-settings):
```
DATABASE_URL=postgresql+asyncpg://rag_user:rag_password@localhost:5432/rag_db
REDIS_URL=redis://localhost:6379/0
OPENAI_API_KEY=sk-...
```

---

## ⚠️ Heads-up: Python version

The `venv` now uses **Python 3.11** (taken from the `curio` conda env). On Python 3.14 the install failed: `pydantic-core` and `tiktoken` couldn't be built because their build tool (PyO3) only supports up to Python 3.13. If you ever rebuild the venv, use Python 3.11 or 3.12:

```zsh
rm -rf venv
python3.12 -m venv venv      # or: conda create -n rag python=3.12
source venv/bin/activate
pip install -r requirements.txt
```
