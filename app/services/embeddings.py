# embeddings.py
# ---------------------------------------------------------------------------
# Turns text into EMBEDDINGS: lists of numbers that capture what the text means.
#
# Where this fits in the RAG pipeline:
#   chunking.py  →  chunks (text)  →  embeddings.py  →  vectors  →  chunks.embedding column
#
#   - At ingest time: embed every chunk and store the vector in Postgres.
#   - At query time: embed the user's question the SAME way, then use pgvector
#     to find the chunks whose vectors are closest (ORDER BY embedding <=> :q).
#
# Important: chunks and questions MUST be embedded with the same model and the
# same dimension, or the numbers can't be compared.
#
# This file uses Google's Gemini embedding API (not OpenAI).
# ---------------------------------------------------------------------------

import asyncio  # Python's built-in library for running many async tasks at once

# Google's official Gemini SDK.
# ⚠️ NOT INSTALLED: "google-genai" is missing from requirements.txt and from the venv.
#    Fix: add `google-genai` to requirements.txt, then run `pip install google-genai`.
from google import genai
from google.genai import types  # typed config objects, e.g. EmbedContentConfig

# tenacity = automatic retries. API calls fail sometimes (rate limits, timeouts),
# so we retry instead of crashing the whole ingestion job.
from tenacity import retry, stop_after_attempt, wait_exponential

# Our settings, loaded from .env by pydantic-settings (see app/config.py).
from app.config import settings

# One shared API client, created once when the module is imported.
# ⚠️ BUG: config.py defines `genai_api_key` (matching GENAI_API_KEY in .env),
#    but this line reads `gemini_api_key`, which doesn't exist. Importing this
#    file raises AttributeError. Fix: use settings.genai_api_key (or rename
#    both the config field and the .env key to GEMINI_API_KEY).
client = genai.Client(api_key=settings.gemini_api_key)

# Which embedding model to use. Changing it later means re-embedding ALL chunks,
# because vectors from different models aren't comparable.
EMBEDDING_MODEL = "gemini-embedding-2"

# How many numbers each embedding has. Gemini can return shorter vectors on
# request (output_dimensionality below). This MUST match `vector(1536)` in
# schema.sql, or Postgres rejects the insert.
EMBEDDING_DIM = 1536

# Max texts sent in one API request. Batching = far fewer HTTP calls:
# 3,000 chunks → 30 requests instead of 3,000.
MAX_BATCH_SIZE = 100


# Embeds ONE batch (up to 100 texts) in a single API call.
# The leading underscore means "private helper": call embed_texts() instead.
#
# @retry: if the call raises an error, try again:
#   stop_after_attempt(5)      → give up after 5 tries in total
#   wait_exponential(...)      → wait 1s, 2s, 4s, 8s… (max 30s) between tries,
#                                which gives a rate-limited API time to recover
#   reraise=True               → if all 5 fail, raise the ORIGINAL error
#                                (clearer than tenacity's RetryError wrapper)
@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=30), reraise=True)
async def _embed_batch(texts: list[str]) -> list[list[float]]:
    # client.aio = the async version of the client. `await` lets Python run
    # other batches while this one waits on the network.
    resp = await client.aio.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,  # a list of strings → one embedding per string, same order
        config=types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIM),
        # 💡 Optional improvement: Gemini can tune embeddings for search with
        #    task_type="RETRIEVAL_DOCUMENT" for chunks and "RETRIEVAL_QUERY" for
        #    questions. That usually gives better matches.
    )
    # resp.embeddings is a list of objects; .values is the actual list of floats.
    return [item.values for item in resp.embeddings]


# PUBLIC function: embed any number of texts, fast but polite to the API.
#
#   texts:       e.g. all chunks of a document
#   concurrency: max number of batches in flight at the same time (default 5)
#
# Returns one embedding per input text, in the SAME order as `texts`, so
# result[i] belongs to texts[i] (and to chunk_index i).
async def embed_texts(texts: list[str], concurrency: int = 5) -> list[list[float]]:
    # Cut the list into batches of 100:  [0:100], [100:200], ...
    batches = [texts[i:i + MAX_BATCH_SIZE] for i in range(0, len(texts), MAX_BATCH_SIZE)]

    # A semaphore is a counter that limits how many tasks run at once.
    # Without it, 30 batches would hit the API at the same moment and trigger
    # rate limits. With it, at most `concurrency` batches run at a time.
    semaphore = asyncio.Semaphore(concurrency)

    async def _run(batch):
        async with semaphore:          # wait for a free slot…
            return await _embed_batch(batch)  # …then embed (with retries)

    # Start all batches; gather waits for all of them and returns the results
    # in the SAME order the tasks were given (not the order they finished).
    results = await asyncio.gather(*[_run(b) for b in batches])

    # Flatten [[e1, e2, ...], [e101, ...], ...] → [e1, e2, ..., e101, ...]
    return [emb for batch_result in results for emb in batch_result]
