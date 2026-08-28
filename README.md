# local_rag

A self-contained Retrieval-Augmented Generation (RAG) service: upload documents,
they get chunked and embedded locally, and `/query` answers questions against
them with citations grounded in the retrieved text. Embedding and reranking run
on-box via ONNX Runtime; the only network call is generation, via the Anthropic
API.

## Pipelines

**Ingest** — `POST /ingest` accepts one or more files, sanitizes and stores
them, and returns a `job_id` immediately (202). A background task then
extracts text, chunks it, embeds each chunk, writes the vectors into LanceDB,
and builds the full-text index. Progress and failures are visible on
`GET /jobs/{job_id}`.

**Query** — `POST /query` embeds the question, hybrid-retrieves candidates
(vector search + full-text search, combined by LanceDB's own reranker),
optionally reranks the top candidates with a local cross-encoder, and asks
Claude to answer using only the retrieved chunks. Citations are native
Anthropic citations (grounded in exact spans of the source text), not
prompted `[n]` markers. If nothing relevant is retrieved, the response still
comes back 200 with an answer from the model's general knowledge and a
`warning` field flagging that it isn't grounded in your documents.

## Libraries, packages, and models

1. **FastAPI** (`fastapi`) — the HTTP framework serving `/ingest`, `/jobs`,
   `/query`, and `/health`.
2. **Uvicorn** (`uvicorn[standard]`) — the ASGI server that runs the app, pinned
   to a single worker (LanceDB's optimistic-concurrency writes don't tolerate a
   second writer process on the same table).
3. **Pydantic / pydantic-settings** — request/response schemas and environment-driven
   configuration (`app/config.py`).
4. **fastembed** (ONNX Runtime, no PyTorch) — runs both local models used in
   the pipeline:
   - **Embedding model: `BAAI/bge-small-en-v1.5`** (384-dim vectors) — turns
     each chunk and each incoming question into a vector for similarity search.
   - **Reranking model: `Xenova/ms-marco-MiniLM-L-6-v2`** (cross-encoder) —
     re-scores the top retrieved candidates against the question for a more
     precise final ordering before generation; toggleable via `RERANK_ENABLED`.
5. **LanceDB** (`lancedb`) — the vector store. Stores chunk text, vectors, and
   metadata (source, page, content hash, ACL) in an embedded, file-backed
   table; also holds the full-text (BM25) index used for hybrid retrieval.
6. **Anthropic SDK** (`anthropic`) — the only component that leaves the
   container. Sends the retrieved chunks as cited `document` content blocks
   and generates the final grounded answer (default model `claude-opus-5`).
7. **SQLite** (via the standard library) — the job store backing
   `GET /jobs/{job_id}`, durable across restarts.

## Running it

Everything runs in Docker; nothing needs to be installed on the host.

```bash
cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY=sk-ant-...

docker compose up --build
```

The service listens on **`http://localhost:8000`**.

- `GET /health` — 200 `ok` once the embedder, reranker, and database are
  loaded; 503 `degraded` otherwise.
- `POST /ingest` — upload one or more documents (`multipart/form-data`,
  field name `files`). Returns `202` with a `job_id`.
- `GET /jobs/{job_id}` — poll ingestion progress/state.
- `POST /query` — ask a question (`application/json`, field `question`).
  Returns the answer plus citations.

### 1. Check the service is healthy

```bash
curl http://localhost:8000/health
```

### 2. Ingest the bundled sample document

A ~10-page excerpt of *Pride and Prejudice* is bundled at
`sample_docs/pride_and_prejudice_excerpt.txt` for exactly this purpose.

```bash
curl -X POST http://localhost:8000/ingest \
  -F "files=@sample_docs/pride_and_prejudice_excerpt.txt"
```

This returns something like:

```json
{"job_id": "b3f1...", "documents": ["pride_and_prejudice_excerpt.txt"]}
```

Poll until it finishes:

```bash
curl http://localhost:8000/jobs/b3f1...
```

### 3. Query it

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Why does Mrs. Bennet want Mr. Bennet to visit Mr. Bingley?"}'
```

Example response shape:

```json
{
  "answer": "Because Mr. Bingley is a wealthy single man who has just moved into the neighbourhood, and Mrs. Bennet hopes he will marry one of her daughters [1].",
  "citations": [
    {"marker": 1, "source": "pride_and_prejudice_excerpt.txt", "page": null}
  ],
  "warning": null
}
```
