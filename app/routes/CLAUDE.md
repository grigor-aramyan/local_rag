# app/routes/

Endpoint contracts. Handlers stay thin — they validate, delegate, and shape the
response; the work lives in `app/services/` (see `app/services/CLAUDE.md`).

Handlers are `async def`, so any CPU-bound call (the embedder, the reranker)
must go through `run_in_threadpool` rather than being awaited inline. Resources
come from `app/routes/deps.py`, never from module-level construction.

## POST /ingest

Upload → sanitize/stage/commit → create job → return `job_id` with 202. Returns
as soon as the files are on disk; extraction and embedding are far too slow to
hold a request open. The upload path is security-critical — read the untrusted
input section in `app/CLAUDE.md` before touching it. Ingestion itself runs as a
background task through `run_ingestion`.

## GET /jobs/{id}

Reports progress from the SQLite job store. Jobs are durable across restarts;
interrupted ones are closed out at startup, so a `running` job in the store
always belongs to the current process.

## POST /query

Currently returns 501 (build step 10). The pipeline it will front is embed →
hybrid retrieve (vector + FTS) `top_k≈50` → rerank to ~5 → numbered context
blocks → stream from Claude. Retrieval and generation details are in
`app/services/CLAUDE.md`.

## GET /health

200 when the resources are ready, 503 degraded otherwise. It reads
`Resources.ready`, so a partially constructed startup must surface here rather
than failing on first query.
