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
background task through `run_ingestion`; failures surface on `GET /jobs/{id}`,
never on this response, which has already returned by then.

## GET /jobs/{id}

Reports progress from the SQLite job store. Jobs are durable across restarts;
interrupted ones are closed out at startup, so a `running` job in the store
always belongs to the current process.

## POST /query

Build step 10. Embed → hybrid retrieve (vector + FTS) `top_k` → rerank to
`rerank_top_n` (toggleable via `rerank_enabled`) → generate, grounded via native
Anthropic citations rather than prompted `[n]` markers. Retrieval and
generation details are in `app/services/CLAUDE.md`.

Nothing relevant retrieved is not an error (brief decision #12): the response
still comes back 200, generated from the model's own knowledge, with
`QueryResponse.warning` set so the caller can tell the two cases apart.

The handler maps generation failures to HTTP status rather than letting them
surface as bare 500s: `anthropic.RateLimitError` → 429,
`APIConnectionError`/`APIStatusError` → 502. A missing Anthropic credential is a
special case — the SDK (`anthropic==1.1.0`) raises a bare `TypeError` for it,
not a typed exception, matched by message and turned into a 503; any other
`TypeError` is re-raised rather than swallowed as a config error.

## GET /health

200 when the resources are ready, 503 degraded otherwise. It reads
`Resources.ready`, so a partially constructed startup must surface here rather
than failing on first query. An index built under incompatible settings is
degraded for the same reason — everything loaded, but a query would return
quietly wrong results — and the reason comes back in `detail`.

`/health` takes `ResourcesDep`, deliberately not `ReadyResourcesDep`: it has to
be able to *describe* a degraded service rather than be blocked by it. Handlers
that touch the index take `ReadyResourcesDep`, which 503s with the same message.
`detail` is omitted when null (`response_model_exclude_none`), so a healthy
response stays the four documented fields.
