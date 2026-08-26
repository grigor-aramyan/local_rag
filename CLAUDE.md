# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project state

Pre-implementation scaffold. `docs/brief.md` is the authoritative spec — read it
before starting any work; it contains the build order, the architectural
constraints below, and an open-questions section whose answers are not yet
recorded anywhere. If a decision from that section is needed, ask rather than
assume.

Present: `Dockerfile`, `docker-compose.yml`, a stub `main.py`, `docs/brief.md`.
Absent: `requirements.txt`, the `app/` package, `sample_docs/`, tests, any commit.

## Layout

Application code belongs in an `app/` package (`app/main.py` exposes `app`).
The root-level `main.py` is a leftover lifespan stub — fold it into `app/main.py`
rather than growing it in place.

Two known Dockerfile bugs to fix when the package lands: `COPY app/ /app` flattens
the package so `uvicorn app.main:app` cannot import it (should be `COPY app/ /app/app`
with `WORKDIR /app`, or equivalent), and `requirements.txt` is copied but does not exist.

## Commands

```bash
docker compose up --build          # build image (downloads models in stage 1) and run
docker compose up                  # run against the existing image
docker compose logs -f rag
curl localhost:8000/health         # must confirm models loaded, not just liveness

pytest                             # full suite
pytest tests/test_chunking.py::test_overlap -x    # single test
ruff check . && ruff format .
```

`OPENAI_API_KEY` must be set in the environment (or `.env`) before compose starts;
`OPENAI_BASE_URL` defaults to the OpenAI endpoint and exists so a local
OpenAI-compatible server can be swapped in.

Source documents are read from the read-only bind mount `./documents` →
`/data/documents`. LanceDB lives on the named volume at `/data/lancedb`; deleting
that volume discards the index and forces a full re-ingest.

## Architecture constraints

These are load-bearing — violating them produces failures that are not obvious
from a local test run.

**One process, one writer.** LanceDB uses optimistic concurrency over a manifest;
two writer processes on the same table produce commit conflicts. uvicorn stays at
a single worker. Anything that writes must run in-process.

**No PyTorch.** Inference goes through `fastembed` / ONNX Runtime only. Pulling in
torch (directly or as a transitive dependency of a reranker/extractor library)
adds 3–4 GB to a ~1 GB image target. Check new dependencies for it.

**ONNX inference is CPU-bound and synchronous.** Call embedder and reranker via
`run_in_threadpool` — never directly inside an `async def` handler, or the event
loop stalls for every concurrent request.

**Models load once, at startup.** Embedder, reranker, and the LanceDB connection
are constructed in the `lifespan` handler and held on `app.state`. The image bakes
weights into `/models` with `HF_HUB_OFFLINE=1`; a code path that triggers a download
at runtime will hang or fail in a container with no network.

**Docker layer order.** The model layer must precede the app-code layer, or every
code edit re-downloads several hundred MB of weights.

**Config-compatibility guard.** Embedding model name, dimension, and chunking params
are written to a metadata row on first ingest and verified at startup. Divergence
must refuse to serve with an explicit error — a changed embedder against an old index
degrades retrieval silently rather than crashing.

## Pipelines

**Ingest** — `POST /ingest` returns a `job_id` immediately and runs the work as a
background task; `GET /jobs/{id}` reports progress. Job state persists to SQLite on
the LanceDB volume so a restart does not strand jobs in `running`. The pipeline is
extract → chunk → embed → write → build FTS index → `table.optimize()`.

Chunk IDs key on content hash, which is what makes re-ingestion idempotent.

**Query** — embed → hybrid retrieve (vector + FTS) `top_k≈50` → rerank to ~5 →
numbered context blocks in the prompt → stream from the LLM. The model emits `[n]`
markers that are mapped back to `source`/`page` in the response payload.

LanceDB table columns: `id`, `text`, `vector`, `source`, `page`, `content_hash`,
`ingested_at`. Raw chunk text is stored, not just vectors — prompts and citations
need it.

**Indexes.** Create the FTS index at ingest time; retrofitting one requires a full
rebuild. Skip the ANN index below ~100k vectors (brute force is faster there);
create IVF-PQ above that threshold.

## Tuning defaults

Chunk size 500, overlap 50, `top_k` 50, rerank depth 5. Per the brief these are the
highest-leverage knobs in the system; change them against the eval harness, not by
intuition. The eval harness is a separate tool from the service and measures
retrieval (recall@k) and generation (groundedness) independently — most "RAG is
broken" reports turn out to be retrieval failures.
