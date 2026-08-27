# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project state

Scaffold, not a working RAG system yet. `docs/brief.md` is the authoritative
spec — read it before starting work; it holds the build order and an
open-questions section whose answers are not recorded anywhere. If a decision
from that section is needed, ask rather than assume.

Built: config, upload handling, the SQLite job store, `/health`, the route
surface, the ingestion pipeline (steps 7–9), and the config-compatibility guard
(step 6). Stubbed: `POST /query` (step 10, returns 501). An upload is stored,
indexed, and searchable by both vector and FTS — but nothing queries it yet.

## Where the rest lives

Module-scoped guidance sits next to the code it governs; read the file for the
area you are touching.

- `app/CLAUDE.md` — untrusted uploads, `paths.py`, the config-compatibility
  guard, jobs and storage
- `app/routes/CLAUDE.md` — endpoint contracts for `/ingest`, `/query`, `/jobs`,
  `/health`
- `app/services/CLAUDE.md` — the fastembed traps, LanceDB schema and indexes,
  the ingest and query pipelines, generation against the Anthropic API
- `tests/CLAUDE.md` — tests as executable spec, the `make_client` factory

## Commands

Everything runs in Docker; nothing is installed on the host.

```bash
docker build --target dev -t local-rag-dev .     # test image (deps + dev deps)
docker run --rm local-rag-dev pytest             # full suite
docker run --rm local-rag-dev pytest tests/test_paths.py -x   # one file
docker run --rm local-rag-dev ruff check . && docker run --rm local-rag-dev ruff format .

docker compose up --build                        # run the service
docker compose --profile dev run --rm tests      # tests via compose
curl localhost:8000/health                       # 200 ok / 503 degraded
```

Verify pins without a full model download:
`docker run --rm -v "$PWD/requirements-dev.txt:/r.txt:ro" -v "$PWD/requirements.txt:/requirements.txt:ro" python:3.12-slim pip install --dry-run -q -r /r.txt`

`ANTHROPIC_API_KEY` must be in the environment or `.env` (see `.env.example`).
Uploads land in the `./documents` bind mount (read-write — `/ingest` writes
there). LanceDB and `jobs.db` live on the `lancedb` named volume; deleting that
volume discards the index and the job history.

## Architecture constraints

Load-bearing — violating these produces failures that are not obvious from a
local test run.

**One process, one writer.** LanceDB uses optimistic concurrency over a
manifest; two writer processes on the same table produce commit conflicts.
uvicorn stays at `--workers 1`. Anything that writes must run in-process.

**No PyTorch.** Inference goes through `fastembed` / ONNX Runtime only. Pulling
in torch (directly, or transitively via a reranker or extractor library) adds
3–4 GB to a ~1 GB image target. Check every new dependency for it.

**ONNX inference is CPU-bound and synchronous.** Call the embedder and reranker
through `run_in_threadpool` — never directly inside an `async def` handler, or
the event loop stalls for every concurrent request.

**Models load once, at startup.** `build_resources` constructs the embedder,
reranker, LanceDB handle, and `JobStore`; `lifespan` holds them on
`app.state.resources`. The image bakes weights into `/models` with
`HF_HUB_OFFLINE=1`, so any code path that triggers a runtime download hangs or
fails in a container with no network. The fastembed-specific traps behind this
are in `app/services/CLAUDE.md`.

**Docker layer order.** The model layer and `pip install` both precede any app
code, or every source edit re-downloads several hundred MB of weights.

## Tuning defaults

Chunk size 500, overlap 50, `top_k` 50, rerank depth 5 (`app/config.py`). Per
the brief these are the highest-leverage knobs in the system; move them against
the eval harness, not by intuition. The harness is a separate tool from the
service and measures retrieval (recall@k) and generation (groundedness)
independently — most "RAG is broken" reports are retrieval failures.

## Conventions

Tests come first — write the spec, then the implementation. How they are
written is in `tests/CLAUDE.md`.
