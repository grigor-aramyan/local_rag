# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project state

Scaffold, not a working RAG system yet. `docs/brief.md` is the authoritative
spec — read it before starting work; it holds the build order and an
open-questions section whose answers are not recorded anywhere. If a decision
from that section is needed, ask rather than assume.

Built: config, upload handling, the SQLite job store, `/health`, and the route
surface. Stubbed: `run_ingestion` in `app/services/ingestion.py` (build steps
7–9) and `POST /query` (step 10, returns 501). An upload is stored and gets a
job, but nothing is indexed yet — the job deliberately fails with a message
saying so rather than pretending to succeed.

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
fails in a container with no network.

Three fastembed traps, all of which only surface when the container actually
runs — the test suite passes regardless:

1. `TextCrossEncoder` is **not** re-exported at the `fastembed` top level —
   import it from `fastembed.rerank.cross_encoder`. Both the Dockerfile's
   model-download stage and `registry.py` need the same import.
2. fastembed ignores `HF_HOME` for its ONNX cache and uses
   `FASTEMBED_CACHE_PATH` (default: a temp dir). If only `HF_HOME` is set, stage
   1 writes the weights somewhere stage 2 never copies.
3. Even with the cache present, fastembed's HuggingFace path *raises* under
   `HF_HUB_OFFLINE=1` instead of falling back to it — `local_files_only=True`
   is what makes it read the baked weights. `tests/test_registry.py` pins this.

**Docker layer order.** The model layer and `pip install` both precede any app
code, or every source edit re-downloads several hundred MB of weights.

**Config-compatibility guard (not built yet).** Embedding model name, dimension,
and chunking params go into a metadata row on first ingest and get verified at
startup; divergence must refuse to serve with an explicit error. A changed
embedder against an old index degrades retrieval silently rather than crashing.

## Uploads are untrusted input

`POST /ingest` is a multipart upload, so the `filename` is attacker-controlled.
`sanitize_upload_filename` strips POSIX *and* Windows separators, rejects null
bytes, forces the result to a plain basename, strips leading dots, and gates on
an extension allowlist; `resolve_within` is the containment backstop that also
catches symlinks planted in the mount. Both are in `app/paths.py` and are the
most heavily tested code here — extend the tests when touching them.

The batch is all-or-nothing: files stream to `.staging-*` temp files beside
their destination, counting bytes against `max_upload_bytes` as they arrive
(never trusting Content-Length) and validating UTF-8 incrementally, then
`os.replace` into place only once every file has passed. A rejected upload must
never leave a partial file behind. Re-uploading a name overwrites and
re-ingests — chunk IDs keyed on content hash are what make that idempotent.

## Pipelines

**Ingest** — upload → sanitize/stage/commit → create job → return `job_id`
(202). Background task runs extract → chunk → embed → write → build FTS index →
`table.optimize()`. `GET /jobs/{id}` reports progress. Job state persists to
SQLite so a restart doesn't strand jobs; `fail_interrupted_jobs()` runs at
startup and closes out anything left `pending` or `running`, since background
tasks live in-process and none can legitimately survive a restart.

**Query** — embed → hybrid retrieve (vector + FTS) `top_k≈50` → rerank to ~5 →
numbered context blocks → stream from Claude.

LanceDB table columns: `id`, `text`, `vector`, `source`, `page`, `content_hash`,
`ingested_at`. Raw chunk text is stored, not just vectors — prompts and
citations need it.

**Indexes.** Create the FTS index at ingest time; retrofitting one requires a
full rebuild. Skip the ANN index below ~100k vectors (brute force is faster
there); create IVF-PQ above `ann_index_threshold`.

## Generation

Anthropic API via the official `anthropic` SDK (pinned 1.1.0, which is built on
`httpx2`, not `httpx`). Default model `claude-opus-5`. Only generation leaves
the container — embedding and reranking are local, so the corpus is never sent
anywhere.

Prefer the API's **native citations** over prompting for `[n]` markers: set
`citations: {enabled: true}` on each retrieved chunk passed as a `document`
content block and the response comes back split into text blocks carrying
`cited_text` plus a char or page location — grounded in real spans rather than
in the model's willingness to follow a format instruction. Note they are
incompatible with `output_config.format`.

Use `thinking: {type: "adaptive"}` and `output_config: {effort: ...}`;
`budget_tokens` is rejected with a 400 on current models. Stream long
generations. Invoke the `claude-api` skill before writing SDK code rather than
working from memory — the API surface has drifted.

## Tuning defaults

Chunk size 500, overlap 50, `top_k` 50, rerank depth 5 (`app/config.py`). Per
the brief these are the highest-leverage knobs in the system; move them against
the eval harness, not by intuition. The harness is a separate tool from the
service and measures retrieval (recall@k) and generation (groundedness)
independently — most "RAG is broken" reports are retrieval failures.

## Conventions

Tests come first, and they are written as executable spec: one behaviour per
test, named as a sentence, with a docstring only where the *why* isn't obvious
from the name. `tests/conftest.py` provides a `make_client(**setting_overrides)`
factory that patches `build_resources` with stand-ins, so no test loads a real
ONNX model.
