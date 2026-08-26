# Self-Hosted RAG System — Project Brief

## Goal

A self-hosted RAG service over a custom knowledge base. Ships as a Docker
image with embedding and reranking models baked in, so a user runs
`docker compose up` and can ingest documents and query immediately, with no
extra downloads or setup.

## Stack

- **Language:** Python 3.12
- **API:** FastAPI + uvicorn (single worker)
- **Vector DB:** LanceDB (embedded, not a server)
- **Model runtime:** ONNX Runtime via `fastembed` — **no PyTorch**, it would
  add ~3–4 GB to the image
- **Embedder:** small ONNX model (e.g. `bge-small-en-v1.5`)
- **Reranker:** ONNX cross-encoder (e.g. `ms-marco-MiniLM-L-6-v2`)
- **Generation:** external OpenAI-compatible API (configurable base URL)

Target image size: ~1 GB.

## Architecture

Two pipelines, **one container, one process** (LanceDB uses optimistic
concurrency on a manifest — two writer processes on the same table produce
commit conflicts).

**Ingestion:** extract → chunk → embed → write to LanceDB → build FTS index
→ compact.

**Query:** embed question → hybrid search (vector + full-text) → rerank →
assemble prompt → call LLM → stream response with citations.

---

## Build Steps

1. **Project scaffold.** FastAPI app, `lifespan` handler, config via
   environment variables, `requirements.txt` pinned.

2. **Dockerfile, multi-stage.** Stage 1 downloads models into `/models`.
   Stage 2 copies them in, sets `HF_HOME=/models`, `HF_HUB_OFFLINE=1`,
   `TRANSFORMERS_OFFLINE=1`, `OMP_NUM_THREADS=4`. Model layer must come
   *before* the app-code layer so code edits don't invalidate it.

3. **docker-compose.yml.** Named volume for LanceDB data, read-only bind
   mount for source documents, healthcheck with a generous `start_period`.

4. **Model loading at startup.** Embedder, reranker, and LanceDB connection
   loaded once in `lifespan` and held on `app.state`. ONNX inference is
   CPU-bound — call it through `run_in_threadpool`, never directly inside an
   `async def` handler.

5. **LanceDB schema.** Columns: `id`, `text`, `vector`, `source`, `page`,
   `content_hash`, `ingested_at`. Store the raw chunk text — it's needed for
   prompts and citations.

6. **Config-compatibility guard.** Write embedding model name, dimension, and
   chunking params into a metadata row on first ingest. Verify on startup;
   refuse to serve with a clear error if they diverge from current config.
   Prevents silently broken retrieval after a config change.

7. **Extraction + chunking.** Per-format extractors; token-based chunking
   with overlap. Chunk IDs keyed on content hash so re-ingestion is
   idempotent.

8. **Ingestion as background jobs.** `POST /ingest` returns a `job_id`
   immediately; work runs in a `BackgroundTasks` task; `GET /jobs/{id}`
   reports progress. Persist job state to SQLite on the same volume so a
   restart doesn't strand jobs in `running`.

9. **Index management.** Create the FTS index at ingest time (retrofitting
   requires a full rebuild). Skip the ANN index below ~100k vectors — brute
   force is faster. Create/re-create IVF-PQ above that threshold. Run
   `table.optimize()` after each ingestion job to compact fragments.

10. **Query pipeline.** Embed → hybrid retrieve `top_k≈50` → rerank to ~5 →
    build prompt with numbered context blocks → stream from the LLM.

11. **Citations.** Prompt the model to emit `[n]` markers; map them back to
    `source`/`page` in the response payload.

12. **Health and demo.** `/health` must verify models actually loaded, not
    just that the process is up. Ship `sample_docs/` plus a one-command demo
    ingest so a fresh container isn't an empty index with no next step.

13. **Evaluation harness.** Separate from the service. Golden Q&A set;
    measure retrieval (recall@k) and generation (groundedness) independently
    — most "RAG is broken" reports are retrieval failures.

---

## Decisions to Make Before / During Coding

**Product scope**
1. Multi-tenancy: single shared knowledge base, or per-user collections with
   ACL filtering pushed into the LanceDB query? Answer: multi-tenancy baked-in for future use, but default ACL to `all` value for current implementation
2. Is the deliverable an API only, or is a minimal UI expected? Answer: API only
3. Ingestion trigger: watched folder, upload endpoint, or both? Answer: upload endpoint

**Documents**
4. Which formats must work at v1? PDF is the quality bottleneck — scanned and
   multi-column documents need real layout parsing, which is heavy. Consider
   deferring to v2 if the corpus is markdown/HTML/text. Answer: Stick with md/html/text files for now
5. OCR for scanned PDFs — in scope, or explicitly unsupported? Answer: unsupported for current implementation
6. Expected corpus size? Drives the ANN-index threshold and whether image
   size or index tuning matters more. Answer: below 10 MB for now
7. Document updates and deletes: supported, or is re-ingest-everything
   acceptable? Answer: updates and deletes supported. can add additional endpoints for these if needed

**Models**
8. Exact embedder and reranker, and whether to use int8-quantized variants
   (roughly halves model size, small quality cost). Answer: embedder - BAAI/bge-small-en-v1.5, reranker - Xenova/ms-marco-MiniLM-L-6-v2. no need for int8-quantized variants for now
9. English-only or multilingual? Changes model choice and image size
   significantly. Answer: english only for now
10. Should the reranker be optional/toggleable? Measure whether it helps
    before making it mandatory in the hot path. Answer: yes, toggleable

**Generation**
11. Confirm external-API-only for v1. If bundled local generation is wanted,
    make it a separate compose profile with Ollama as a sidecar — do not bake
    LLM weights into the main image. Answer: generation will be via external API call
12. Behaviour when retrieval returns nothing relevant: refuse to answer, or
    answer from model knowledge with a warning? Answer: answer with a warning

**Operational**
13. Auth on the API — none, static token, or something more? Answer: none for now
14. Target hardware: CPU-only, or is a GPU assumed available? Answer: make GPU optional if possible, it might not be available always
15. Chunk size, overlap, `top_k`, and rerank depth: start with 500/50/50/5
    and tune against the eval set. These are the highest-leverage knobs;
    getting them right matters more than any other choice here. Answer: agree, start with 500/50/50/5, i will tune them later if needed