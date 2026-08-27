# app/services/

The pipelines and the model/database handles they run on. Endpoint contracts are
in `app/routes/CLAUDE.md`; the global invariants (one writer, no PyTorch,
threadpool for ONNX, models load once) are in the root `CLAUDE.md`.

## fastembed traps

Three of them, all of which only surface when the container actually runs — the
test suite passes regardless:

1. `TextCrossEncoder` is **not** re-exported at the `fastembed` top level —
   import it from `fastembed.rerank.cross_encoder`. Both the Dockerfile's
   model-download stage and `registry.py` need the same import.
2. fastembed ignores `HF_HOME` for its ONNX cache and uses
   `FASTEMBED_CACHE_PATH` (default: a temp dir). If only `HF_HOME` is set, stage
   1 writes the weights somewhere stage 2 never copies.
3. Even with the cache present, fastembed's HuggingFace path *raises* under
   `HF_HUB_OFFLINE=1` instead of falling back to it — `local_files_only=True`
   is what makes it read the baked weights. `tests/test_registry.py` pins this.

## Ingest pipeline

Upload → sanitize/stage/commit → create job → return `job_id` (202). The
background task runs extract → chunk → embed → write → build FTS index →
`table.optimize()`, reporting progress through the job store as it goes.
Chunk IDs key on content hash, which is what makes re-ingesting the same
document idempotent.

## Query pipeline

Embed → hybrid retrieve (vector + FTS) `top_k≈50` → rerank to ~5 → numbered
context blocks → stream from Claude.

## LanceDB

Table columns: `id`, `text`, `vector`, `source`, `page`, `content_hash`,
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
