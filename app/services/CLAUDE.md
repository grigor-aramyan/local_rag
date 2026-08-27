# app/services/

The pipelines and the model/database handles they run on. Endpoint contracts are
in `app/routes/CLAUDE.md`; the global invariants (one writer, no PyTorch,
threadpool for ONNX, models load once) are in the root `CLAUDE.md`.

## fastembed traps

Four of them, all of which only surface when the container actually runs — the
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
4. Chunking measures text with the model's own tokenizer, reached through
   `embedder.model.tokenizer` — not public API, so a version bump can move it.
   That tokenizer ships with **truncation pinned at 512**: encode a long
   document through it directly and it reports only the first 512 tokens, so
   everything past that is never chunked and never indexed, with no error
   anywhere. `registry.tokenizer_for` disables truncation on a *copy* — the
   original is shared with the live ONNX session and must keep truncating.

## Ingest pipeline

Upload → sanitize/stage/commit → create job → return `job_id` (202). The
background task runs extract → chunk → embed → write → build FTS index →
`table.optimize()`, reporting progress through the job store as it goes.
Starlette runs a sync background task on the threadpool, so the ONNX calls in
`run_ingestion` are already off the event loop and are not wrapped again.

Chunk IDs key on content hash, which is what makes re-ingesting the same
document idempotent; source and ordinal join the hash so repeated boilerplate
stays as separate rows. Each document is one `merge_insert` scoped by
`when_not_matched_by_source_delete`, so a shortened document loses its stale
tail in the same commit that writes its new chunks.

A document that fails stops the run — documents already written stay written.
They are individually valid, and re-uploading re-ingests idempotently. Job
errors reach the client through `GET /jobs/{id}`, so only the domain errors
listed in `_REPORTABLE` pass their message through; anything else is logged in
full and reported generically rather than leaking a host path or traceback.

`chunk_size` is measured in the embedder's tokens, so it must stay under the
model's context length (512 for bge-small) — see trap 4.

## Query pipeline

Embed → hybrid retrieve (vector + FTS) `top_k≈50` → rerank to ~5 → numbered
context blocks → stream from Claude.

## LanceDB

Table columns: `id`, `text`, `vector`, `source`, `page`, `content_hash`,
`ingested_at`, `acl`. Raw chunk text is stored, not just vectors — prompts and
citations need it. `page` is null for md/HTML/text, which have no pages; only a
PDF extractor would set it. `acl` is the brief's baked-in multi-tenancy and is
always `"all"` for now — nothing filters on it, but adding a column to a
populated table means rebuilding it, so it is carried from the first write.

Rows are bound to `table.schema` via `pa.Table.from_pylist` before writing:
LanceDB infers every column of a plain dict as nullable, which the non-nullable
schema then rejects on append.

A `source` is interpolated into a LanceDB filter expression, so
`vectorstore._checked_source` re-validates it against the sanitiser's alphabet
rather than trusting that it came from the sanitiser. Without that, a name
carrying a quote could rewrite the predicate and delete another document.

**Indexes.** Create the FTS index at ingest time; retrofitting one requires a
full rebuild. Skip the ANN index below ~100k vectors (brute force is faster
there); create IVF-PQ above `ann_index_threshold`, with `num_partitions` capped
at the row count — IVF training fails outright when asked to cluster more
partitions than it has vectors.

**API notes** (lancedb 0.24.0): `db.table_names()` silently defaults to
`limit=10`; a missing table raises a plain `ValueError`; and `merge_insert`
with an empty batch raises `IndexError`, so "this document now has no chunks"
is a `table.delete()`, not a merge.

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
