# app/

Guidance for the application package. Root `CLAUDE.md` holds the project state,
commands, and the global architecture constraints; endpoint contracts are in
`app/routes/CLAUDE.md` and the pipelines are in `app/services/CLAUDE.md`.

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

## Jobs

Job state persists to SQLite (`jobs.db` on the `lancedb` volume) so a restart
doesn't strand jobs. `fail_interrupted_jobs()` runs at startup and closes out
anything left `pending` or `running`, since background tasks live in-process and
none can legitimately survive a restart.

## Config-compatibility guard (not built yet)

Embedding model name, dimension, and chunking params go into a metadata row on
first ingest and get verified at startup; divergence must refuse to serve with
an explicit error. A changed embedder against an old index degrades retrieval
silently rather than crashing.

## Startup wiring

`build_resources` (in `app/services/registry.py`) constructs the embedder,
reranker, LanceDB handle, and `JobStore`; `lifespan` in `app/main.py` holds them
on `app.state.resources` for the process lifetime. Nothing may rebuild these per
request — see the one-process-one-writer and models-load-once constraints in the
root file.
