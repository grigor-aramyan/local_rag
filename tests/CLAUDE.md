# tests/

Tests come first: write the spec, then the implementation against it.

They are written as executable spec — one behaviour per test, named as a
sentence, with a docstring only where the *why* isn't obvious from the name.

`conftest.py` provides a `make_client(**setting_overrides)` factory that patches
`build_resources` with stand-ins, so no test loads a real ONNX model. Use it
rather than constructing the app directly.

Alongside it: `word_tokenizer` (a vocab-free whitespace `Tokenizer` that reports
real character offsets — chunking only needs `.offsets`), `truncating_tokenizer`
(the same, truncating, standing in for the embedder's 512-token limit),
`fake_embedder` (deterministic vectors, same call shape as `TextEmbedding`), and
`blank_db` (a *real*, empty LanceDB connection). LanceDB is embedded and creates
nothing until written, so store and ingest tests use the real thing rather than a
double — and the startup config guard inspects that handle, so it cannot be a
bare `object()`.

`test_paths.py` covers the upload sanitizer and the containment backstop — the
most heavily tested code in the repo, because `POST /ingest` takes an
attacker-controlled filename. Extend it whenever `app/paths.py` changes; the
rules it enforces are described in `app/CLAUDE.md`.

`test_registry.py` pins the fastembed offline-loading behaviour and
`tokenizer_for` (see `app/services/CLAUDE.md`). Note that the container-only
traps listed there do *not* show up in this suite — a green run says nothing
about whether the image starts. After changing anything in `app/services/`,
build the runtime image and actually ingest a document through it.

`test_chunking.py`, `test_vectorstore.py`, and `test_ingestion.py` cover the
pipeline: windowing and chunk identity, the LanceDB schema, per-document
replacement and the filter-injection guard, and the orchestration including how
failures are reported without leaking internals.

Run everything in Docker (`docker run --rm local-rag-dev pytest`); nothing is
installed on the host.
