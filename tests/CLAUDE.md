# tests/

Tests come first: write the spec, then the implementation against it.

They are written as executable spec — one behaviour per test, named as a
sentence, with a docstring only where the *why* isn't obvious from the name.

`conftest.py` provides a `make_client(**setting_overrides)` factory that patches
`build_resources` with stand-ins, so no test loads a real ONNX model. Use it
rather than constructing the app directly.

`test_paths.py` covers the upload sanitizer and the containment backstop — the
most heavily tested code in the repo, because `POST /ingest` takes an
attacker-controlled filename. Extend it whenever `app/paths.py` changes; the
rules it enforces are described in `app/CLAUDE.md`.

`test_registry.py` pins the fastembed offline-loading behaviour (see
`app/services/CLAUDE.md`). Note that the container-only traps listed there do
*not* show up in this suite — a green run says nothing about whether the image
starts.

Run everything in Docker (`docker run --rm local-rag-dev pytest`); nothing is
installed on the host.
