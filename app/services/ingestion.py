from __future__ import annotations

import logging
from typing import Any

from app.config import Settings
from app.jobs import JobStore
from app.paths import PathTraversalError, resolve_within
from app.services.chunking import chunk_document
from app.services.extraction import ExtractionError, extract_text
from app.services.registry import Resources
from app.services.vectorstore import (
    ConfigMismatchError,
    UnsafeSourceError,
    compact,
    ensure_ann_index,
    ensure_fts_index,
    open_chunks_table,
    record_config,
    replace_document,
    rows_from_chunks,
)

logger = logging.getLogger(__name__)

UNEXPECTED_ERROR = "indexing failed unexpectedly; see the service log for details"

# Errors whose message is written for the operator and safe to hand back over
# HTTP. Anything else is logged in full and reported generically, so a stray
# traceback or host path never reaches a client through `GET /jobs/{id}`.
_REPORTABLE = (ExtractionError, ConfigMismatchError, UnsafeSourceError)


def run_ingestion(job_id: str, resources: Resources, settings: Settings) -> None:
    """Process one ingestion job. Blocking — runs as a background task.

    Build steps 7-9 of the brief: extract per format, token-chunk with overlap,
    embed, write to LanceDB, build the FTS index, then compact. Starlette runs a
    sync background task on the threadpool, so the ONNX calls below are already
    off the event loop — that is why they are not wrapped again here.

    A document that fails stops the run. Documents already written stay written:
    each is one atomic replacement, they are valid on their own, and re-uploading
    re-ingests idempotently.
    """
    jobs: JobStore | None = resources.jobs
    if jobs is None:  # pragma: no cover - lifespan always provides one
        raise RuntimeError("job store unavailable")

    job = jobs.get(job_id)
    if job is None:
        logger.error("ingestion asked for unknown job %s", job_id)
        return

    jobs.mark_running(job_id)
    processed = 0
    document = ""

    try:
        # Before the table is touched: writing new vectors into a table built by a
        # different embedder degrades retrieval silently rather than failing.
        record_config(resources.db, settings)
        table = open_chunks_table(resources.db, settings)

        for document in job.documents:
            _ingest_document(document, table, resources, settings)
            processed += 1
            jobs.set_progress(job_id, processed)

        ensure_fts_index(table)
        ensure_ann_index(table, settings)
        compact(table)
    except _REPORTABLE as exc:
        logger.warning("job %s failed on %r: %s", job_id, document, exc)
        jobs.mark_failed(job_id, str(exc))
    except PathTraversalError:
        # The message names the documents root; the client gets the name only.
        logger.warning("job %s rejected unsafe document name %r", job_id, document)
        jobs.mark_failed(job_id, f"{document}: rejected as an unsafe document name")
    except Exception:
        logger.exception("job %s failed unexpectedly on %r", job_id, document)
        jobs.mark_failed(job_id, f"{document}: {UNEXPECTED_ERROR}")
    else:
        jobs.mark_completed(job_id)


def _ingest_document(name: str, table: Any, resources: Resources, settings: Settings) -> None:
    """Extract, chunk, embed, and write one document as a single replacement."""
    # Defence in depth: names come from the sanitiser via the job store, but the
    # job store is not a trust boundary and this is what opens a file.
    path = resolve_within(settings.documents_path, name)

    text = extract_text(path)
    chunks = chunk_document(
        text,
        source=name,
        tokenizer=_tokenizer(resources),
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    if not chunks:
        # Reporting success on a document that indexed nothing is the worse
        # failure: the user would query it and get silence.
        raise ExtractionError(f"{name} produced no indexable text")

    vectors = _embed(resources, settings, [chunk.text for chunk in chunks])
    replace_document(table, name, rows_from_chunks(chunks, vectors))
    logger.info("indexed %s as %d chunk(s)", name, len(chunks))


def _embed(resources: Resources, settings: Settings, texts: list[str]) -> list[list[float]]:
    """Embed chunk text. CPU-bound ONNX, already on a worker thread."""
    embedded = list(resources.embedder.embed(texts, batch_size=settings.embed_batch_size))
    if len(embedded) != len(texts):  # pragma: no cover - defensive
        raise RuntimeError(f"embedder returned {len(embedded)} vectors for {len(texts)} chunks")

    vectors = [[float(value) for value in vector] for vector in embedded]
    wrong = {len(vector) for vector in vectors} - {settings.embedding_dim}
    if wrong:
        raise ConfigMismatchError(
            f"the embedder produced {sorted(wrong)}-dimensional vectors but "
            f"embedding_dim is {settings.embedding_dim}"
        )
    return vectors


def _tokenizer(resources: Resources) -> Any:
    if resources.tokenizer is None:  # pragma: no cover - lifespan always provides one
        raise RuntimeError("tokenizer unavailable; chunking cannot measure the model's tokens")
    return resources.tokenizer
