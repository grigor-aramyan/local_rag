from __future__ import annotations

import logging

from app.config import Settings
from app.jobs import JobStore
from app.services.registry import Resources

logger = logging.getLogger(__name__)

NOT_IMPLEMENTED = (
    "the ingestion pipeline (extract, chunk, embed, index) is not built yet; "
    "the document was stored but not indexed"
)


def run_ingestion(job_id: str, resources: Resources, settings: Settings) -> None:
    """Process one ingestion job. Blocking — runs as a background task.

    Build steps 7–9 of the brief land here: extract per format, token-chunk with
    overlap, embed, write to LanceDB, build the FTS index, then `table.optimize()`.
    Chunk IDs key on content hash, which is what makes re-ingesting the same
    document idempotent.
    """
    jobs: JobStore | None = resources.jobs
    if jobs is None:  # pragma: no cover - lifespan always provides one
        raise RuntimeError("job store unavailable")

    jobs.mark_running(job_id)
    logger.warning("job %s: %s", job_id, NOT_IMPLEMENTED)
    jobs.mark_failed(job_id, NOT_IMPLEMENTED)
