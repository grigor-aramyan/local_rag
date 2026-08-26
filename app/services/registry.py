from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from app.config import Settings
from app.jobs import JobStore

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Resources:
    """Everything loaded once at startup and held for the process lifetime.

    ONNX sessions and the LanceDB handle are expensive to construct and must not
    be rebuilt per request. LanceDB's optimistic concurrency also means a single
    writer per table — which is why uvicorn stays at one worker.
    """

    embedder: Any | None = None
    reranker: Any | None = None
    db: Any | None = None
    jobs: JobStore | None = None

    @property
    def ready(self) -> bool:
        return all(part is not None for part in (self.embedder, self.reranker, self.db))


def build_resources(settings: Settings) -> Resources:
    """Load models and open the database. Blocking — call via `run_in_threadpool`.

    Imports are local so the module stays importable (and unit-testable) without
    paying the ONNX import cost.
    """
    import lancedb
    from fastembed import TextEmbedding
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    settings.lancedb_path.mkdir(parents=True, exist_ok=True)
    settings.documents_path.mkdir(parents=True, exist_ok=True)

    load_options = {
        "cache_dir": str(settings.model_cache_path),
        "local_files_only": settings.model_local_files_only,
    }

    logger.info("loading embedder %s from %s", settings.embedding_model, load_options["cache_dir"])
    embedder = TextEmbedding(settings.embedding_model, **load_options)

    logger.info("loading reranker %s from %s", settings.reranker_model, load_options["cache_dir"])
    reranker = TextCrossEncoder(settings.reranker_model, **load_options)

    logger.info("opening lancedb at %s", settings.lancedb_path)
    db = lancedb.connect(str(settings.lancedb_path))

    return Resources(
        embedder=embedder,
        reranker=reranker,
        db=db,
        jobs=JobStore(settings.jobs_db_path),
    )
