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
    tokenizer: Any | None = None
    # Set when the index on disk was built under incompatible settings. Held
    # rather than raised so `/health` can explain the refusal instead of the
    # container crash-looping with the reason buried in its logs.
    config_error: str | None = None

    @property
    def ready(self) -> bool:
        if self.config_error is not None:
            return False
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
        tokenizer=tokenizer_for(embedder),
    )


def tokenizer_for(embedder: Any) -> Any:
    """Borrow the embedder's tokenizer for chunking, with truncation disabled.

    Chunking has to measure text in the units the model actually consumes, so it
    uses the model's own tokenizer rather than a word or character approximation
    that would overshoot the 512-token limit and lose each chunk's tail.

    Two traps, both silent: the tokenizer ships with truncation pinned at the
    context length, so encoding a long document through it reports only the first
    512 tokens and the rest of the file never gets chunked at all; and the object
    is shared with the live ONNX session, so truncation is disabled on a copy
    rather than in place. fastembed does not expose this — reaching through
    `.model.tokenizer` is the only route, and a version bump can move it.
    """
    from tokenizers import Tokenizer

    inner = getattr(getattr(embedder, "model", None), "tokenizer", None)
    if inner is None:
        raise RuntimeError(
            "could not reach the embedder's tokenizer at `.model.tokenizer`; "
            "fastembed's internals have moved and chunking cannot measure tokens"
        )

    copy = Tokenizer.from_str(inner.to_str())
    copy.no_truncation()
    copy.no_padding()
    return copy
