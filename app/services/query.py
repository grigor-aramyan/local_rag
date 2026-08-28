from __future__ import annotations

from typing import Any

from app.config import Settings
from app.schemas import QueryResponse
from app.services.generation import RetrievedChunk, generate_answer
from app.services.registry import Resources
from app.services.vectorstore import hybrid_search, open_chunks_table

NO_CONTEXT_WARNING = (
    "nothing in the indexed corpus looked relevant to this question; the answer "
    "below comes from the model's general knowledge, not your documents"
)


def run_query(
    question: str, top_k: int | None, resources: Resources, settings: Settings
) -> QueryResponse:
    """Embed -> hybrid retrieve -> rerank -> generate. Blocking — call via `run_in_threadpool`.

    Build step 10 of the brief. Retrieval finding nothing relevant is not an
    error (brief decision #12): the answer still comes back, generated from the
    model's own knowledge, with `warning` set so the caller can tell the two
    cases apart.
    """
    table = open_chunks_table(resources.db, settings)
    query_vector = _embed(resources, question)
    candidates = hybrid_search(table, query_vector, question, top_k or settings.top_k)

    chunks = _select_chunks(resources, settings, question, candidates) if candidates else []
    answer, citations = generate_answer(resources.llm_client, settings, question, chunks)

    warning = NO_CONTEXT_WARNING if not candidates else None
    return QueryResponse(answer=answer, citations=citations, warning=warning)


def _embed(resources: Resources, question: str) -> list[float]:
    vectors = list(resources.embedder.embed([question]))
    return [float(value) for value in vectors[0]]


def _select_chunks(
    resources: Resources,
    settings: Settings,
    question: str,
    candidates: list[dict[str, Any]],
) -> list[RetrievedChunk]:
    """Narrow `top_k` hybrid candidates down to `rerank_top_n` for the prompt.

    The reranker is toggleable (brief decision #10): disabled, the hybrid
    ranking — already best-first — is trusted as-is.
    """
    if settings.rerank_enabled:
        scores = list(resources.reranker.rerank(question, [row["text"] for row in candidates]))
        ranked = sorted(
            zip(candidates, scores, strict=True), key=lambda pair: pair[1], reverse=True
        )
        candidates = [row for row, _ in ranked]

    return [
        RetrievedChunk(text=row["text"], source=row["source"], page=row["page"])
        for row in candidates[: settings.rerank_top_n]
    ]
