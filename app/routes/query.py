from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from app.schemas import QueryRequest, QueryResponse

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """Answer a question from the indexed corpus.

    Build step 10 of the brief lands here: embed the question, hybrid-retrieve
    `top_k`, rerank down to `rerank_top_n`, assemble numbered context blocks,
    then stream the model's answer with `[n]` citation markers resolved back to
    source and page.
    """
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="the query pipeline is not built yet",
    )
