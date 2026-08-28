from __future__ import annotations

import anthropic
from fastapi import APIRouter, HTTPException, status
from starlette.concurrency import run_in_threadpool

from app.routes.deps import ReadyResourcesDep, SettingsDep
from app.schemas import QueryRequest, QueryResponse
from app.services.query import run_query

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest, resources: ReadyResourcesDep, settings: SettingsDep
) -> QueryResponse:
    """Embed -> hybrid retrieve -> rerank -> generate, grounded via native citations.

    The embedder, the reranker, and the call to Claude are all blocking, so the
    whole pipeline runs on the threadpool rather than inline in this handler.
    """
    try:
        return await run_in_threadpool(
            run_query, request.question, request.top_k, resources, settings
        )
    except anthropic.RateLimitError as exc:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            "the LLM is rate-limiting this service; try again shortly",
        ) from exc
    except (anthropic.APIConnectionError, anthropic.APIStatusError) as exc:
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY, "generation failed: the LLM API is unavailable"
        ) from exc
    except TypeError as exc:
        # The SDK raises a plain TypeError (not a typed anthropic.* exception) when
        # it cannot resolve any credential before ever sending a request — verified
        # against anthropic==1.1.0 with no ANTHROPIC_API_KEY set. Anything else with
        # this type is a real bug and must not be swallowed as a config error.
        if "authentication" not in str(exc).lower():
            raise
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "generation is not configured: no Anthropic API credentials are set",
        ) from exc
