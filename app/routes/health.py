from __future__ import annotations

from fastapi import APIRouter, Response, status

from app.routes.deps import ResourcesDep
from app.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, response_model_exclude_none=True)
async def health(response: Response, resources: ResourcesDep) -> HealthResponse:
    """Report whether the models and database actually loaded.

    A liveness check that only proves the process is up would let an orchestrator
    route traffic to a container whose embedder failed to load. An index built
    under incompatible settings is degraded for the same reason — everything is
    loaded, but a query would return quietly wrong results — so the reason is
    reported here rather than left in the startup log.
    """
    ready = resources.ready
    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return HealthResponse(
        status="ok" if ready else "degraded",
        embedder=resources.embedder is not None,
        reranker=resources.reranker is not None,
        database=resources.db is not None,
        detail=resources.config_error,
    )
