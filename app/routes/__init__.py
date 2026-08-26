from fastapi import APIRouter

from app.routes import health, ingest, query

router = APIRouter()
router.include_router(health.router)
router.include_router(ingest.router)
router.include_router(query.router)

__all__ = ["router"]
