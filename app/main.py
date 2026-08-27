from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.concurrency import run_in_threadpool

from app.config import get_settings
from app.routes import router
from app.services.registry import Resources, build_resources
from app.services.vectorstore import ConfigMismatchError, verify_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the models and open the database once, before serving anything.

    `build_resources` is blocking — loading ONNX sessions on the event loop would
    stall every other startup task — so it goes through the threadpool.
    """
    settings = get_settings()
    app.state.settings = settings
    app.state.resources = Resources()

    resources = await run_in_threadpool(build_resources, settings)
    app.state.resources = resources

    # A changed embedder or chunk size against an existing index degrades
    # retrieval silently. Refuse to serve, but stay up: the operator needs to read
    # the reason from `/health`, not from a crash loop.
    if resources.db is not None:
        try:
            await run_in_threadpool(verify_config, resources.db, settings)
        except ConfigMismatchError as exc:
            logger.error("refusing to serve: %s", exc)
            resources.config_error = str(exc)

    if resources.jobs is not None:
        stranded = resources.jobs.fail_interrupted_jobs()
        if stranded:
            logger.warning("closed out %d job(s) interrupted by a restart", stranded)

    logger.info("ready")
    try:
        yield
    finally:
        if resources.jobs is not None:
            resources.jobs.close()
        app.state.resources = Resources()


app = FastAPI(
    title="local_rag",
    summary="Self-hosted RAG over a custom knowledge base.",
    lifespan=lifespan,
)
app.include_router(router)
