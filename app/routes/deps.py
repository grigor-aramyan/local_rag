from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException, Request, status

from app.config import Settings
from app.jobs import JobStore
from app.services.registry import Resources


def get_settings_dep(request: Request) -> Settings:
    return request.app.state.settings


def get_resources(request: Request) -> Resources:
    return request.app.state.resources


def get_ready_resources(request: Request) -> Resources:
    """Resources for handlers that touch the index, refusing an incompatible one.

    `/health` deliberately uses `get_resources` instead: it has to be able to
    describe a degraded service rather than be blocked by it.
    """
    resources: Resources = request.app.state.resources
    if resources.config_error is not None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=resources.config_error,
        )
    if not resources.ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="models are still loading or failed to load",
        )
    return resources


def get_jobs(request: Request) -> JobStore:
    jobs = request.app.state.resources.jobs
    if jobs is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="job store unavailable",
        )
    return jobs


SettingsDep = Annotated[Settings, Depends(get_settings_dep)]
ResourcesDep = Annotated[Resources, Depends(get_resources)]
ReadyResourcesDep = Annotated[Resources, Depends(get_ready_resources)]
JobsDep = Annotated[JobStore, Depends(get_jobs)]
