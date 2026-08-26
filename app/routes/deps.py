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
JobsDep = Annotated[JobStore, Depends(get_jobs)]
