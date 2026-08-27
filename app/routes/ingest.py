from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile, status

from app.paths import PathTraversalError, UnsupportedFormatError
from app.routes.deps import JobsDep, ReadyResourcesDep, SettingsDep
from app.schemas import IngestResponse, JobStatus
from app.services.ingestion import run_ingestion
from app.storage import UploadTooLargeError, save_uploads

router = APIRouter(tags=["ingest"])


@router.post("/ingest", response_model=IngestResponse, status_code=status.HTTP_202_ACCEPTED)
async def ingest(
    background: BackgroundTasks,
    resources: ReadyResourcesDep,
    settings: SettingsDep,
    jobs: JobsDep,
    files: Annotated[list[UploadFile], File(description="Markdown, HTML, or text documents")],
) -> IngestResponse:
    """Accept uploaded documents, store them, and index them in the background.

    Returns as soon as the files are on disk — extraction and embedding are far
    too slow to hold a request open. Poll `GET /jobs/{job_id}` for progress.
    """
    try:
        stored = await save_uploads(files, settings)
    except UnsupportedFormatError as exc:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, str(exc)) from exc
    except UploadTooLargeError as exc:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, str(exc)) from exc
    except PathTraversalError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc

    job = jobs.create(stored)
    background.add_task(run_ingestion, job.job_id, resources, settings)

    return IngestResponse(job_id=job.job_id, documents=stored)


@router.get("/jobs/{job_id}", response_model=JobStatus)
async def job_status(job_id: str, jobs: JobsDep) -> JobStatus:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"no job {job_id!r}")

    return JobStatus(
        job_id=job.job_id,
        state=job.state,
        documents=job.documents,
        processed=job.processed,
        total=job.total,
        error=job.error,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )
