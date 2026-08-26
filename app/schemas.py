from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

JobState = Literal["pending", "running", "completed", "failed"]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    embedder: bool
    reranker: bool
    database: bool


class IngestResponse(BaseModel):
    """Returned immediately; the work continues in the background."""

    job_id: str
    documents: list[str] = Field(description="Stored filenames, after sanitizing.")


class JobStatus(BaseModel):
    job_id: str
    state: JobState
    documents: list[str]
    processed: int
    total: int
    error: str | None = None
    created_at: datetime
    updated_at: datetime


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4_000)
    top_k: int | None = Field(default=None, gt=0, le=200)

    @field_validator("question")
    @classmethod
    def _reject_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("question must not be blank")
        return stripped


class Citation(BaseModel):
    marker: int
    source: str
    page: int | None = None


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
