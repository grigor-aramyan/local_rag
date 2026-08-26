from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from app.schemas import JobState

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id     TEXT PRIMARY KEY,
    state      TEXT NOT NULL,
    documents  TEXT NOT NULL,
    processed  INTEGER NOT NULL DEFAULT 0,
    total      INTEGER NOT NULL,
    error      TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""

UNFINISHED: tuple[JobState, ...] = ("pending", "running")
INTERRUPTED_MESSAGE = "interrupted by a restart before it finished"


@dataclass(frozen=True, slots=True)
class Job:
    job_id: str
    state: JobState
    documents: list[str]
    processed: int
    total: int
    error: str | None
    created_at: datetime
    updated_at: datetime


def _now() -> datetime:
    return datetime.now(UTC)


class JobStore:
    """Ingestion job state, persisted to SQLite on the data volume.

    Jobs outlive the request that created them but not the process, so the
    durable copy is what lets `GET /jobs/{id}` answer honestly after a restart
    instead of reporting work that is no longer running.
    """

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._lock = threading.Lock()
        # Ingestion runs on the threadpool while requests read on the event
        # loop, so the connection is shared across threads under `_lock`.
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def create(self, documents: list[str]) -> Job:
        job = Job(
            job_id=uuid.uuid4().hex,
            state="pending",
            documents=list(documents),
            processed=0,
            total=len(documents),
            error=None,
            created_at=_now(),
            updated_at=_now(),
        )
        with self._lock:
            self._conn.execute(
                "INSERT INTO jobs (job_id, state, documents, processed, total, error,"
                " created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    job.job_id,
                    job.state,
                    json.dumps(job.documents),
                    job.processed,
                    job.total,
                    job.error,
                    job.created_at.isoformat(),
                    job.updated_at.isoformat(),
                ),
            )
            self._conn.commit()
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return _to_job(row) if row is not None else None

    def mark_running(self, job_id: str) -> None:
        self._update(job_id, state="running")

    def set_progress(self, job_id: str, processed: int) -> None:
        self._update(job_id, processed=processed)

    def mark_completed(self, job_id: str) -> None:
        self._update(job_id, state="completed")

    def mark_failed(self, job_id: str, error: str) -> None:
        self._update(job_id, state="failed", error=error)

    def fail_interrupted_jobs(self) -> int:
        """Close out jobs left unfinished by a previous process. Returns the count."""
        placeholders = ", ".join("?" for _ in UNFINISHED)
        with self._lock:
            cursor = self._conn.execute(
                f"UPDATE jobs SET state = 'failed', error = ?, updated_at = ?"  # noqa: S608
                f" WHERE state IN ({placeholders})",
                (INTERRUPTED_MESSAGE, _now().isoformat(), *UNFINISHED),
            )
            self._conn.commit()
        return cursor.rowcount

    def _update(self, job_id: str, **fields: object) -> None:
        assignments = ", ".join(f"{name} = ?" for name in fields)
        with self._lock:
            self._conn.execute(
                f"UPDATE jobs SET {assignments}, updated_at = ? WHERE job_id = ?",  # noqa: S608
                (*fields.values(), _now().isoformat(), job_id),
            )
            self._conn.commit()


def _to_job(row: sqlite3.Row) -> Job:
    return Job(
        job_id=row["job_id"],
        state=row["state"],
        documents=json.loads(row["documents"]),
        processed=row["processed"],
        total=row["total"],
        error=row["error"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )
