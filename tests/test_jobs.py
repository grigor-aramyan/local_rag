from __future__ import annotations

from pathlib import Path

import pytest

from app.jobs import JobStore


@pytest.fixture
def store(tmp_path: Path) -> JobStore:
    return JobStore(tmp_path / "state" / "jobs.db")


def test_creates_the_database_and_its_parent_directory(tmp_path: Path) -> None:
    JobStore(tmp_path / "state" / "jobs.db")

    assert (tmp_path / "state" / "jobs.db").exists()


def test_a_new_job_starts_pending(store: JobStore) -> None:
    job = store.create(["a.md", "b.md"])

    assert job.state == "pending"
    assert job.documents == ["a.md", "b.md"]
    assert job.total == 2
    assert job.processed == 0
    assert job.error is None


def test_job_ids_are_unique(store: JobStore) -> None:
    ids = {store.create(["a.md"]).job_id for _ in range(50)}

    assert len(ids) == 50


def test_round_trips_through_storage(store: JobStore) -> None:
    created = store.create(["a.md"])

    fetched = store.get(created.job_id)

    assert fetched == created


def test_unknown_job_is_none(store: JobStore) -> None:
    assert store.get("does-not-exist") is None


def test_progress_updates_are_visible(store: JobStore) -> None:
    job = store.create(["a.md", "b.md"])

    store.mark_running(job.job_id)
    store.set_progress(job.job_id, processed=1)
    fetched = store.get(job.job_id)

    assert fetched is not None
    assert fetched.state == "running"
    assert fetched.processed == 1
    assert fetched.updated_at >= job.updated_at


def test_completion_and_failure_are_terminal_states(store: JobStore) -> None:
    done = store.create(["a.md"])
    broken = store.create(["b.md"])

    store.mark_completed(done.job_id)
    store.mark_failed(broken.job_id, "extraction blew up")

    assert store.get(done.job_id).state == "completed"  # type: ignore[union-attr]
    assert store.get(broken.job_id).state == "failed"  # type: ignore[union-attr]
    assert store.get(broken.job_id).error == "extraction blew up"  # type: ignore[union-attr]


def test_state_survives_reopening_the_database(tmp_path: Path) -> None:
    path = tmp_path / "jobs.db"
    job_id = JobStore(path).create(["a.md"]).job_id

    reopened = JobStore(path).get(job_id)

    assert reopened is not None
    assert reopened.documents == ["a.md"]


def test_a_restart_does_not_strand_unfinished_jobs(tmp_path: Path) -> None:
    """Background tasks live in-process, so nothing can still be in flight at startup.

    A crash mid-ingest must not leave a job reporting `running` forever, and a
    job that never got off `pending` is equally never going to run.
    """
    path = tmp_path / "jobs.db"
    first = JobStore(path)
    running = first.create(["a.md"])
    pending = first.create(["b.md"])
    finished = first.create(["c.md"])
    first.mark_running(running.job_id)
    first.mark_completed(finished.job_id)

    second = JobStore(path)
    recovered = second.fail_interrupted_jobs()

    assert recovered == 2
    assert second.get(running.job_id).state == "failed"  # type: ignore[union-attr]
    assert second.get(pending.job_id).state == "failed"  # type: ignore[union-attr]
    assert "interrupted" in second.get(running.job_id).error.lower()  # type: ignore[union-attr]
    assert second.get(finished.job_id).state == "completed"  # type: ignore[union-attr]
