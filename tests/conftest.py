from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.jobs import JobStore
from app.services.registry import Resources


@pytest.fixture
def documents_dir(tmp_path: Path) -> Path:
    path = tmp_path / "documents"
    path.mkdir()
    return path


@pytest.fixture
def make_settings(tmp_path: Path, documents_dir: Path) -> Callable[..., Settings]:
    def _make(**overrides: object) -> Settings:
        defaults: dict[str, object] = {
            "lancedb_path": tmp_path / "lancedb",
            "documents_path": documents_dir,
            "jobs_db_path": tmp_path / "lancedb" / "jobs.db",
        }
        return Settings(**{**defaults, **overrides})

    return _make


@pytest.fixture
def settings(make_settings: Callable[..., Settings]) -> Settings:
    return make_settings()


@pytest.fixture
def make_client(monkeypatch, make_settings) -> Iterator[Callable[..., TestClient]]:
    """Build a TestClient with the real app but stand-in models.

    Loading the real ONNX models costs seconds and hundreds of MB; nothing under
    test here calls into them, only checks that they were populated.
    """
    exit_stack: list[TestClient] = []

    def _make(*, resources: Resources | None = None, **overrides: object) -> TestClient:
        import app.main as main

        settings = make_settings(**overrides)
        built = (
            resources
            if resources is not None
            else Resources(
                embedder=object(),
                reranker=object(),
                db=object(),
                jobs=JobStore(settings.jobs_db_path),
            )
        )
        if built.jobs is None:
            built.jobs = JobStore(settings.jobs_db_path)

        monkeypatch.setattr(main, "get_settings", lambda: settings)
        monkeypatch.setattr(main, "build_resources", lambda _settings: built)

        client = TestClient(main.app)
        client.__enter__()
        exit_stack.append(client)
        return client

    yield _make

    for client in reversed(exit_stack):
        client.__exit__(None, None, None)


@pytest.fixture
def client(make_client: Callable[..., TestClient]) -> TestClient:
    return make_client()
