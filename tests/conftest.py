from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from tokenizers import Tokenizer, models, pre_tokenizers

from app.config import Settings
from app.jobs import JobStore
from app.services.registry import Resources


@pytest.fixture
def documents_dir(tmp_path: Path) -> Path:
    path = tmp_path / "documents"
    path.mkdir()
    return path


def _word_tokenizer() -> Tokenizer:
    """A whitespace tokenizer that reports real character offsets.

    Chunking only needs `encode(...).offsets`, so a vocab-free stand-in exercises
    the windowing exactly while keeping the suite free of the ONNX model.
    """
    tokenizer = Tokenizer(models.WordLevel(vocab={"[UNK]": 0}, unk_token="[UNK]"))  # noqa: S106
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    return tokenizer


@pytest.fixture
def word_tokenizer() -> Tokenizer:
    return _word_tokenizer()


@pytest.fixture
def truncating_tokenizer() -> Tokenizer:
    """Stands in for the embedder's own tokenizer, which truncates at 512 tokens."""
    tokenizer = _word_tokenizer()
    tokenizer.enable_truncation(max_length=5)
    return tokenizer


class FakeEmbedder:
    """Deterministic stand-in for `TextEmbedding` — same call shape, no ONNX session."""

    def __init__(self, dim: int = 4) -> None:
        self.dim = dim
        self.batches: list[list[str]] = []

    def embed(self, documents: Sequence[str], batch_size: int = 256, **kwargs: Any) -> list[Any]:
        import numpy as np

        texts = list(documents)
        self.batches.append(texts)
        return [
            np.frombuffer(
                hashlib.sha256(text.encode()).digest()[: self.dim * 4], dtype=np.float32
            ).copy()
            for text in texts
        ]


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


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
def blank_db(tmp_path: Path) -> Any:
    """A real, empty LanceDB connection.

    LanceDB is embedded and creates nothing until a table is written, so tests use
    the real handle rather than a double — the startup config guard inspects it.
    """
    import lancedb

    path = tmp_path / "lancedb"
    path.mkdir(parents=True, exist_ok=True)
    return lancedb.connect(str(path))


@pytest.fixture
def make_client(monkeypatch, make_settings, blank_db) -> Iterator[Callable[..., TestClient]]:
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
                db=blank_db,
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
