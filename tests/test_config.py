from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.config import Settings


def test_defaults_match_the_brief() -> None:
    settings = Settings()

    assert settings.chunk_size == 500
    assert settings.chunk_overlap == 50
    assert settings.top_k == 50
    assert settings.rerank_top_n == 5
    assert settings.embedding_model == "BAAI/bge-small-en-v1.5"
    assert settings.embedding_dim == 384
    assert settings.llm_model == "claude-opus-5"
    assert settings.rerank_enabled is True


def test_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ValidationError, match="chunk_overlap"):
        Settings(chunk_size=100, chunk_overlap=100)


def test_rerank_depth_cannot_exceed_retrieval_depth() -> None:
    with pytest.raises(ValidationError, match="rerank_top_n"):
        Settings(top_k=10, rerank_top_n=20)


def test_reads_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("CHUNK_SIZE", "800")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-value")

    settings = Settings()

    assert settings.chunk_size == 800
    assert settings.anthropic_api_key is not None
    assert settings.anthropic_api_key.get_secret_value() == "sk-ant-test-value"


def test_api_key_is_not_exposed_by_repr(monkeypatch) -> None:
    """A settings dump reaching a log line must not leak the key."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret-value")

    settings = Settings()

    assert "sk-ant-secret-value" not in repr(settings)
    assert "sk-ant-secret-value" not in str(settings.model_dump())


@pytest.mark.parametrize("field, value", [("chunk_size", 0), ("top_k", 0), ("rerank_top_n", 0)])
def test_sizes_must_be_positive(field: str, value: int) -> None:
    with pytest.raises(ValidationError):
        Settings(**{field: value})


def test_reranking_can_be_disabled_from_the_environment(monkeypatch) -> None:
    monkeypatch.setenv("RERANK_ENABLED", "false")

    assert Settings().rerank_enabled is False
