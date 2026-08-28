from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from app.config import Settings
from app.services.generation import NO_CONTEXT_SYSTEM_PROMPT, SYSTEM_PROMPT
from app.services.query import NO_CONTEXT_WARNING, run_query
from app.services.registry import Resources
from tests.conftest import FakeAnthropicClient, FakeEmbedder, FakeReranker


def make_resources(db: Any, embedder: Any, llm_client: Any, reranker: Any = None) -> Resources:
    return Resources(embedder=embedder, reranker=reranker, db=db, llm_client=llm_client)


def candidate(source: str, text: str = "text", page: int | None = None) -> dict[str, Any]:
    return {
        "id": f"{source}:0",
        "text": text,
        "source": source,
        "page": page,
        "content_hash": "0" * 64,
    }


@pytest.fixture
def small_settings(make_settings: Callable[..., Settings]) -> Settings:
    return make_settings(embedding_dim=4)


class TestEmptyRetrieval:
    def test_returns_a_warning_when_nothing_is_retrieved(
        self,
        blank_db: Any,
        small_settings: Settings,
        fake_embedder: FakeEmbedder,
        monkeypatch: pytest.MonkeyPatch,
        make_fake_message: Callable[..., Any],
    ) -> None:
        monkeypatch.setattr("app.services.query.hybrid_search", lambda *a, **k: [])
        client = FakeAnthropicClient(make_fake_message())
        resources = make_resources(blank_db, fake_embedder, client)

        response = run_query("what is this?", None, resources, small_settings)

        assert response.warning == NO_CONTEXT_WARNING
        assert response.citations == []

    def test_generation_gets_no_documents_and_the_no_context_prompt(
        self,
        blank_db: Any,
        small_settings: Settings,
        fake_embedder: FakeEmbedder,
        monkeypatch: pytest.MonkeyPatch,
        make_fake_message: Callable[..., Any],
    ) -> None:
        monkeypatch.setattr("app.services.query.hybrid_search", lambda *a, **k: [])
        client = FakeAnthropicClient(make_fake_message())
        resources = make_resources(blank_db, fake_embedder, client)

        run_query("what is this?", None, resources, small_settings)

        sent = client.calls[0]
        assert sent["system"] == NO_CONTEXT_SYSTEM_PROMPT
        assert sent["messages"][0]["content"] == "what is this?"

    def test_the_reranker_is_never_called(
        self,
        blank_db: Any,
        small_settings: Settings,
        fake_embedder: FakeEmbedder,
        monkeypatch: pytest.MonkeyPatch,
        make_fake_message: Callable[..., Any],
    ) -> None:
        monkeypatch.setattr("app.services.query.hybrid_search", lambda *a, **k: [])
        client = FakeAnthropicClient(make_fake_message())
        reranker = FakeReranker()
        resources = make_resources(blank_db, fake_embedder, client, reranker)

        run_query("what is this?", None, resources, small_settings)

        assert reranker.calls == []


class TestPopulatedRetrieval:
    def test_no_warning_when_something_is_retrieved(
        self,
        blank_db: Any,
        small_settings: Settings,
        fake_embedder: FakeEmbedder,
        monkeypatch: pytest.MonkeyPatch,
        make_fake_message: Callable[..., Any],
    ) -> None:
        monkeypatch.setattr("app.services.query.hybrid_search", lambda *a, **k: [candidate("a.md")])
        client = FakeAnthropicClient(make_fake_message())
        resources = make_resources(blank_db, fake_embedder, client, FakeReranker())

        response = run_query("q", None, resources, small_settings)

        assert response.warning is None

    def test_documents_are_sent_using_the_grounded_prompt(
        self,
        blank_db: Any,
        small_settings: Settings,
        fake_embedder: FakeEmbedder,
        monkeypatch: pytest.MonkeyPatch,
        make_fake_message: Callable[..., Any],
    ) -> None:
        monkeypatch.setattr("app.services.query.hybrid_search", lambda *a, **k: [candidate("a.md")])
        client = FakeAnthropicClient(make_fake_message())
        resources = make_resources(blank_db, fake_embedder, client, FakeReranker())

        run_query("q", None, resources, small_settings)

        assert client.calls[0]["system"] == SYSTEM_PROMPT

    def test_reranking_reorders_candidates_before_generation(
        self,
        blank_db: Any,
        make_settings: Callable[..., Settings],
        fake_embedder: FakeEmbedder,
        monkeypatch: pytest.MonkeyPatch,
        make_fake_message: Callable[..., Any],
    ) -> None:
        settings = make_settings(embedding_dim=4, rerank_enabled=True, rerank_top_n=5, top_k=10)
        candidates = [candidate("low.md", text="low"), candidate("high.md", text="high")]
        monkeypatch.setattr("app.services.query.hybrid_search", lambda *a, **k: candidates)
        client = FakeAnthropicClient(make_fake_message())
        reranker = FakeReranker(scores={"low": 0.1, "high": 0.9})
        resources = make_resources(blank_db, fake_embedder, client, reranker)

        run_query("q", None, resources, settings)

        content = client.calls[0]["messages"][0]["content"]
        assert content[0]["title"] == "high.md"
        assert content[1]["title"] == "low.md"

    def test_reranking_is_skipped_when_disabled(
        self,
        blank_db: Any,
        make_settings: Callable[..., Settings],
        fake_embedder: FakeEmbedder,
        monkeypatch: pytest.MonkeyPatch,
        make_fake_message: Callable[..., Any],
    ) -> None:
        settings = make_settings(embedding_dim=4, rerank_enabled=False, rerank_top_n=5, top_k=10)
        candidates = [candidate("first.md"), candidate("second.md")]
        monkeypatch.setattr("app.services.query.hybrid_search", lambda *a, **k: candidates)
        client = FakeAnthropicClient(make_fake_message())
        reranker = FakeReranker()
        resources = make_resources(blank_db, fake_embedder, client, reranker)

        run_query("q", None, resources, settings)

        assert reranker.calls == []
        content = client.calls[0]["messages"][0]["content"]
        assert content[0]["title"] == "first.md"

    def test_rerank_top_n_caps_the_documents_sent_to_generation(
        self,
        blank_db: Any,
        make_settings: Callable[..., Settings],
        fake_embedder: FakeEmbedder,
        monkeypatch: pytest.MonkeyPatch,
        make_fake_message: Callable[..., Any],
    ) -> None:
        settings = make_settings(embedding_dim=4, rerank_enabled=False, rerank_top_n=1, top_k=10)
        candidates = [candidate("a.md"), candidate("b.md"), candidate("c.md")]
        monkeypatch.setattr("app.services.query.hybrid_search", lambda *a, **k: candidates)
        client = FakeAnthropicClient(make_fake_message())
        resources = make_resources(blank_db, fake_embedder, client, FakeReranker())

        run_query("q", None, resources, settings)

        content = client.calls[0]["messages"][0]["content"]
        # one document block plus the trailing question block
        assert len(content) == 2

    def test_top_k_override_is_forwarded_to_hybrid_search(
        self,
        blank_db: Any,
        small_settings: Settings,
        fake_embedder: FakeEmbedder,
        monkeypatch: pytest.MonkeyPatch,
        make_fake_message: Callable[..., Any],
    ) -> None:
        recorded: dict[str, Any] = {}

        def fake_hybrid_search(
            table: Any, vector: Any, text: str, limit: int
        ) -> list[dict[str, Any]]:
            recorded["limit"] = limit
            return []

        monkeypatch.setattr("app.services.query.hybrid_search", fake_hybrid_search)
        client = FakeAnthropicClient(make_fake_message())
        resources = make_resources(blank_db, fake_embedder, client)

        run_query("q", 7, resources, small_settings)

        assert recorded["limit"] == 7

    def test_settings_top_k_is_used_when_no_override_is_given(
        self,
        blank_db: Any,
        small_settings: Settings,
        fake_embedder: FakeEmbedder,
        monkeypatch: pytest.MonkeyPatch,
        make_fake_message: Callable[..., Any],
    ) -> None:
        recorded: dict[str, Any] = {}

        def fake_hybrid_search(
            table: Any, vector: Any, text: str, limit: int
        ) -> list[dict[str, Any]]:
            recorded["limit"] = limit
            return []

        monkeypatch.setattr("app.services.query.hybrid_search", fake_hybrid_search)
        client = FakeAnthropicClient(make_fake_message())
        resources = make_resources(blank_db, fake_embedder, client)

        run_query("q", None, resources, small_settings)

        assert recorded["limit"] == small_settings.top_k

    def test_citations_from_generation_flow_into_the_response(
        self,
        blank_db: Any,
        small_settings: Settings,
        fake_embedder: FakeEmbedder,
        monkeypatch: pytest.MonkeyPatch,
        make_fake_message: Callable[..., Any],
        make_text_block: Callable[..., Any],
        make_citation: Callable[..., Any],
    ) -> None:
        monkeypatch.setattr("app.services.query.hybrid_search", lambda *a, **k: [candidate("a.md")])
        message = make_fake_message(make_text_block("claim", citations=[make_citation(0)]))
        client = FakeAnthropicClient(message)
        resources = make_resources(blank_db, fake_embedder, client, FakeReranker())

        response = run_query("q", None, resources, small_settings)

        assert response.citations[0].source == "a.md"
        assert response.warning is None
