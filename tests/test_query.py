from __future__ import annotations

from collections.abc import Callable
from typing import Any

import anthropic
import httpx2
import pytest
from fastapi.testclient import TestClient

from app.services.registry import Resources
from tests.conftest import FakeAnthropicClient, FakeEmbedder, FakeReranker


def make_query_resources(
    db: Any, *, embedder: Any = None, reranker: Any = None, llm_client: Any = None
) -> Resources:
    return Resources(
        embedder=embedder or FakeEmbedder(),
        reranker=reranker or FakeReranker(),
        db=db,
        llm_client=llm_client,
    )


@pytest.mark.parametrize("question", ["", "   ", "\n\t"])
def test_rejects_a_blank_question(client: TestClient, question: str) -> None:
    assert client.post("/query", json={"question": question}).status_code == 422


def test_rejects_an_overlong_question(client: TestClient) -> None:
    assert client.post("/query", json={"question": "x" * 5_000}).status_code == 422


def test_rejects_an_out_of_range_top_k(client: TestClient) -> None:
    response = client.post("/query", json={"question": "hi", "top_k": 10_000})

    assert response.status_code == 422


class TestSuccessfulQuery:
    def test_returns_the_generated_answer_and_citations(
        self,
        make_client: Callable[..., TestClient],
        blank_db: Any,
        monkeypatch: pytest.MonkeyPatch,
        make_fake_message: Callable[..., Any],
        make_text_block: Callable[..., Any],
        make_citation: Callable[..., Any],
    ) -> None:
        monkeypatch.setattr(
            "app.services.query.hybrid_search",
            lambda *a, **k: [
                {
                    "id": "a.md:0",
                    "text": "the sky is blue",
                    "source": "a.md",
                    "page": None,
                    "content_hash": "0" * 64,
                }
            ],
        )
        message = make_fake_message(
            make_text_block("the sky is blue", citations=[make_citation(0)])
        )
        resources = make_query_resources(blank_db, llm_client=FakeAnthropicClient(message))
        test_client = make_client(resources=resources)

        response = test_client.post("/query", json={"question": "what color is the sky?"})

        assert response.status_code == 200
        body = response.json()
        assert body["answer"] == "the sky is blue"
        assert body["citations"] == [{"marker": 1, "source": "a.md", "page": None}]
        assert body["warning"] is None

    def test_returns_a_warning_when_nothing_is_retrieved(
        self,
        make_client: Callable[..., TestClient],
        blank_db: Any,
        monkeypatch: pytest.MonkeyPatch,
        make_fake_message: Callable[..., Any],
    ) -> None:
        monkeypatch.setattr("app.services.query.hybrid_search", lambda *a, **k: [])
        resources = make_query_resources(
            blank_db, llm_client=FakeAnthropicClient(make_fake_message())
        )
        test_client = make_client(resources=resources)

        response = test_client.post("/query", json={"question": "what is this?"})

        assert response.status_code == 200
        body = response.json()
        assert body["warning"]
        assert body["citations"] == []


class TestGenerationFailures:
    @staticmethod
    def _raise(exc: Exception) -> Callable[..., Any]:
        def _stream(**kwargs: Any) -> Any:
            raise exc

        return _stream

    def test_a_rate_limited_llm_call_is_a_429(
        self, make_client: Callable[..., TestClient], blank_db: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("app.services.query.hybrid_search", lambda *a, **k: [])
        request = httpx2.Request("POST", "http://test")
        response = httpx2.Response(429, request=request)
        error = anthropic.RateLimitError("rate limited", response=response, body=None)

        llm_client = FakeAnthropicClient(None)
        llm_client.stream = self._raise(error)
        resources = make_query_resources(blank_db, llm_client=llm_client)
        test_client = make_client(resources=resources)

        response = test_client.post("/query", json={"question": "what is this?"})

        assert response.status_code == 429

    def test_a_missing_credential_is_a_503(
        self, make_client: Callable[..., TestClient], blank_db: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """anthropic==1.1.0 raises a bare TypeError here, not a typed exception."""
        monkeypatch.setattr("app.services.query.hybrid_search", lambda *a, **k: [])
        error = TypeError("Could not resolve authentication method. Expected one of api_key...")

        llm_client = FakeAnthropicClient(None)
        llm_client.stream = self._raise(error)
        resources = make_query_resources(blank_db, llm_client=llm_client)
        test_client = make_client(resources=resources)

        response = test_client.post("/query", json={"question": "what is this?"})

        assert response.status_code == 503

    def test_an_unrelated_type_error_is_not_swallowed(
        self, make_client: Callable[..., TestClient], blank_db: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Only the specific auth-resolution TypeError is treated as a config error."""
        monkeypatch.setattr("app.services.query.hybrid_search", lambda *a, **k: [])
        error = TypeError("unexpected keyword argument 'foo'")

        llm_client = FakeAnthropicClient(None)
        llm_client.stream = self._raise(error)
        resources = make_query_resources(blank_db, llm_client=llm_client)
        test_client = make_client(resources=resources)

        with pytest.raises(TypeError, match="unexpected keyword"):
            test_client.post("/query", json={"question": "what is this?"})

    def test_an_unavailable_llm_api_is_a_502(
        self, make_client: Callable[..., TestClient], blank_db: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("app.services.query.hybrid_search", lambda *a, **k: [])
        request = httpx2.Request("POST", "http://test")
        error = anthropic.APIConnectionError(message="connection failed", request=request)

        llm_client = FakeAnthropicClient(None)
        llm_client.stream = self._raise(error)
        resources = make_query_resources(blank_db, llm_client=llm_client)
        test_client = make_client(resources=resources)

        response = test_client.post("/query", json={"question": "what is this?"})

        assert response.status_code == 502
