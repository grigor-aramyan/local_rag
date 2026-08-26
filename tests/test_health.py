from __future__ import annotations

from collections.abc import Callable

import pytest
from fastapi.testclient import TestClient

from app.services.registry import Resources


def test_health_reports_ok_when_everything_loaded(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "embedder": True,
        "reranker": True,
        "database": True,
    }


@pytest.mark.parametrize("missing", ["embedder", "reranker", "db"])
def test_health_fails_when_a_component_is_missing(
    make_client: Callable[..., TestClient], missing: str
) -> None:
    """`/health` must prove the models loaded, not merely that the process is up."""
    loaded = {"embedder": object(), "reranker": object(), "db": object()}
    loaded[missing] = None

    client = make_client(resources=Resources(**loaded))
    response = client.get("/health")

    assert response.status_code == 503
    assert response.json()["status"] == "degraded"


def test_health_names_the_component_that_failed(make_client: Callable[..., TestClient]) -> None:
    client = make_client(resources=Resources(embedder=None, reranker=object(), db=object()))

    body = client.get("/health").json()

    assert body["embedder"] is False
    assert body["reranker"] is True
    assert body["database"] is True
