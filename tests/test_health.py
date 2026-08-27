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
    make_client: Callable[..., TestClient], blank_db, missing: str
) -> None:
    """`/health` must prove the models loaded, not merely that the process is up."""
    loaded = {"embedder": object(), "reranker": object(), "db": blank_db}
    loaded[missing] = None

    client = make_client(resources=Resources(**loaded))
    response = client.get("/health")

    assert response.status_code == 503
    assert response.json()["status"] == "degraded"


def test_health_names_the_component_that_failed(
    make_client: Callable[..., TestClient], blank_db
) -> None:
    client = make_client(resources=Resources(embedder=None, reranker=object(), db=blank_db))

    body = client.get("/health").json()

    assert body["embedder"] is False
    assert body["reranker"] is True
    assert body["database"] is True


def test_health_reports_an_incompatible_index_instead_of_serving_it(
    make_client: Callable[..., TestClient], blank_db, settings
) -> None:
    """A config change against an existing index degrades retrieval silently, so refuse loudly."""
    from app.services.vectorstore import record_config

    record_config(blank_db, settings)

    client = make_client(resources=Resources(embedder=object(), reranker=object(), db=blank_db))
    response = client.get("/health")

    assert response.status_code == 200

    mismatched = make_client(
        resources=Resources(embedder=object(), reranker=object(), db=blank_db),
        embedding_dim=768,
    )
    body = mismatched.get("/health")

    assert body.status_code == 503
    assert body.json()["status"] == "degraded"
    assert "embedding_dim" in body.json()["detail"]
