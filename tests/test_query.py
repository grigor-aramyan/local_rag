from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


def test_query_is_not_implemented_yet(client: TestClient) -> None:
    """The retrieval pipeline is build step 10; the contract exists, the body does not."""
    response = client.post("/query", json={"question": "what is this?"})

    assert response.status_code == 501


@pytest.mark.parametrize("question", ["", "   ", "\n\t"])
def test_rejects_a_blank_question(client: TestClient, question: str) -> None:
    assert client.post("/query", json={"question": question}).status_code == 422


def test_rejects_an_overlong_question(client: TestClient) -> None:
    assert client.post("/query", json={"question": "x" * 5_000}).status_code == 422


def test_rejects_an_out_of_range_top_k(client: TestClient) -> None:
    response = client.post("/query", json={"question": "hi", "top_k": 10_000})

    assert response.status_code == 422
