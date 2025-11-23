# tests/test_api_smoke.py
from __future__ import annotations

"""
Smoke tests for the FastAPI application in src.api.app.

We do not rely on a real FAISS index or embedding model here.
Instead, we monkeypatch `get_retriever` to return a tiny in-memory
dummy retriever so that /search and /chat can be exercised safely.
"""

from typing import List

from fastapi.testclient import TestClient

from src.api.app import app, get_retriever


class DummySearchResult:
    """Minimal search result object compatible with the API schema."""

    def __init__(self, doc_id: str, chunk_id: int, score: float, text: str) -> None:
        self.doc_id = doc_id
        self.chunk_id = chunk_id
        self.score = score
        self.text = text


class DummyRetriever:
    """
    Very small in-memory retriever used for tests.

    The `search` method ignores embeddings and just returns a fixed
    set of results so that the API shape can be tested.
    """

    def search(self, query: str, top_k: int | None = None, with_text: bool = True) -> List[DummySearchResult]:
        k = top_k or 3
        base_text = f"Dummy result for query: {query}"
        results: List[DummySearchResult] = []
        for i in range(k):
            results.append(
                DummySearchResult(
                    doc_id=f"doc_{i}",
                    chunk_id=i,
                    score=1.0 - 0.1 * i,
                    text=base_text + f" (#{i})",
                )
            )
        return results


def test_health_ok() -> None:
    """
    /health should respond with HTTP 200 and {"status": "ok"}.
    """
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, dict)
    assert data.get("status") == "ok"


def test_search_uses_dummy_retriever(monkeypatch) -> None:
    """
    /search should return results from the dummy retriever.

    We monkeypatch `get_retriever` so that the FastAPI app does not
    try to load a real FAISS index or embedding model during tests.
    """

    def fake_get_retriever() -> DummyRetriever:
        return DummyRetriever()

    # override the cached getter
    monkeypatch.setattr("src.api.app.get_retriever", fake_get_retriever)

    client = TestClient(app)

    payload = {"query": "test query", "top_k": 2}
    resp = client.post("/search", json=payload)

    assert resp.status_code == 200
    data = resp.json()
    assert data["query"] == "test query"
    results = data["results"]
    assert isinstance(results, list)
    assert len(results) == 2

    first = results[0]
    assert "doc_id" in first
    assert "chunk_id" in first
    assert "score" in first
    assert "text" in first
    assert first["doc_id"] == "doc_0"
    assert "Dummy result for query" in first["text"]


def test_chat_stub_returns_context(monkeypatch) -> None:
    """
    /chat should return a stub answer and include retrieved context.

    We again monkeypatch `get_retriever` to use DummyRetriever.
    """

    def fake_get_retriever() -> DummyRetriever:
        return DummyRetriever()

    monkeypatch.setattr("src.api.app.get_retriever", fake_get_retriever)

    client = TestClient(app)

    payload = {"query": "explain SSH hardening", "top_k": 3}
    resp = client.post("/chat", json=payload)

    assert resp.status_code == 200
    data = resp.json()

    assert data["query"] == "explain SSH hardening"
    assert isinstance(data["answer"], str)
    assert "stub chat endpoint" in data["answer"].lower()

    context = data["context"]
    assert isinstance(context, list)
    assert len(context) == 3

    # basic shape of context items
    ctx0 = context[0]
    for key in ("doc_id", "chunk_id", "score", "text"):
        assert key in ctx0
