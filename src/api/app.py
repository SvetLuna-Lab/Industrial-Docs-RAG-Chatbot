from __future__ import annotations

from typing import List, Optional, Any

from fastapi import FastAPI
from pydantic import BaseModel

from src.retriever import VectorRetriever

app = FastAPI(
    title="RAG Vector API",
    description="Minimal FastAPI app with /health, /search and /chat endpoints.",
    version="0.1.0",
)


# ---------- Pydantic models ----------


class SearchRequest(BaseModel):
    """Request body for /search endpoint."""
    query: str
    top_k: int = 5


class SearchResult(BaseModel):
    """
    One retrieved chunk as returned by /search and /chat.

    Fields match tests/test_api_smoke.py expectations:
    - doc_id
    - chunk_id
    - score
    - text
    """
    doc_id: str
    chunk_id: int
    score: float
    text: str


class SearchResponse(BaseModel):
    """Response for /search: original query + list of results."""
    query: str
    results: List[SearchResult]


class ChatRequest(BaseModel):
    """Request body for /chat endpoint."""
    query: str
    top_k: int = 3


class ChatResponse(BaseModel):
    """
    Response for /chat:

    - answer: stub answer
    - context: list of SearchResult used as context
    """
    query: str
    answer: str
    context: List[SearchResult]


# ---------- Lazy retriever singleton ----------

_retriever: Optional[VectorRetriever] = None


def get_retriever() -> VectorRetriever:
    """
    Returns a singleton instance of VectorRetriever.

    In production:
    - loads FAISS index and metadata via VectorRetriever.from_config().

    In tests:
    - tests/test_api_smoke.py monkeypatches this function to return DummyRetriever.
    """
    global _retriever
    if _retriever is None:
        _retriever = VectorRetriever.from_config()
    return _retriever


# ---------- Helpers to adapt result objects ----------


def _get_attr_or_key(obj: Any, name: str, default=None):
    """
    Universal accessor for result fields:
    - if obj is dict, use obj[name];
    - if obj has attribute `name`, use getattr(obj, name);
    - otherwise return default.
    """
    if isinstance(obj, dict):
        return obj.get(name, default)
    if hasattr(obj, name):
        return getattr(obj, name)
    return default


def _result_to_search_result(r: Any) -> SearchResult:
    """
    Convert a retriever result (dict or object with attributes)
    to a Pydantic SearchResult.

    Compatible with:
    - real VectorRetriever results
    - DummyRetriever.DummySearchResult from tests/test_api_smoke.py
    """
    doc_id = _get_attr_or_key(r, "doc_id", "unknown")
    chunk_id = int(_get_attr_or_key(r, "chunk_id", 0))
    score = float(_get_attr_or_key(r, "score", 0.0))

    # Try direct "text" first
    text = _get_attr_or_key(r, "text", None)

    # Fallback: many retrievers keep chunk text inside r["metadata"]["text"]
    if (text is None or text == "") and isinstance(r, dict):
        meta = r.get("metadata") or {}
        if isinstance(meta, dict):
            text = meta.get("text", "")

    if text is None:
        text = ""

    return SearchResult(
        doc_id=str(doc_id),
        chunk_id=chunk_id,
        score=score,
        text=str(text),
    )


# ---------- Endpoints ----------


@app.get("/health")
def health() -> dict:
    """Simple health-check used by smoke tests and monitoring."""
    return {"status": "ok"}


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    """
    Vector search endpoint.

    Returns:
    {
      "query": "...",
      "results": [ {doc_id, chunk_id, score, text}, ... ]
    }

    Response shape matches tests/test_api_smoke.py.
    """
    retriever = get_retriever()
    raw_results = retriever.search(request.query, top_k=request.top_k)

    results = [_result_to_search_result(r) for r in raw_results]

    return SearchResponse(
        query=request.query,
        results=results,
    )


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """
    Stub chat endpoint on top of the retriever.

    tests/test_api_smoke.py expects:
    - `answer` contains substring "stub chat endpoint" (case-insensitive)
    - `context` is a list of items with doc_id, chunk_id, score, text.
    """
    retriever = get_retriever()
    raw_results = retriever.search(request.query, top_k=request.top_k)
    context = [_result_to_search_result(r) for r in raw_results]

    # IMPORTANT: include "stub chat endpoint" for the test assertion
    answer = (
        "This is a stub chat endpoint response. "
        f"Your query was: '{request.query}'. "
        "In a full RAG pipeline, this would call an LLM with the retrieved context."
    )

    return ChatResponse(
        query=request.query,
        answer=answer,
        context=context,
    )

