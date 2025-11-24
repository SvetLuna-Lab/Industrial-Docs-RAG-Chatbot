from __future__ import annotations

from typing import List, Optional

import os
import httpx
from fastapi import FastAPI
from pydantic import BaseModel

from src import config
from src.retriever import VectorRetriever


app = FastAPI(
    title="RAG Vector API",
    description="Minimal FastAPI app with /health, /search and /chat endpoints.",
    version="0.2.0",
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

    - answer: LLM answer or stub
    - context: list of SearchResult used as context
    """
    query: str
    answer: str
    context: List[SearchResult]


# ---------- Lazy singletons ----------

_retriever: Optional[VectorRetriever] = None
_app_config: Optional[config.AppConfig] = None


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


def get_app_config() -> config.AppConfig:
    """
    Lazily load application configuration.

    Used for LLM backend settings (provider, model_name, api_base, api_key_env).
    """
    global _app_config
    if _app_config is None:
        _app_config = config.load_app_config()
    return _app_config


# ---------- Helpers to adapt result objects ----------


def _get_attr_or_key(obj, name: str, default=None):
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


def _result_to_search_result(r) -> SearchResult:
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
    text = _get_attr_or_key(r, "text", "")

    return SearchResult(
        doc_id=str(doc_id),
        chunk_id=chunk_id,
        score=score,
        text=str(text),
    )


# ---------- LLM helpers ----------


def _build_llm_prompt(query: str, context: List[SearchResult]) -> str:
    """
    Build a simple text prompt for the LLM based on user query and retrieved context.
    """
    lines: List[str] = []
    lines.append(f"User query:\n{query}\n")
    if context:
        lines.append("Relevant documentation chunks:\n")
        for item in context:
            snippet = item.text.replace("\n", " ").strip()
            lines.append(
                f"- doc_id={item.doc_id}, chunk_id={item.chunk_id}, score={item.score:.3f}\n"
                f"  {snippet}"
            )
    else:
        lines.append("No relevant documentation chunks were found in the index.\n")

    lines.append(
        "\nUsing the documentation above, answer the user's query in a concise, practical way. "
        "If the answer is not present in the documents, clearly say that the documentation does not cover this question."
    )
    return "\n".join(lines)


def _call_llm_via_httpx(query: str, context: List[SearchResult]) -> str:
    """
    Minimal LLM backend using an OpenAI-compatible HTTP API.

    Uses config.llm.* settings:
      - provider: 'stub' (no call) or 'openai' (or any OpenAI-compatible endpoint)
      - model_name: target model ID
      - api_base: base URL, e.g. 'https://api.openai.com/v1'
      - api_key_env: name of env var with the API key

    If provider == 'stub' or API is misconfigured/unavailable, falls back
    to a stub answer mentioning that this is a stub chat endpoint.
    """
    cfg = get_app_config().llm

    # Stub mode: keep old behaviour (important for tests and local runs)
    if cfg.provider.lower() == "stub":
        return (
            "This is a stub chat endpoint response. "
            f"Your query was: '{query}'. "
            "In a full RAG pipeline, this would call an LLM with the retrieved context."
        )

    api_key = os.getenv(cfg.api_key_env, "")
    if not api_key:
        return (
            "This is a stub chat endpoint response. "
            "LLM provider is configured but API key is missing. "
            "Please set the environment variable "
            f"{cfg.api_key_env} to enable real LLM calls."
        )

    api_base = cfg.api_base or "https://api.openai.com/v1"
    url = api_base.rstrip("/") + "/chat/completions"

    prompt = _build_llm_prompt(query, context)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": cfg.model_name,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an assistant that answers strictly based on the provided "
                    "internal industrial documentation. "
                    "If the answer is not in the documents, say so explicitly."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "temperature": 0.2,
    }

    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            return (
                "This is a stub chat endpoint response. "
                "LLM call returned no choices. Please check your model and request parameters."
            )
        message = choices[0].get("message", {})
        content = message.get("content") or ""
        if not isinstance(content, str) or not content.strip():
            return (
                "This is a stub chat endpoint response. "
                "LLM call returned an empty answer. Please verify the API configuration."
            )
        return content.strip()
    except Exception as exc:
        # Fail-safe: never crash the API, just fall back to stub behaviour
        return (
            "This is a stub chat endpoint response. "
            "An error occurred while calling the LLM backend: "
            f"{type(exc).__name__}. "
            "Check logs and LLM API configuration."
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
    Chat endpoint on top of the retriever.

    Behaviour:
    - always performs retrieval to get context;
    - if llm.provider == 'stub' → returns the historical stub answer
      (must contain substring 'stub chat endpoint' for tests);
    - otherwise, calls an OpenAI-compatible LLM backend using httpx.
    """
    retriever = get_retriever()
    raw_results = retriever.search(request.query, top_k=request.top_k)
    context = [_result_to_search_result(r) for r in raw_results]

    answer = _call_llm_via_httpx(request.query, context)

    return ChatResponse(
        query=request.query,
        answer=answer,
        context=context,
    )
