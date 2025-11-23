# src/api/app.py
from __future__ import annotations

from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from src import config
from src.retriever import VectorRetriever


app = FastAPI(
    title="RAG Vector API",
    description="Minimal FastAPI app with /health, /search and /chat endpoints.",
    version="0.1.0",
)


# ---------- Pydantic models ----------


class SearchRequest(BaseModel):
    """Запрос к эндпоинту /search."""
    query: str
    top_k: int = 5


class SearchResult(BaseModel):
    """
    Один фрагмент контекста в ответе /search и /chat.

    Поля соответствуют тому, что ожидает tests/test_api_smoke.py:
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
    """Ответ /search: исходный запрос и список найденных фрагментов."""
    query: str
    results: List[SearchResult]


class ChatRequest(BaseModel):
    """Запрос к эндпоинту /chat."""
    query: str
    top_k: int = 3


class ChatResponse(BaseModel):
    """
    Ответ /chat:

    - answer: stub-ответ от чата,
    - context: список SearchResult, использованный как контекст.
    """
    query: str
    answer: str
    context: List[SearchResult]


# ---------- Lazy retriever singleton ----------

_retriever: Optional[VectorRetriever] = None


def get_retriever() -> VectorRetriever:
    """
    Возвращает singleton-экземпляр VectorRetriever.

    В боевом режиме:
    - загружает FAISS-индекс и метаданные через VectorRetriever.from_config(config).

    В тестах:
    - monkeypatch в tests/test_api_smoke.py подменяет эту функцию так,
      чтобы возвращался DummyRetriever.
    """
    global _retriever
    if _retriever is None:
        _retriever = VectorRetriever.from_config(config)
    return _retriever


# ---------- Helpers для извлечения полей из результатов ----------


def _get_attr_or_key(obj, name: str, default=None):
    """
    Универсальный доступ к полю результата:
    - если obj — dict, берем obj[name];
    - если у obj есть атрибут name, берем его;
    - иначе возвращаем default.
    """
    if isinstance(obj, dict):
        return obj.get(name, default)
    if hasattr(obj, name):
        return getattr(obj, name)
    return default


def _result_to_search_result(r) -> SearchResult:
    """
    Преобразует объект результата ретривера (dict или объект с атрибутами)
    к Pydantic-модели SearchResult.

    Совместимо как с реальным VectorRetriever, так и с DummyRetriever
    из tests/test_api_smoke.py (DummySearchResult).
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


# ---------- Endpoints ----------


@app.get("/health")
def health() -> dict:
    """Простой health-check для smoke-тестов и мониторинга."""
    return {"status": "ok"}


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    """
    Эндпоинт поиска по векторному индексу.

    Использует get_retriever().search(...) и возвращает:
    {
      "query": "...",
      "results": [ {doc_id, chunk_id, score, text}, ... ]
    }

    Формат строго соответствует ожиданиям tests/test_api_smoke.py.
    """
    retriever = get_retriever()
    # В реальном коде VectorRetriever.search может не иметь параметра with_text,
    # но DummyRetriever в тестах его принимает. Поэтому передадим только то,
    # что гарантированно есть.
    raw_results = retriever.search(request.query, top_k=request.top_k)

    results = [_result_to_search_result(r) for r in raw_results]

    return SearchResponse(
        query=request.query,
        results=results,
    )


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """
    Stub-чат поверх ретривера.

    Тест ожидает, что:
    - ответ содержит строку "stub chat endpoint" (в любом регистре);
    - в поле context лежит список фрагментов, аналогичных /search.
    """
    retriever = get_retriever()
    raw_results = retriever.search(request.query, top_k=request.top_k)

    context = [_result_to_search_result(r) for r in raw_results]

    # Важно: тест ищет подстроку "stub chat endpoint" в answer.lower()
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
