# src/api/app.py
from __future__ import annotations

from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from src import config
from src.retriever import VectorRetriever


app = FastAPI(
    title="Simple RAG Demo",
    description="Minimal API with /health and /search powered by a vector retriever.",
    version="0.1.0",
)


class SearchRequest(BaseModel):
    """Запрос к эндпоинту /search."""
    query: str
    top_k: int = 5


class SearchHit(BaseModel):
    """Одна найденная «подсказка» из индекса."""
    score: float
    text: str
    doc_id: Optional[str] = None
    chunk_id: Optional[int] = None
    source_path: Optional[str] = None


class SearchResponse(BaseModel):
    """Ответ /search: исходный запрос + список найденных фрагментов."""
    query: str
    top_k: int
    hits: List[SearchHit]


# Ленивая инициализация ретривера, чтобы не создавать его при каждом запросе
_retriever: Optional[VectorRetriever] = None


def get_retriever() -> VectorRetriever:
    """
    Возвращает singleton-экземпляр VectorRetriever.

    Использует конфиг src.config:
    - config.INDEX_PATH
    - config.METADATA_PATH

    При первом вызове загружает FAISS-индекс и метаданные с диска.
    """
    global _retriever
    if _retriever is None:
        _retriever = VectorRetriever.from_config(config)
    return _retriever


@app.get("/health")
def health() -> dict:
    """Простой health-check, удобно для smoke-тестов и мониторинга."""
    return {"status": "ok"}


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    """
    Эндпоинт поиска по векторному индексу.

    Берёт текстовый запрос, запускает VectorRetriever.search(...) и
    возвращает топ-k фрагментов с оценкой релевантности и базовыми метаданными.
    """
    retriever = get_retriever()
    results = retriever.search(request.query, top_k=request.top_k)

    hits: List[SearchHit] = []
    for r in results:
        meta = r.get("metadata", {}) if isinstance(r, dict) else {}
        hits.append(
            SearchHit(
                score=float(r.get("score", 0.0)),
                text=str(meta.get("text", "")),
                doc_id=meta.get("doc_id"),
                chunk_id=meta.get("chunk_id"),
                source_path=meta.get("source_path"),
            )
        )

    return SearchResponse(
        query=request.query,
        top_k=request.top_k,
        hits=hits,
    )
