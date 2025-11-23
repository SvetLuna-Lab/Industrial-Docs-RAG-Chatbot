# src/api/app.py
from __future__ import annotations

"""
Minimal FastAPI app for the Industrial Docs RAG Chatbot.

Endpoints:
- GET /health  – simple health check.
- POST /search – vector search over indexed documentation.
- POST /chat   – stub chat endpoint that uses retrieval only
                 (no real LLM yet, just echoes retrieved context).

This app is intended as a thin HTTP layer on top of VectorRetriever.
It can later be extended with a proper LLM backend.
"""

from functools import lru_cache
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from src.retriever import VectorRetriever


# --------------------------------------------------------------------
# FastAPI app
# --------------------------------------------------------------------

app = FastAPI(
    title="Industrial Docs RAG Chatbot API",
    version="0.1.0",
    description="HTTP API for vector search and retrieval-augmented chat "
                "over internal documentation.",
)


# --------------------------------------------------------------------
# Dependency: retriever singleton
# --------------------------------------------------------------------

@lru_cache()
def get_retriever() -> VectorRetriever:
    """
    Lazily construct a singleton VectorRetriever.

    Using @lru_cache avoids re-loading FAISS index and embedding model
    on every request. The retriever is created on first access and then
    reused for all subsequent calls.
    """
    return VectorRetriever.from_default()


# --------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------


class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = None


class RetrievedChunkModel(BaseModel):
    doc_id: str
    chunk_id: int
    score: float
    text: Optional[str] = None


class SearchResponse(BaseModel):
    query: str
    results: List[RetrievedChunkModel]


class ChatRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5


class ChatResponse(BaseModel):
    query: str
    answer: str
    context: List[RetrievedChunkModel]


# --------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------


@app.get("/health")
def health() -> dict:
    """
    Simple health check endpoint.

    Returns "ok" if the API process is up and dependencies can be imported.
    Note: it does not force index/model loading; that happens on first
    search/chat call via get_retriever().
    """
    return {"status": "ok"}


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    """
    Run vector search over the indexed documentation.

    The implementation:
    - uses VectorRetriever.from_default() via get_retriever();
    - runs search(query, top_k);
    - returns a list of chunks with scores and text snippets.
    """
    retriever = get_retriever()
    results = retriever.search(req.query, top_k=req.top_k, with_text=True)

    return SearchResponse(
        query=req.query,
        results=[
            RetrievedChunkModel(
                doc_id=r.doc_id,
                chunk_id=r.chunk_id,
                score=r.score,
                text=r.text,
            )
            for r in results
        ],
    )


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """
    Stub chat endpoint.

    Current behavior:
    - runs retrieval over documentation with the given query;
    - builds a simple "answer" string that explains this is a stub;
    - returns retrieved chunks as context.

    In a real RAG system this is where an LLM would be called with
    (query + retrieved context) to generate a full answer.
    """
    retriever = get_retriever()
    results = retriever.search(req.query, top_k=req.top_k, with_text=True)

    # Very simple stub answer
    answer = (
        "This is a stub chat endpoint. The system retrieved "
        f"{len(results)} relevant chunks from the documentation. "
        "A real LLM backend can be plugged in here to generate "
        "a detailed answer based on the context."
    )

    context_models = [
        RetrievedChunkModel(
            doc_id=r.doc_id,
            chunk_id=r.chunk_id,
            score=r.score,
            text=r.text,
        )
        for r in results
    ]

    return ChatResponse(
        query=req.query,
        answer=answer,
        context=context_models,
    )
