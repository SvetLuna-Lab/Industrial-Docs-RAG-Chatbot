# src/retriever.py
from __future__ import annotations

"""
Vector retriever for the Industrial Docs RAG Chatbot.

Responsibilities:
- load FAISS index and metadata built by src.ingest;
- encode user queries with the same embedding model;
- perform top-K similarity search;
- optionally reconstruct chunk text from data/raw using the same
  splitting logic as in src.ingest.

This module is intentionally simple and self-contained so that it can be
used both by CLI tools and by a future FastAPI app.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import json

import numpy as np
from sentence_transformers import SentenceTransformer

try:
    import faiss  # type: ignore
except ImportError as exc:
    raise RuntimeError(
        "faiss is required for the retriever. "
        "Please install faiss-cpu in your environment."
    ) from exc

from .config import PATHS, load_app_config
from . import ingest  # to reuse read_text_file / split_into_chunks


# --------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------


@dataclass
class RetrievedChunk:
    """
    A single retrieval result for a user query.

    Attributes:
        doc_id: logical document identifier (relative path from data/raw).
        chunk_id: index of the chunk within the document.
        score: similarity score (inner product / cosine similarity).
        text: optional chunk text; can be populated on demand.
    """

    doc_id: str
    chunk_id: int
    score: float
    text: Optional[str] = None


# --------------------------------------------------------------------
# Metadata loading helpers
# --------------------------------------------------------------------


def _load_metadata(meta_path: Path) -> List[Dict[str, Any]]:
    """
    Load metadata.jsonl created by src.ingest.save_index_and_metadata.

    Each line is a JSON object with at least:
        - row_id
        - doc_id
        - chunk_id
    """
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    rows: List[Dict[str, Any]] = []
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows.append(obj)
    return rows


def _build_rowid_to_meta(meta_rows: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """
    Build a lookup dict row_id -> metadata dict for faster access.
    """
    mapping: Dict[int, Dict[str, Any]] = {}
    for row in meta_rows:
        row_id = int(row["row_id"])
        mapping[row_id] = row
    return mapping


# --------------------------------------------------------------------
# Retriever class
# --------------------------------------------------------------------


class VectorRetriever:
    """
    Vector similarity retriever backed by FAISS and sentence-transformers.

    Typical usage:

        retriever = VectorRetriever.from_default()
        results = retriever.search("Как настроить безопасный SSH?", top_k=5)

    By default it:
    - loads FAISS index from PATHS.index_dir / "faiss_index.bin";
    - loads metadata.jsonl from the same directory;
    - uses the embedding model specified in configs/default.yaml.
    """

    def __init__(
        self,
        index: faiss.Index,
        metadata: List[Dict[str, Any]],
        embedding_model: SentenceTransformer,
        raw_data_dir: Path,
        default_top_k: int,
    ) -> None:
        self.index = index
        self._metadata = metadata
        self._rowid_to_meta = _build_rowid_to_meta(metadata)
        self.embedding_model = embedding_model
        self.raw_data_dir = raw_data_dir
        self.default_top_k = default_top_k

    # --------------------------- construction ------------------------ #

    @classmethod
    def from_default(cls) -> "VectorRetriever":
        """
        Construct a retriever using default paths and configuration:

        - configs/default.yaml (or env override via load_app_config);
        - index and metadata under PATHS.index_dir;
        - raw documents under PATHS.raw_data_dir.
        """
        cfg = load_app_config()

        index_path = PATHS.index_dir / "faiss_index.bin"
        meta_path = PATHS.index_dir / "metadata.jsonl"

        if not index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {index_path}. "
                "Did you run `python -m src.ingest` first?"
            )

        index = faiss.read_index(str(index_path))
        metadata = _load_metadata(meta_path)

        model = SentenceTransformer(
            cfg.embedding.model_name,
            device=cfg.embedding.device,
        )

        return cls(
            index=index,
            metadata=metadata,
            embedding_model=model,
            raw_data_dir=PATHS.raw_data_dir,
            default_top_k=cfg.retrieval.top_k,
        )

    # --------------------------- core API ---------------------------- #

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a single text query into a normalized embedding vector.
        """
        emb = self.embedding_model.encode(
            [query],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        # emb shape: (1, D)
        return emb.astype("float32")

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        with_text: bool = True,
    ) -> List[RetrievedChunk]:
        """
        Perform a vector search over the indexed chunks.

        Args:
            query: user query string.
            top_k: number of results to return (defaults to config.retrieval.top_k).
            with_text: if True, reconstruct chunk text from data/raw.

        Returns:
            List of RetrievedChunk sorted by descending score.
        """
        if self.index.ntotal == 0:
            return []

        k = top_k or self.default_top_k
        if k <= 0:
            return []

        query_vec = self.encode_query(query)
        scores, indices = self.index.search(query_vec, k)

        # scores: (1, k), indices: (1, k)
        scores_row = scores[0]
        idx_row = indices[0]

        results: List[RetrievedChunk] = []

        for score, row_id in zip(scores_row, idx_row):
            if row_id < 0:
                continue  # FAISS uses -1 for empty results

            meta = self._rowid_to_meta.get(int(row_id))
            if meta is None:
                continue

            doc_id = str(meta["doc_id"])
            chunk_id = int(meta["chunk_id"])

            text: Optional[str] = None
            if with_text:
                text = self._load_chunk_text(doc_id, chunk_id)

            results.append(
                RetrievedChunk(
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    score=float(score),
                    text=text,
                )
            )

        # Already sorted by FAISS, but we sort explicitly just in case
        results.sort(key=lambda r: r.score, reverse=True)
        return results

    # ---------------------- internal helpers ------------------------- #

    def _load_chunk_text(self, doc_id: str, chunk_id: int) -> Optional[str]:
        """
        Reconstruct a chunk's text by:
        - reading the original document from data/raw;
        - splitting it with the same logic as in src.ingest;
        - selecting the chunk by index.

        This avoids storing full text in metadata.jsonl.
        """
        file_path = self.raw_data_dir / Path(doc_id)
        if not file_path.exists():
            return None

        try:
            raw_text = ingest.read_text_file(file_path)
            chunks = ingest.split_into_chunks(raw_text)
        except Exception:
            return None

        if 0 <= chunk_id < len(chunks):
            return chunks[chunk_id]

        return None
