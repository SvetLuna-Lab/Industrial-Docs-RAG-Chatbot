from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Optional
import json

import numpy as np

from . import config

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover
    faiss = None


class VectorRetriever:
    """
    Vector-based retriever for document chunks.

    Responsibilities:
    - encode texts into embedding vectors;
    - build/save/load a FAISS index;
    - execute top-k similarity search and return metadata for chunks.

    In this skeleton implementation we use a very simple hash-based
    embedding instead of a real encoder model. This is enough to keep
    the RAG pipeline shape without adding heavy dependencies.
    """

    def __init__(
        self,
        index_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
        app_config: Optional[config.AppConfig] = None,
    ) -> None:
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.app_config = app_config

        self.index = None
        self.metadata: List[Dict[str, Any]] = []

        # Lazy init — we only try to load if paths are provided
        if index_path is not None and index_path.exists() and faiss is not None:
            self.index = faiss.read_index(str(index_path))

        if metadata_path is not None and metadata_path.exists():
            with metadata_path.open("r", encoding="utf-8") as f:
                self.metadata = [json.loads(line) for line in f]

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls) -> "VectorRetriever":
        """
        Construct a retriever using paths and defaults from src.config.
        """
        app_cfg = config.load_app_config()
        index_path = config.INDEX_PATH
        metadata_path = config.METADATA_PATH
        return cls(index_path=index_path, metadata_path=metadata_path, app_config=app_cfg)

    @classmethod
    def from_default(cls) -> "VectorRetriever":
        """
        Backwards-compatible alias for from_config().

        This name is used in some README examples.
        """
        return cls.from_config()

    @classmethod
    def for_index_building(cls) -> "VectorRetriever":
        """
        Construct a retriever for index building.

        In this mode we do not expect an existing FAISS index or metadata.
        The instance is used only to:
        - encode text chunks;
        - build and save a new index.
        """
        return cls(index_path=None, metadata_path=None, app_config=None)

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        Encode a list of texts into embedding vectors.

        Current implementation: simple hash-based bag-of-words encoder.
        - Dimension is fixed (dim = 384).
        - For each token we increment one bucket determined by hash(token).
        - Vectors are L2-normalized.

        This is a placeholder that can later be replaced by:
        - Sentence-Transformers;
        - OpenAI embeddings;
        - any other encoder.
        """
        dim = 384
        vectors = np.zeros((len(texts), dim), dtype="float32")

        for i, text in enumerate(texts):
            for token in text.split():
                h = hash(token) % dim
                vectors[i, h] += 1.0

        # L2 normalization
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        vectors = vectors / norms
        return vectors.astype("float32")

    # ------------------------------------------------------------------
    # FAISS index helpers
    # ------------------------------------------------------------------
    def build_faiss_index(self, embeddings: np.ndarray):
        """
        Build a FAISS index from embeddings.

        Uses inner-product (cosine-like, if embeddings are normalized).
        """
        if faiss is None:  # pragma: no cover
            raise RuntimeError("faiss is not installed; cannot build index")

        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype("float32")

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        return index

    def save_faiss_index(self, index, path: Path) -> None:
        """
        Save FAISS index to disk.
        """
        if faiss is None:  # pragma: no cover
            raise RuntimeError("faiss is not installed; cannot save index")

        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(path))

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        with_text: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Run a top-k similarity search for a query string.

        Returns a list of records:
        {
            "rank": int,
            "score": float,
            "doc_id": str | None,
            "chunk_id": int | None,
            "source_path": str | None,
            "text": str | None,              # if with_text=True and present in metadata
            "metadata": dict,                # raw metadata for the chunk
        }
        """
        if self.index is None or faiss is None or not self.metadata:
            # No index, no FAISS, or no metadata – nothing to search
            return []

        # Determine effective top_k: passed value > config > hard default
        if top_k is None or top_k <= 0:
            if self.app_config is not None:
                top_k = max(1, self.app_config.retrieval.top_k)
            else:
                top_k = 5

        query_vec = self.encode_texts([query])
        scores, indices = self.index.search(query_vec, top_k)

        results: List[Dict[str, Any]] = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0:
                continue

            meta: Dict[str, Any] = {}
            if 0 <= idx < len(self.metadata):
                meta = dict(self.metadata[idx])

            # Fallbacks if keys are missing
            doc_id = meta.get("doc_id")
            chunk_id = meta.get("chunk_id")
            source_path = meta.get("source_path")
            chunk_text = meta.get("text")

            result: Dict[str, Any] = {
                "rank": rank,
                "score": float(score),
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "source_path": source_path,
                "metadata": meta,
            }

            if with_text:
                result["text"] = chunk_text

            results.append(result)

        return results
