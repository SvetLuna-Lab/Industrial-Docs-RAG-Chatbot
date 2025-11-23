# tests/test_retriever_encode_and_search.py
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.retriever import VectorRetriever

# Если faiss не установлен, этот тест просто будет пропущен
faiss = pytest.importorskip("faiss")


def test_encode_build_save_load_and_search(tmp_path: Path):
    """
    End-to-end smoke test for VectorRetriever:

    - encode a few dummy texts;
    - build a FAISS index;
    - save index + metadata to disk;
    - load a new retriever instance from those files;
    - run a top-k search and check that we get sensible results.
    """
    texts = [
        "Python retrieval and FAISS index example",
        "Clinical notes and genomics knowledge base",
        "Rocket mechanics and symbolic gas dynamics",
    ]

    # 1) Encode texts
    builder = VectorRetriever.for_index_building()
    embeddings = builder.encode_texts(texts)

    # Basic shape & normalization checks
    assert embeddings.shape[0] == len(texts)
    # We fixed dim=384 in the implementation
    assert embeddings.shape[1] == 384

    norms = np.linalg.norm(embeddings, axis=1)
    # All non-zero and close to 1.0
    assert np.all(norms > 0.0)
    assert np.allclose(norms, 1.0, atol=1e-5)

    # 2) Build FAISS index
    index = builder.build_faiss_index(embeddings)

    # 3) Save index and aligned metadata
    index_path = tmp_path / "test.index"
    metadata_path = tmp_path / "test_metadata.jsonl"

    builder.save_faiss_index(index, index_path)

    with metadata_path.open("w", encoding="utf-8") as f:
        for i, text in enumerate(texts):
            record = {
                "doc_id": f"doc-{i}",
                "chunk_id": i,
                "source_path": f"doc-{i}.txt",
                "text": text,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    assert index_path.exists()
    assert metadata_path.exists()

    # 4) Load a new retriever from saved files
    retr = VectorRetriever(index_path=index_path, metadata_path=metadata_path)

    # 5) Run a search and check basic properties
    query = "python index"
    results = retr.search(query, top_k=3)

    assert results, "Expected at least one search result"
    assert 1 <= len(results) <= 3

    # Scores must be floats and sorted by descending score (rank 0 is best)
    prev_score = float("inf")
    for r in results:
        assert isinstance(r["score"], float)
        assert r["score"] <= prev_score + 1e-6
        prev_score = r["score"]

        # Metadata should be present and consistent
        meta = r["metadata"]
        assert "doc_id" in meta
        assert "chunk_id" in meta
        assert "source_path" in meta
        assert meta["text"] in texts



def test_encode_texts_deterministic():
    """
    encode_texts(...) должна быть детерминированной:

    - одинаковые тексты → одинаковые векторы;
    - разные тексты → векторы, отличающиеся хотя бы в одной компоненте.

    Это важно для воспроизводимости индекса и тестов.
    """
    retr = VectorRetriever.for_index_building()

    texts = [
        "Python retrieval test",
        "Python retrieval test",  # тот же текст, должен дать тот же вектор
        "Completely different content",
    ]

    embeddings = retr.encode_texts(texts)

    # Проверяем, что первый и второй вектора совпадают
    assert np.allclose(embeddings[0], embeddings[1])

    # А третий отличается хотя бы где-то
    diff = np.abs(embeddings[0] - embeddings[2])
    assert (diff > 1e-6).any(), "Embedding for a different text should not be identical"


