# src/ingest.py
from __future__ import annotations

"""
Ingestion pipeline for the Industrial Docs RAG Chatbot.

Responsibilities:
- walk through data/raw/ and collect documents;
- extract plain text from simple formats (.txt, .md);
- split text into overlapping chunks;
- compute embeddings for all chunks;
- build and persist a FAISS index + metadata.

This module is written as a simple CLI script that can be called as:

    python -m src.ingest

Later it can be extended to support PDFs, DOCX, more advanced chunking,
and incremental updates.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Dict, Tuple

import argparse
import json

import numpy as np

from sentence_transformers import SentenceTransformer

try:
    import faiss  # type: ignore
except ImportError as exc:
    raise RuntimeError(
        "faiss is required for the ingestion pipeline. "
        "Please install faiss-cpu in your environment."
    ) from exc

from .config import PATHS, load_app_config


# -------------------------------------------------------------------
# Data structures
# -------------------------------------------------------------------


@dataclass
class Chunk:
    """
    A single text chunk extracted from a document.

    Attributes:
        doc_id: logical document identifier (usually the relative path).
        chunk_id: running integer index within the document.
        text: the actual chunk text.
    """

    doc_id: str
    chunk_id: int
    text: str


# -------------------------------------------------------------------
# Document loading and chunking
# -------------------------------------------------------------------


def iter_raw_files(raw_dir: Path) -> Iterable[Path]:
    """
    Iterate over files in the raw data directory.

    For now we support only .txt and .md files explicitly.
    PDFs and other formats can be added later via dedicated parsers.
    """
    if not raw_dir.exists():
        return []

    for path in raw_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in {".txt", ".md"}:
            yield path
        # TODO: add .pdf / .docx support with proper parsers


def read_text_file(path: Path, encoding: str = "utf-8") -> str:
    """
    Read a text file into a single string.

    For Markdown we simply read the raw text; rendering is not required
    for retrieval.
    """
    with path.open("r", encoding=encoding, errors="ignore") as f:
        return f.read()


def split_into_chunks(
    text: str,
    max_chars: int = 1000,
    overlap_chars: int = 200,
) -> List[str]:
    """
    Split a long text into overlapping chunks based on character count.

    This is a simple, model-agnostic splitter:
    - split by paragraphs (double newline) first;
    - then pack paragraphs into chunks no longer than `max_chars`;
    - consecutive chunks overlap by `overlap_chars` characters.

    Later this can be replaced by a token-based splitter (e.g. tiktoken).
    """
    # Normalize line breaks and split by empty lines
    paragraphs = [p.strip() for p in text.replace("\r\n", "\n").split("\n\n")]
    paragraphs = [p for p in paragraphs if p]

    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for p in paragraphs:
        # +2 for the extra newlines we add when joining
        p_len = len(p) + 2
        if current and current_len + p_len > max_chars:
            # flush current chunk
            chunk_text = "\n\n".join(current)
            chunks.append(chunk_text)

            # prepare the overlapping part
            if overlap_chars > 0 and len(chunk_text) > overlap_chars:
                overlap = chunk_text[-overlap_chars:]
                current = [overlap]
                current_len = len(overlap)
            else:
                current = []
                current_len = 0

        current.append(p)
        current_len += p_len

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def build_chunks_for_corpus(raw_dir: Path) -> List[Chunk]:
    """
    Walk through data/raw and build a flat list of text chunks.

    Each chunk knows:
    - which logical document it comes from (doc_id),
    - its position within that document (chunk_id),
    - the text itself.

    The doc_id is stored as a POSIX-style relative path from raw_dir.
    """
    all_chunks: List[Chunk] = []

    for file_path in iter_raw_files(raw_dir):
        # doc_id: relative path from raw_dir, POSIX-style
        rel_path = file_path.relative_to(raw_dir).as_posix()
        raw_text = read_text_file(file_path)

        chunks = split_into_chunks(raw_text)
        for idx, ch_text in enumerate(chunks):
            all_chunks.append(
                Chunk(doc_id=rel_path, chunk_id=idx, text=ch_text)
            )

    return all_chunks


# -------------------------------------------------------------------
# Embeddings and FAISS index
# -------------------------------------------------------------------


def embed_chunks(
    chunks: List[Chunk],
    model_name: str,
    device: str,
    batch_size: int,
) -> np.ndarray:
    """
    Compute embeddings for all chunks using a sentence-transformers model.

    Returns:
        A NumPy array of shape (N, D) where N = number of chunks,
        D = embedding dimension.
    """
    if not chunks:
        raise ValueError("No chunks provided for embedding.")

    model = SentenceTransformer(model_name, device=device)

    texts = [c.text for c in chunks]
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # cosine similarity friendly
    )

    return embeddings


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build a FAISS index over the given embeddings.

    We use an IndexFlatIP (inner product) with normalized embeddings,
    which corresponds to cosine similarity.
    """
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings array, got shape {embeddings.shape}.")

    n_vectors, dim = embeddings.shape
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype("float32"))

    if index.ntotal != n_vectors:
        raise RuntimeError("FAISS index size does not match number of embeddings.")

    return index


def save_index_and_metadata(
    index: faiss.Index,
    chunks: List[Chunk],
    index_dir: Path,
) -> None:
    """
    Persist the FAISS index and its metadata to disk.

    Files:
        - data/index/faiss_index.bin
        - data/index/metadata.jsonl  (one JSON per line)
    """
    index_dir.mkdir(parents=True, exist_ok=True)

    index_path = index_dir / "faiss_index.bin"
    meta_path = index_dir / "metadata.jsonl"

    faiss.write_index(index, str(index_path))

    with meta_path.open("w", encoding="utf-8") as f:
        for row_id, ch in enumerate(chunks):
            meta = {
                "row_id": row_id,
                "doc_id": ch.doc_id,
                "chunk_id": ch.chunk_id,
                # we do not store the full text here to keep metadata small;
                # the retriever can load text by doc_id + chunk_id later.
            }
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")

    print(f"[OK] Saved FAISS index to {index_path}")
    print(f"[OK] Saved metadata for {len(chunks)} chunks to {meta_path}")


# -------------------------------------------------------------------
# CLI entry point
# -------------------------------------------------------------------


def run_ingestion(config_path: Path | None = None) -> None:
    """
    Top-level function that runs the ingestion pipeline:

    - load configuration (embedding / retrieval / llm);
    - build chunks from data/raw;
    - compute embeddings;
    - build FAISS index;
    - save index + metadata under data/index.
    """
    cfg = load_app_config(config_path)

    raw_dir = PATHS.raw_data_dir
    index_dir = PATHS.index_dir

    print(f"[INFO] Project root: {PATHS.project_root}")
    print(f"[INFO] Raw docs directory: {raw_dir}")
    print(f"[INFO] Index directory: {index_dir}")
    print(f"[INFO] Embedding model: {cfg.embedding.model_name} (device={cfg.embedding.device})")

    chunks = build_chunks_for_corpus(raw_dir)
    if not chunks:
        print("[WARN] No chunks found in data/raw. Nothing to index.")
        return

    print(f"[INFO] Built {len(chunks)} chunks from raw documents.")

    embeddings = embed_chunks(
        chunks,
        model_name=cfg.embedding.model_name,
        device=cfg.embedding.device,
        batch_size=cfg.embedding.batch_size,
    )
    print(f"[INFO] Embeddings shape: {embeddings.shape}")

    index = build_faiss_index(embeddings)
    save_index_and_metadata(index, chunks, index_dir)

    print("[DONE] Ingestion pipeline completed.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build vector index from documents in data/raw."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Optional path to a YAML config file (defaults to configs/default.yaml).",
    )

    args = parser.parse_args()
    cfg_path = Path(args.config).resolve() if args.config else None

    run_ingestion(cfg_path)


if __name__ == "__main__":
    main()
