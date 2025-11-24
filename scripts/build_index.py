from __future__ import annotations

"""
Build FAISS index and metadata from a directory with raw text documents.

This script is intentionally simple and can be customized
for a specific company / documentation layout.

High-level steps:
1. Collect all text files under a given input directory.
2. Convert them into (doc_id, text) records.
3. Chunk long texts into smaller passages.
4. Compute embeddings for all chunks.
5. Build and save a FAISS index + JSONL metadata with chunk info.

The index and metadata paths are taken from src.config.
"""

import argparse
from pathlib import Path
from typing import Iterable, List, Dict, Any

import json

import numpy as np
from tqdm import tqdm

from src import config
from src.retriever import VectorRetriever


def iter_text_files(
    root: Path,
    suffixes: tuple[str, ...] = (".txt", ".md"),
) -> Iterable[Path]:
    """
    Recursively yield all files under `root` with given suffixes.
    """
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in suffixes:
            yield path


def read_document(path: Path) -> str:
    """
    Read a document as UTF-8 text.

    For now we assume plain text / Markdown.
    Conversion from PDF/HTML can be added separately.
    """
    return path.read_text(encoding="utf-8", errors="ignore")


def simple_chunk_text(
    text: str,
    max_chars: int = 800,
    overlap: int = 100,
) -> List[str]:
    """
    Very simple character-based chunking.

    - max_chars: target length of a chunk,
    - overlap: number of characters to overlap between consecutive chunks.

    This is a placeholder that can be replaced by a sentence-level
    or section-aware chunker later.
    """
    cleaned = text.replace("\r\n", "\n")
    chunks: List[str] = []

    start = 0
    n = len(cleaned)
    while start < n:
        end = min(start + max_chars, n)
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = end - overlap

    return chunks


def build_index_from_dir(
    input_dir: Path,
    index_path: Path,
    metadata_path: Path,
    max_chars: int = 800,
    overlap: int = 100,
) -> None:
    """
    Main pipeline:
    - iterate over text files,
    - chunk their content,
    - embed chunks,
    - build and save FAISS index + metadata JSONL.

    Each metadata record has the shape:
    {
        "doc_id": str,
        "chunk_id": int,
        "source_path": str,
        "text": str,        # chunk text
    }
    """
    # Collect chunks
    docs_meta: List[Dict[str, Any]] = []
    texts: List[str] = []

    print(f"[build_index] Scanning directory: {input_dir}")
    for doc_idx, path in enumerate(
        tqdm(iter_text_files(input_dir), desc="Docs", unit="doc")
    ):
        doc_id = path.stem
        raw_text = read_document(path)
        chunks = simple_chunk_text(raw_text, max_chars=max_chars, overlap=overlap)

        for chunk_idx, chunk_text in enumerate(chunks):
            texts.append(chunk_text)
            docs_meta.append(
                {
                    "doc_id": doc_id,
                    "chunk_id": chunk_idx,
                    "source_path": str(path),
                    "text": chunk_text,
                }
            )

    if not texts:
        raise RuntimeError(f"No text chunks found in {input_dir}")

    print(f"[build_index] Total chunks: {len(texts)}")

    # Initialize retriever in "index builder" mode.
    # We rely on it to provide an embedding function and a FAISS index builder.
    retriever = VectorRetriever.for_index_building()

    print("[build_index] Computing embeddings...")
    embeddings = retriever.encode_texts(texts)  # expected shape: (N, dim)
    if isinstance(embeddings, list):
        embeddings = np.array(embeddings, dtype="float32")
    else:
        embeddings = embeddings.astype("float32")

    print(f"[build_index] Embeddings shape: {embeddings.shape}")

    print("[build_index] Building FAISS index...")
    index = retriever.build_faiss_index(embeddings)

    # Ensure parent directories exist
    index_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[build_index] Saving index to: {index_path}")
    retriever.save_faiss_index(index, index_path)

    print(f"[build_index] Saving metadata to: {metadata_path}")
    with metadata_path.open("w", encoding="utf-8") as f:
        for meta in docs_meta:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")

    print("[build_index] Done.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build FAISS index from raw text documents."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory with raw text / Markdown documents.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=800,
        help="Max characters per chunk (default: 800).",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=100,
        help="Character overlap between consecutive chunks (default: 100).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    # Преобразуем config.INDEX_PATH / METADATA_PATH к Path,
    # чтобы поддерживать и str, и Path в конфиге/тестах.
    index_path = Path(config.INDEX_PATH).resolve()
    metadata_path = Path(config.METADATA_PATH).resolve()

    build_index_from_dir(
        input_dir=input_dir,
        index_path=index_path,
        metadata_path=metadata_path,
        max_chars=args.max_chars,
        overlap=args.overlap,
    )


if __name__ == "__main__":
    main()
