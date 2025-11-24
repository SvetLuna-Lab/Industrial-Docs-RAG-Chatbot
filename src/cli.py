from __future__ import annotations

"""
Simple CLI entry point for the Industrial Docs RAG Chatbot.

Current functionality:
- `search` command: encode a user query, run vector search over FAISS index,
  print top-K chunks with scores and short text snippets.

Example usage (from project root):

    python -m src.cli search "How to harden SSH on Ubuntu?" --top-k 5

The CLI uses:
- default paths from src.config.PATHS,
- default settings from configs/default.yaml,
- VectorRetriever.from_default() for loading index + metadata.
"""

import argparse
from textwrap import shorten
from typing import Any

from .retriever import VectorRetriever


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


def _extract_text(result: Any) -> str:
    """
    Extract chunk text from a retriever result.

    Priority:
    1) direct `text` attribute / key;
    2) `metadata["text"]` if result is a dict with `metadata` dict;
    3) empty string if nothing found.
    """
    text = _get_attr_or_key(result, "text", None)

    if (text is None or text == "") and isinstance(result, dict):
        meta = result.get("metadata") or {}
        if isinstance(meta, dict):
            text = meta.get("text", "")

    if text is None:
        text = ""

    return str(text)


def cmd_search(args: argparse.Namespace) -> None:
    """
    Handle the `search` subcommand.

    Steps:
    - load VectorRetriever via from_default();
    - run search(query, top_k, with_text=True);
    - print ranked results with score, doc_id, chunk_id and snippet.
    """
    retriever = VectorRetriever.from_default()

    top_k = args.top_k
    query = args.query

    # with_text=True – чтобы API возвращал текст чанка (если он есть в metadata)
    results = retriever.search(query, top_k=top_k, with_text=True)

    if not results:
        print("No results found. Did you run the ingest step and build the index?")
        return

    print(f"Query: {query!r}")
    print(f"Top-{len(results)} results:\n")

    for i, r in enumerate(results, start=1):
        doc_id = _get_attr_or_key(r, "doc_id", "unknown")
        chunk_id = int(_get_attr_or_key(r, "chunk_id", 0))
        score = float(_get_attr_or_key(r, "score", 0.0))

        raw_text = _extract_text(r)
        snippet = shorten(raw_text.replace("\n", " "), width=200, placeholder="…")

        print(f"[{i}] score={score:.4f}  doc={doc_id}  chunk={chunk_id}")
        if snippet:
            print(f"    {snippet}")
        print()


def build_parser() -> argparse.ArgumentParser:
    """
    Build the top-level argparse parser with subcommands.

    For now we only provide `search`, but this can be extended later
    with commands like `ingest`, `serve-api`, etc.
    """
    parser = argparse.ArgumentParser(
        prog="industrial-docs-rag-cli",
        description="CLI tools for the Industrial Docs RAG Chatbot.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- search subcommand ---
    p_search = subparsers.add_parser(
        "search",
        help="Run vector search over the indexed documentation.",
    )
    p_search.add_argument(
        "query",
        type=str,
        help="User query text.",
    )
    p_search.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Number of results to return (defaults to value from config).",
    )
    p_search.set_defaults(func=cmd_search)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # dispatch to subcommand handler
    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        return

    func(args)


if __name__ == "__main__":
    main()
