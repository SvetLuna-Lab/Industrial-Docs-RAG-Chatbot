# src/config.py
from __future__ import annotations

"""
Central configuration module for the Industrial Docs RAG Chatbot.

Responsibilities:
- define project-level paths (raw docs, index, configs);
- define typed config models for embeddings, retrieval and LLM client;
- load overrides from a YAML file (configs/default.yaml) when present.

This keeps the rest of the code (ingest, retriever, API) clean and
independent from hard-coded paths or magic constants.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field


# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------


class Paths(BaseModel):
    """Convenience container for important project paths."""

    project_root: Path
    configs_dir: Path
    raw_data_dir: Path
    index_dir: Path

    @classmethod
    def from_project_root(cls, root: Optional[Path] = None) -> "Paths":
        """
        Build a Paths object starting from the project root.

        If root is not provided, the root is inferred as two levels
        above this file: project_root/src/config.py → project_root.
        """
        if root is None:
            root = Path(__file__).resolve().parents[1]

        return cls(
            project_root=root,
            configs_dir=root / "configs",
            raw_data_dir=root / "data" / "raw",
            index_dir=root / "data" / "index",
        )


# A global instance that can be imported from other modules.
PATHS = Paths.from_project_root()

# -------------------------------------------------------------------
# Config sections
# -------------------------------------------------------------------


class EmbeddingConfig(BaseModel):
    """
    Configuration for the embedding model used in the retriever.

    model_name: HuggingFace / sentence-transformers model.
    device: "cpu" or "cuda" (for now we default to CPU).
    batch_size: how many chunks to encode in one go.
    """

    model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Embedding model identifier.",
    )
    device: str = Field(
        default="cpu",
        description="Device for embedding computations: 'cpu' or 'cuda'.",
    )
    batch_size: int = Field(
        default=16,
        ge=1,
        description="Batch size for embedding computation.",
    )


class RetrievalConfig(BaseModel):
    """
    Configuration for the vector search / retriever.

    top_k: how many chunks to retrieve for each query.
    """

    top_k: int = Field(
        default=5,
        ge=1,
        description="Number of top chunks to retrieve for a query.",
    )


class LLMConfig(BaseModel):
    """
    Configuration for the LLM client.

    In the first iteration this can be a stub (simple template-based
    answer). Later it can be wired to any API-compatible provider.
    """

    provider: str = Field(
        default="stub",
        description="LLM provider identifier, e.g. 'stub', 'openai', 'local'.",
    )
    model_name: str = Field(
        default="gpt-4o-mini",
        description="Model name for the chosen provider.",
    )
    api_base: str = Field(
        default="",
        description="Optional custom API base URL (for self-hosted / proxies).",
    )
    api_key_env: str = Field(
        default="OPENAI_API_KEY",
        description="Name of the environment variable that stores the API key.",
    )


class AppConfig(BaseModel):
    """
    Top-level application config.

    All of these fields can be overridden via configs/default.yaml, e.g.:

    embedding:
      model_name: sentence-transformers/all-mpnet-base-v2
      device: cuda

    retrieval:
      top_k: 8

    llm:
      provider: stub
    """

    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)


# -------------------------------------------------------------------
# YAML loading helpers
# -------------------------------------------------------------------


DEFAULT_CONFIG_PATH = PATHS.configs_dir / "default.yaml"


def _load_yaml(path: Path) -> Dict[str, Any]:
    """
    Load a YAML file into a plain dict.

    If the file does not exist or is empty, returns an empty dict.
    """
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping at top level in {path}, got {type(data)}")
    return data


def load_app_config(path: Optional[Path] = None) -> AppConfig:
    """
    Load application configuration from a YAML file and merge it
    with the defaults defined in AppConfig.

    - If 'path' is None, configs/default.yaml is used.
    - Missing fields fall back to the defaults from the dataclasses.
    - Extra fields in YAML are ignored.
    """
    cfg_path = path or DEFAULT_CONFIG_PATH
    raw = _load_yaml(cfg_path)

    # pydantic will ignore unknown keys by default, and recursively construct
    # nested models from dicts: {"embedding": {...}, "retrieval": {...}, ...}
    return AppConfig(**raw)
