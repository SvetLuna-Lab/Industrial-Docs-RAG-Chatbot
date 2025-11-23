# tests/test_build_index_script.py
from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest

from src import config

# Если faiss не установлен, тест просто будет пропущен
faiss = pytest.importorskip("faiss")


def test_build_index_script_creates_index_and_metadata(tmp_path: Path, monkeypatch):
    """
    Лёгкий end-to-end тест для скрипта scripts/build_index.py:

    - создаём временную папку с .txt/.md файлами;
    - временно переназначаем config.INDEX_PATH и METADATA_PATH в tmp_path;
    - запускаем модуль scripts.build_index как скрипт;
    - проверяем, что:
        * создан FAISS-индекс,
        * создан непустой metadata.jsonl.
    """
    # 1) Готовим входные файлы
    input_dir = tmp_path / "docs"
    input_dir.mkdir()

    (input_dir / "doc_a.txt").write_text(
        "First dummy document about rockets and gas dynamics.",
        encoding="utf-8",
    )
    (input_dir / "doc_b.md").write_text(
        "Second dummy document about clinical RAG and retrieval.",
        encoding="utf-8",
    )

    # 2) Переопределяем пути индекса и метаданных на временную директорию
    index_path = tmp_path / "test_index.faiss"
    metadata_path = tmp_path / "test_index_metadata.jsonl"

    # В config пути строковые — подменяем на str(...)
    monkeypatch.setattr(config, "INDEX_PATH", str(index_path))
    monkeypatch.setattr(config, "METADATA_PATH", str(metadata_path))

    # 3) Запускаем scripts.build_index как если бы он был вызван из командной строки
    argv_backup = sys.argv[:]
    try:
        sys.argv = ["scripts.build_index", "--input-dir", str(input_dir)]
        runpy.run_module("scripts.build_index", run_name="__main__")
    finally:
        sys.argv = argv_backup

    # 4) Проверяем, что файлы созданы и не пустые
    assert index_path.exists(), "FAISS index file was not created"
    assert index_path.stat().st_size > 0, "FAISS index file is empty"

    assert metadata_path.exists(), "Metadata JSONL file was not created"
    content = metadata_path.read_text(encoding="utf-8").strip()
    assert content != "", "Metadata JSONL file is empty"
    # Минимальная проверка, что там действительно JSONL
    first_line = content.splitlines()[0]
    assert first_line.startswith("{") and first_line.endswith("}")
