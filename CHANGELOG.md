# Changelog

All notable changes to this project are documented in this file.

The project currently has a single public version – this release collects
the initial industrial RAG skeleton, LLM backend option, Docker support
and example documentation corpus.

---

## [0.2.0] - 2025-11-24

### Added

- **Industrial RAG skeleton** for internal technical documentation:
  - typed configuration in `src/config.py` (`Paths`, `AppConfig`);
  - `VectorRetriever` in `src/retriever.py` with FAISS index support;
  - CLI tool `src/cli.py` with a `search` command;
  - minimal FastAPI app in `src/api/app.py` exposing `/health`, `/search`, `/chat`;
  - lightweight index builder in `scripts/build_index.py`;
  - ingestion pipeline `src/ingest.py` using Sentence-Transformers.

- **LLM backend for `/chat`** (optional):
  - OpenAI-compatible HTTP client using `httpx`;
  - configuration via the `llm` section in `configs/default.yaml`;
  - safe fallback to the historical stub answer if `provider: stub` or API is misconfigured.

- **Industrial documentation corpus** under `data/raw/` (RU + EN):
  - SSH hardening guideline and SSH access approval policy;
  - pump unit maintenance regulation;
  - incident report templates;
  - RAG bot operator manual;
  - data classification guideline.

- **Dockerfile** for running the FastAPI service with Uvicorn:
  - `docker build -t industrial-docs-rag .`
  - `docker run --rm -p 8000:8000 industrial-docs-rag`

- **Documentation updates:**
  - main `README.md` describing architecture, LLM backend and Docker usage;
  - short Russian overview `README_RU.md`;
  - project overviews in `docs/Overview_EN.md` and `docs/Overview_RU.md`;
  - screenshots for CLI search, Swagger UI and Uvicorn run.

### Notes

This is the first public version of the project.  
Future releases will continue from this version tag (`0.2.x`, `0.3.x`, etc.).

## [0.1.0] - 2025-11-24

### Added
- Basic project scaffolding for a vector-based RAG API.
- `src/config.py` with central configuration for index and model paths.
- `src/retriever.py` with a `VectorRetriever` abstraction (FAISS + embeddings).
- FastAPI application in `src/api/app.py` with `/health`, `/search`, `/chat` endpoints.
- Tests:
  - `tests/test_api_smoke.py` with DummyRetriever and API smoke checks.
- Dependency files:
  - `requirements.txt` for runtime.
  - `requirements-dev.txt` for development and tests.
- Tooling:
  - `pytest.ini` for pytest configuration.
- Licensing:
  - `LICENSE` with MIT license.
