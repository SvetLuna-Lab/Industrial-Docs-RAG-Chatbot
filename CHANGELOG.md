# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog,
and this project adheres to Semantic Versioning.

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
