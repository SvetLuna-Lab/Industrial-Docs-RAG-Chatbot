# Project Overview (EN)

## 1. Idea

This repository is a **skeleton RAG system** (Retrieval-Augmented Generation)  
for working with internal documentation (instructions, regulations, technical descriptions).

**Goal:**

- collect documents into a single searchable corpus;
- provide a simple API that:
  - finds relevant text fragments for a given query;
  - can later pass these fragments to an LLM to generate an answer.

The current version is not tied to any specific company and does not contain real documentation –  
it is a template that can be adapted to any domain.

---

## 2. Architecture (high-level)

### Configuration (`src/config.py`, `configs/default.yaml`)

- Defines core project paths:
  - `data/raw` — raw text documents;
  - `data/index` — FAISS index + metadata;
  - `configs/default.yaml` — parameters for embeddings, retrieval, and LLM.
- Provides a typed `AppConfig` (embedding / retrieval / llm) and shared paths:
  - `config.INDEX_PATH`
  - `config.METADATA_PATH`

### Indexing (two options)

1. **Lightweight script** `scripts/build_index.py`  
   - Reads `.txt` / `.md` files;
   - splits them into character-based chunks;
   - uses a simple hash-based encoder via `VectorRetriever.encode_texts`;
   - builds a FAISS index and JSONL metadata (`doc_id`, `chunk_id`, `source_path`, `text`).

2. **Heavier ingestion pipeline** `src/ingest.py`  
   - Splits documents into paragraph-based overlapping chunks;
   - uses a Sentence-Transformers model for real embeddings;
   - writes index and metadata to the same files as `scripts/build_index.py`  
     (paths taken from `config.INDEX_PATH` and `config.METADATA_PATH`).

Both options produce a file format that is understood by the retriever and the API.

### Vector Retriever (`src/retriever.py`)

Class **`VectorRetriever`**:

- capabilities:
  - encode texts into embedding vectors (hash-based placeholder encoder);
  - build a FAISS index;
  - save / load index and metadata;
  - run `search(query, top_k, with_text)` and return a list of results containing:
    - `rank`, `score`, `doc_id`, `chunk_id`, `source_path`, `metadata`, and optionally `text`.
- factory helpers:
  - `VectorRetriever.from_config()` / `from_default()` — load index + metadata using paths from `config`;
  - `VectorRetriever.for_index_building()` — “index builder only” mode.

### HTTP API (`src/api/app.py`)

Minimal FastAPI application with endpoints:

- `GET /health` — liveness check;
- `POST /search` — vector search over the index;
- `POST /chat` — stub RAG chat endpoint:
  - returns a template answer;
  - includes retrieved context as a list of `SearchResult` items.

The retriever instance is created lazily via `get_retriever()` and uses `VectorRetriever.from_config()`.

### CLI (`src/cli.py`)

Simple CLI for manual search:

```bash
python -m src.cli search "How to harden SSH on Ubuntu?" --top-k 5

```


- uses VectorRetriever.from_default();

- runs a search;

- prints ranked results with score, doc_id, chunk_id, and a short text snippet.


**Tests**

- tests/test_api_smoke.py — smoke tests for the HTTP API:

- monkeypatches get_retriever with a DummyRetriever;

- checks the shape of responses for /health, /search, /chat.

- tests/test_build_index_script.py — small end-to-end test for scripts/build_index.py:

- creates a temporary directory with text files;

- temporarily overrides config.INDEX_PATH and config.METADATA_PATH;

- runs the script and verifies index + metadata files.

- tests/test_retriever_encode_and_search.py:

- tests VectorRetriever.encode_texts (shape, normalization, determinism);

- checks the full cycle: encode → build index → save → load → search.


**3. Project Structure**

```text
project-root/
├─ configs/
│  └─ default.yaml          # embedding / retrieval / LLM settings
├─ data/
│  ├─ raw/                  # raw documents (.txt, .md)
│  └─ index/                # faiss_index.bin + metadata.jsonl (generated)
├─ src/
│  ├─ __init__.py
│  ├─ config.py             # central settings: paths, AppConfig, INDEX_PATH/METADATA_PATH
│  ├─ retriever.py          # VectorRetriever: FAISS loading / search
│  ├─ cli.py                # CLI: python -m src.cli search "..." --top-k 5
│  ├─ ingest.py             # ingestion pipeline using sentence-transformers
│  └─ api/
│     └─ app.py             # FastAPI app: /health, /search, /chat
├─ scripts/
│  └─ build_index.py        # lightweight index builder script
├─ tests/
│  ├─ test_api_smoke.py
│  ├─ test_build_index_script.py
│  └─ test_retriever_encode_and_search.py
├─ docs/
│  ├─ Overview_EN.md        # this file
│  └─ Overview_RU.md        # Russian overview
│  └─ images/
│     └─ cli_search_example.png
│     └─ swagger_search_example.png
│     └─ uvicorn_run_example.png
├─ requirements.txt         # runtime dependencies
├─ requirements-dev.txt     # dev dependencies (pytest, black, mypy, etc.)
├─ pytest.ini               # pytest configuration
├─ .gitignore               # ignore venv, caches, temp files
└─ README.md                # main project description (EN)

```

**4. Typical Usage Scenarios**

**4.1. Prepare the data**

1.Put your documents into data/raw/ as plain text files:

.txt

.md

2.If needed, convert PDF / DOCX → text before feeding them to the project.


**4.2. Build the index**

**Option A — lightweight, hash-based embeddings**

python -m scripts.build_index --input-dir data/raw


The script will:

- read all .txt / .md files in data/raw;

- split them into character-based chunks;

- compute hash-based embeddings;

- build a FAISS index and write:

- data/index/faiss_index.bin

- data/index/metadata.jsonl

**Option B — ingestion pipeline with Sentence-Transformers**

python -m src.ingest


The script will:

- read all documents from data/raw;

- build paragraph-based overlapping chunks;

- use the model from configs/default.yaml (embedding.model_name) on CPU/GPU;

- build a FAISS index and write to the same files as option A.

Both options are compatible with VectorRetriever and the API.


**4.3. Run the HTTP API**

From the project root:

uvicorn src.api.app:app --reload


Then:

- GET /health — check that the service is alive;

- POST /search — send JSON like:


{
  "query": "How to harden SSH on Ubuntu?",
  "top_k": 5
}


- POST /chat — returns:

- a stub answer ("This is a stub chat endpoint response...");

- context: list of retrieved chunks.


**4.4. CLI search**

Without HTTP, you can quickly test search from the CLI:

python -m src.cli search "How to harden SSH on Ubuntu?" --top-k 5


You will see:

- rank;

- score;

- doc_id, chunk_id;

- a short text snippet.


**5. Project Status**

- Index building is implemented in two flavors:

- lightweight hash-based (scripts/build_index.py);

- Sentence-Transformers-based ingestion (src/ingest.py).

- The RAG /chat endpoint is still a stub:

- returns a fixed answer;

- attaches retrieved context.

- The main focus is on clean architecture and separation of concerns:

- configuration (configs/default.yaml, src/config.py);

- isolated retriever (src/retriever.py);

- thin HTTP API (src/api.app);

- CLI and indexing scripts (src/cli.py, scripts/build_index.py, src/ingest.py);

- simple but illustrative tests (tests/…).

You can use this project as a starting point for:

- an internal documentation chatbot;

- a prototype of a support system for engineers/operators;

- an educational RAG example for students and junior developers;

- a demo of the basic “index → retriever → API → client (CLI/HTTP)” architecture.

