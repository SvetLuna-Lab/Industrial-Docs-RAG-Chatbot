# Industrial Docs RAG Chatbot

Retrieval-augmented search and chat over internal technical / industrial documentation.

This repository contains the **retrieval layer** (embeddings + FAISS index + API)
for a future RAG system. It focuses on:

- consistent path/config management,
- a reusable `VectorRetriever`,
- a simple CLI for ad-hoc search,
- a minimal FastAPI app with `/search` and `/chat` endpoints,
- smoke tests for the API.

The LLM backend and ingestion pipeline can be plugged in later.

---

## Features

- 🔍 **Vector search** over pre-indexed documents using sentence embeddings.
- 🧱 **Config-driven paths** for index and models (`src/config.py`, `configs/default.yaml`).
- 🧪 **API smoke tests** with a dummy retriever (no real FAISS needed on CI).
- 🖥️ **CLI tool** for quick terminal search:
  - `python -m src.cli search "your query" --top-k 5`
- 🌐 **HTTP API** (FastAPI + Uvicorn):
  - `GET /health`
  - `POST /search`
  - `POST /chat` (stub RAG endpoint that only uses retrieval for now)

---

## Project structure

```text
industrial-docs-rag-chatbot/
├─ configs/
│  └─ default.yaml           # default model / search / path settings
├─ src/
│  ├─ __init__.py
│  ├─ config.py              # PATHS, MODEL, SEARCH defaults (central config helper)
│  ├─ retriever.py           # VectorRetriever wrapper (embeddings + FAISS index)
│  ├─ cli.py                 # CLI entry point: `search` command
│  └─ api/
│     └─ app.py              # FastAPI app: /health, /search, /chat (stub)
├─ tests/
│  └─ test_api_smoke.py      # API smoke tests with dummy retriever
├─ requirements.txt          # runtime dependencies (FastAPI, FAISS, embeddings, etc.)
├─ requirements-dev.txt      # dev deps: pytest, httpx, etc. (optional)
├─ pytest.ini                # pytest configuration (optional)
├─ CHANGELOG.md              # version history
├─ LICENSE                   # MIT License
└─ README.md                 # this file



Note: the index building / ingestion script is intentionally left open-ended.
You can plug in your own pipeline that:

reads raw docs, 2) chunks them, 3) computes embeddings, 4) builds the FAISS index
at the path configured in src/config.py / configs/default.yaml.



Configuration

Two main places store configuration:

configs/default.yaml — human-editable YAML with:

model name,

default top_k,

data / index paths,

any extra search parameters.

src/config.py — Python helper to:

resolve paths (PROJECT_ROOT, DATA_DIR, INDEX_PATH, etc.),

load/merge YAML config if needed,

keep path logic in one place.

Typical fields (example):


model:
  name: sentence-transformers/all-MiniLM-L6-v2

search:
  default_top_k: 5

paths:
  index_path: data/index/faiss_index.bin
  embeddings_cache_dir: data/emb_cache


Adjust these to match your ingestion pipeline and filesystem layout.



Vector retriever

Core logic lives in src/retriever.py.

Typical responsibilities:

load sentence-transformer model (or any embedding backend),

open/load FAISS index from the configured path,

expose a clean interface:


from src.retriever import VectorRetriever

retriever = VectorRetriever.from_default()
results = retriever.search("Resetting SSH keys on Ubuntu", top_k=5, with_text=True)

for r in results:
    print(r.score, r.doc_id, r.chunk_id)
    print(r.text)


The retriever is used both by the CLI (for terminal search) and by the HTTP API.



CLI

A simple command-line interface is provided in src/cli.py.

Usage (from the project root):


python -m src.cli search "How to harden SSH on Ubuntu?" --top-k 5



What it does:

loads VectorRetriever via VectorRetriever.from_default(),

runs search(query, top_k),

prints score, doc_id, chunk_id and a short snippet of text for each hit.

Example output:


Query: 'How to harden SSH on Ubuntu?'
Top-3 results:

[1] score=0.8123  doc=linux/ssh_hardening.md  chunk=2
    Disable password logins, enforce key-based auth, and restrict root SSH access via PermitRootLogin no in sshd_config…

[2] score=0.7931  doc=linux/firewall_basics.md  chunk=1
    For SSH, only open port 22 (or a custom port) and limit access by source IP…

[3] score=0.7550  doc=cloud/bastion_host_guide.md  chunk=0
    A bastion host acts as a single hardened entry point for SSH into the private network…


HTTP API (FastAPI)

A minimal HTTP layer is implemented in src/api/app.py using FastAPI.


Endpoints

GET /health
Returns a simple health status:


{"status": "ok"}


POST /search
Request:

{
  "query": "How to harden SSH on Ubuntu?",
  "top_k": 3
}



Response:

{
  "query": "How to harden SSH on Ubuntu?",
  "results": [
    {
      "doc_id": "linux/ssh_hardening.md",
      "chunk_id": 2,
      "score": 0.8123,
      "text": "Disable password logins, enforce key-based auth..."
    }
    // ...
  ]
}



POST /chat (stub RAG endpoint)

For now, /chat:

runs retrieval with the given query,

returns a simple stub answer string,

includes retrieved chunks as context.

Request:


{
  "query": "Explain SSH hardening on Ubuntu using internal docs.",
  "top_k": 5
}


Response (example):

{
  "query": "Explain SSH hardening on Ubuntu using internal docs.",
  "answer": "This is a stub chat endpoint. The system retrieved 5 relevant chunks from the documentation...",
  "context": [
    {
      "doc_id": "linux/ssh_hardening.md",
      "chunk_id": 2,
      "score": 0.8123,
      "text": "Disable password logins, enforce key-based auth..."
    }
  ]
}


Running the API

Install dependencies (see Installation
), then:


uvicorn src.api.app:app --reload


Open the interactive docs at:

Swagger UI: http://127.0.0.1:8000/docs

ReDoc: http://127.0.0.1:8000/redoc



Installation

Create and activate a virtual environment (recommended):


python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows



Install dependencies:

pip install -r requirements.txt


For development (tests, linters, etc.):

pip install -r requirements-dev.txt



Build or place your FAISS index at the path configured in
configs/default.yaml / src/config.py (for example data/index/faiss_index.bin).

This repository does not prescribe a specific ingestion pipeline.
You can use your own script/notebooks to:

parse documents,

chunk text,

compute embeddings,

build the FAISS index.



Testing

Smoke tests for the API live in tests/test_api_smoke.py.

They do not require a real FAISS index or embedding model: the tests
monkeypatch get_retriever() to use a DummyRetriever.

Run tests from the project root:


pytest


(or, to run only the API smoke tests:)


pytest tests/test_api_smoke.py



Roadmap

Planned extensions:

✅ basic CLI search and FastAPI HTTP layer

🔲 ingestion pipeline (document parsing, chunking, embedding, index build)

🔲 proper LLM backend for /chat (OpenAI / local model)

🔲 ranking / reranking on top of vector search

🔲 eval harness for retrieval quality (R@k, MRR, etc.)

🔲 Dockerfile and deployment instructions (on-prem / cloud)



License

This project is licensed under the MIT License.
See LICENSE
 for full text.
