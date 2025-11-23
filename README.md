# Industrial Docs RAG Chatbot

A small end-to-end demo of a retrieval-augmented chatbot for industrial
documentation (PDF, Markdown, plain text).

The goal is to show a clean, production-oriented Python structure for:

- ingesting documents (regulations, manuals, FAQs),
- building a vector index over text chunks,
- exposing a FastAPI `/chat` endpoint that:
  - retrieves the most relevant passages,
  - calls an LLM client (or a stub) to generate an answer,
  - returns both the answer and the supporting context.

This repository is designed as a **portfolio-ready** project:
clear layout, configuration via `configs/default.yaml`, tests, and
Jupyter notebooks for quick inspection and evaluation.
