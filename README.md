# Local RAG AI Assistant with Microsoft Foundry Local

A fully local Retrieval-Augmented Generation (RAG) question-answering system using Microsoft Foundry Local, SQLite and semantic retrieval.

## Features
- Local LLM inference
- Local embedding generation
- Document chunking and ingestion
- SQLite storage
- Cosine similarity retrieval
- Source-grounded answers
- Unsupported-question fallback
- Interactive command-line interface

## Models
- Embedding: qwen3-embedding-0.6b
- Chat: phi-4-mini

## Usage
1. Install dependencies: python -m pip install -r requirements.txt
2. Place .txt documents in data/documents/
3. Run ingestion: python src/ingest.py
4. Start assistant: python main.py

## Retrieval
Top-K: 2
Minimum similarity threshold: 0.35

## Run
python main.py
