# Generic RAG Pipeline

Dataset-independent, terminal-based retrieval augmented generation (RAG). Put documents in `data/`, run `python main.py --rebuild`, then ask questions with `python main.py`. The local vector store is a persistent Chroma collection.

## Quick start

```powershell
Copy-Item WALMART_2020_10K.pdf data\
Copy-Item .env.example .env
python -m pip install -r requirements.txt
python main.py --rebuild
python main.py
```

Set `GROQ_API_KEY` in `.env` to use Groq generation. The default model is `openai/gpt-oss-20b` (hosted by Groq; it does not require an OpenAI key). Without an LLM key, retrieval runs locally and an extractive, source-backed fallback is returned instead of fabricated answers.

## Configuration

All settings live in `.env`: dataset path, hierarchical chunking, top-k, similarity threshold, retrieval mode (`vector`, `bm25`, or `hybrid`), embeddings, reranker, and LLM values. Parent chunks provide context; their smaller child chunks are embedded and searched. Changing datasets requires only replacing files in `DATA_DIR` and running `python main.py --rebuild`.

Supported formats: PDF, TXT, Markdown, DOCX, CSV, JSON, HTML, XML, RST, and common source/text files. Corrupt and unsupported inputs are logged and skipped.
