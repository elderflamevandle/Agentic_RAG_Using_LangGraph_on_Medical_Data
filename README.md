# Agentic RAG (Production-Ready Skeleton)

This is a production-friendly refactor of your notebook into a small, modular Python package.

## What you get
- `scripts/ingest.py`: builds/updates Chroma collections from CSV datasets
- `scripts/run_agent.py`: runs the LangGraph agent end-to-end
- `src/agentic_rag/graph/*`: single LangGraph workflow (router -> retriever/web -> relevance check -> prompt -> generate)
- `src/agentic_rag/config.py`: `.env`-driven configuration via `pydantic-settings`

## Quickstart

1) Create `.env`
```bash
GROQ_API_KEY=...
SERPER_API_KEY=...
```

2) Install deps
```bash
pip install -r requirements.txt
```

3) Ingest (optional, but recommended)
```bash
python scripts/ingest.py
```

4) Run agent
```bash
python scripts/run_agent.py -q "What are the treatments for Kawasaki disease?"
```

## Notes
- Chroma will use its default embedding function unless you configure an explicit embedding pipeline.
- Relevance checker loops to Serper if retrieved context is not relevant, up to `MAX_ITERATIONS`.
