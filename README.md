# Agentic RAG — Medical Q&A Pipeline

An **Agentic Retrieval-Augmented Generation** pipeline built with
[LangGraph](https://github.com/langchain-ai/langgraph) and
[ChromaDB](https://www.trychroma.com/).

The agent intelligently routes each medical query to the best available
source — a local Q&A collection, medical device manuals, or live web
search — then checks relevance before generating a concise answer.

---

## How it works

```
User query
    │
    ▼
 router  ──────────────────────────────────────┐
    │  (LLM decides which source to use)        │
    ├─► retrieve_qna     (ChromaDB — Q&A)        │
    ├─► retrieve_device  (ChromaDB — devices)    │
    └─► web_search       (Serper API)            │
             │                                  │
             ▼                                  │
    relevance_checker                           │
             │                                  │
             ├─► [relevant]  augment → generate → END
             └─► [not relevant, retry] ─────────┘
                 (up to MAX_ITERATIONS times)
```

---

## Project structure

```
.
├── .env.example                   # Template — copy to .env and fill in keys
├── requirements.txt               # Pinned Python dependencies
├── datasets/
│   ├── medical_QnA_Dataset.csv    # 60k medical Q&A rows
│   └── medical_device_manuals_dataset.csv
├── notebooks/
│   └── agentic_RAG_pipeline.ipynb # Development / experimentation notebook
├── scripts/
│   ├── ingest.py                  # Populate ChromaDB from CSV datasets
│   └── run_agent.py               # CLI — ask a question and get an answer
└── src/agentic_rag/
    ├── config.py                  # Pydantic-settings configuration
    ├── llm.py                     # LLM factory (ChatGroq)
    ├── logging_config.py          # Logging setup
    ├── graph/
    │   ├── state.py               # GraphState TypedDict
    │   ├── schemas.py             # Pydantic structured-output schemas
    │   ├── nodes.py               # Node factory functions
    │   └── build.py               # Workflow assembly & compilation
    ├── tools/
    │   └── web_search.py          # Google Serper wrapper
    └── vectorstore/
        └── chroma_store.py        # ChromaDB client wrapper
```

---

## Quickstart

### 1. Set up environment variables

```bash
cp .env.example .env
# Open .env and fill in GROQ_API_KEY and SERPER_API_KEY
```

Get your keys from:
- **Groq** — https://console.groq.com
- **Serper** — https://serper.dev

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Ingest datasets into ChromaDB

Run once before using the agent (and again when datasets change):

```bash
python scripts/ingest.py
```

To do a quick test with a small subset first:

```bash
SAMPLE_ROWS=500 python scripts/ingest.py
```

### 4. Ask a question

```bash
python scripts/run_agent.py -q "What are the treatments for Kawasaki disease?"
python scripts/run_agent.py -q "How does an insulin pump work?"
python scripts/run_agent.py -q "What are contraindications for a ventilator?"
```

---

## Configuration reference

All settings can be overridden via environment variables or `.env`.

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | — | **Required.** Groq API key |
| `SERPER_API_KEY` | — | **Required.** Serper search API key |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model to use |
| `TEMPERATURE` | `0.3` | LLM sampling temperature |
| `CHROMA_PATH` | `./chroma_db` | ChromaDB persistence directory |
| `QNA_COLLECTION` | `medical_q_n_a` | ChromaDB collection for Q&A data |
| `DEVICE_COLLECTION` | `medical_device_manual` | ChromaDB collection for device data |
| `TOP_K` | `3` | Documents to retrieve per query |
| `CHROMA_BATCH_SIZE` | `256` | Documents per ChromaDB insert batch |
| `ANSWER_WORD_LIMIT` | `80` | Soft word-count target for answers |
| `MAX_ITERATIONS` | `3` | Max relevance-check retries before forcing generation |
| `SAMPLE_ROWS` | `0` | Rows per dataset to sample (0 = all) |
| `LOG_LEVEL` | `INFO` | Python log level |

---

## Notes

- **Embeddings**: ChromaDB uses its default
  [sentence-transformers](https://www.sbert.net/) embedding function.
  No additional configuration is required.
- **Relevance loop**: If the retrieved context is judged irrelevant, the
  agent automatically falls back to a live web search and retries up to
  `MAX_ITERATIONS` times before forcing an answer.
- **Security**: Never commit your real `.env` file — it is listed in
  `.gitignore`.  Share `.env.example` instead.
