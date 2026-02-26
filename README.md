# Agentic RAG — Medical Q&A Pipeline

An **Agentic Retrieval-Augmented Generation** pipeline built with
[LangGraph](https://github.com/langchain-ai/langgraph) and
[ChromaDB](https://www.trychroma.com/).

The agent intelligently routes each medical query to the best available
source — a local Q&A collection, medical device manuals, or live web
search — then checks relevance before generating a concise answer.

---

## How it works

### Entry Point

```
CLI: python scripts/run_agent.py --query "..."
        │
        ▼
   main()  [run_agent.py]
        │
        ├─ get_settings()        — loads .env via pydantic-settings
        ├─ setup_logging()
        ├─ build_agent(settings) — compiles the LangGraph workflow
        └─ agent.invoke({"query": query})
                │
         (graph executes below)
```

### LangGraph Workflow

```
┌──────────────────────────────────────────────────────────────────────┐
│                        GraphState (TypedDict)                        │
│  query · route · context · source · is_relevant · iteration_count   │
│  prompt · response                                                   │
└──────────────────────────────────────────────────────────────────────┘

                              START
                                │
                                ▼
                         ┌────────────┐
                         │   router   │  LLM → RouteDecision schema
                         └─────┬──────┘  (Groq structured output)
                               │  state["route"] = one of 3 literals
                    ┌──────────┼──────────┐
                    ▼          ▼          ▼
             retrieve_qna  retrieve_    web_search
             (ChromaDB     device       (Serper API)
              qna coll.)   (ChromaDB
                            device coll.)
                    │          │          │
                    └──────────┴──────────┘
                                │
                                ▼
                      ┌──────────────────┐
                      │ relevance_checker │  LLM → RelevanceDecision schema
                      └────────┬─────────┘  → state["is_relevant"] = "Yes"/"No"
                               │
              ┌────────────────┴────────────────┐
              │  is_relevant == "No"             │  is_relevant == "Yes"
              │  AND iteration < MAX_ITERATIONS  │  (OR MAX_ITERATIONS hit)
              ▼                                  ▼
         web_search ◄──── retry loop ────   ┌─────────┐
              │                             │ augment │  builds RAG prompt
              └──► relevance_checker        └────┬────┘
                   (loops back up)               │
                                                 ▼
                                          ┌──────────┐
                                          │ generate │  LLM free-text answer
                                          └────┬─────┘
                                               │
                                              END
                                               │
                                    state["response"] + state["source"]
                                         printed to stdout
```

### State Mutations Per Node

```
router            → sets: route, source
retrieve_qna      → sets: context, source
retrieve_device   → sets: context, source
web_search        → sets: context, source
relevance_checker → sets: is_relevant
augment           → sets: prompt
generate          → sets: response
```

### Retry Loop Guard

```
relevance_checker → "No" → web_search (repeat)
                         ↑
                iteration_count++
                if iteration_count >= MAX_ITERATIONS:
                    force is_relevant = "Yes"  → breaks the loop
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
