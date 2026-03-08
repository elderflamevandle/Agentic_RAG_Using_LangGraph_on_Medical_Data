# =============================================================================
# Agentic RAG — Dockerfile
# =============================================================================
#
# Build:
#   docker build -t agentic-rag .
#
# ── PowerShell (Windows) ─────────────────────────────────────────────────────
#
# Step 1 — ingest datasets into ChromaDB (run once):
#   docker run --rm --env-file .env `
#     -v "${PWD}/chroma_db:/app/chroma_db" `
#     -v "${PWD}/datasets:/app/datasets" `
#     agentic-rag python scripts/ingest.py
#
# Step 2 — run the Streamlit UI:
#   docker run --rm --env-file .env `
#     -v "${PWD}/chroma_db:/app/chroma_db" `
#     -v "${PWD}/logs:/app/logs" `
#     -p 8501:8501 `
#     agentic-rag
#
# Step 3 (optional) — run a single CLI query:
#   docker run --rm --env-file .env `
#     -v "${PWD}/chroma_db:/app/chroma_db" `
#     agentic-rag python scripts/run_agent.py -q "What causes Kawasaki disease?"
#
# ── Git Bash (Windows) ───────────────────────────────────────────────────────
#
# IMPORTANT: Git Bash mangles absolute container paths (e.g. /app/...) into
# Windows paths. Prefix every docker run command with MSYS_NO_PATHCONV=1.
# $(pwd) still expands correctly — only the container-side paths are affected.
#
# Step 1 — ingest datasets into ChromaDB (run once):
#   MSYS_NO_PATHCONV=1 docker run --rm --env-file .env \
#     -v "$(pwd)/chroma_db:/app/chroma_db" \
#     -v "$(pwd)/datasets:/app/datasets" \
#     agentic-rag python scripts/ingest.py
#
# Step 2 — run the Streamlit UI:
#   MSYS_NO_PATHCONV=1 docker run --rm --env-file .env \
#     -v "$(pwd)/chroma_db:/app/chroma_db" \
#     -v "$(pwd)/logs:/app/logs" \
#     -p 8501:8501 agentic-rag
#
# Step 3 (optional) — run a single CLI query:
#   MSYS_NO_PATHCONV=1 docker run --rm --env-file .env \
#     -v "$(pwd)/chroma_db:/app/chroma_db" \
#     agentic-rag python scripts/run_agent.py -q "What causes Kawasaki disease?"
#
# ── Linux / macOS ────────────────────────────────────────────────────────────
#
# Step 1:  docker run --rm --env-file .env \
#            -v "$(pwd)/chroma_db:/app/chroma_db" \
#            -v "$(pwd)/datasets:/app/datasets" \
#            agentic-rag python scripts/ingest.py
#
# Step 2:  docker run --rm --env-file .env \
#            -v "$(pwd)/chroma_db:/app/chroma_db" \
#            -v "$(pwd)/logs:/app/logs" \
#            -p 8501:8501 agentic-rag
#
# =============================================================================

FROM python:3.12-slim

WORKDIR /app

# ---------------------------------------------------------------------------
# System libraries
#   build-essential + gcc  — compile any packages that lack pre-built wheels
#   libgomp1               — OpenMP runtime required by onnxruntime (chromadb)
# ---------------------------------------------------------------------------
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Python dependencies
# Copy requirements first so Docker can cache this layer — the expensive
# pip install is only re-run when requirements.txt changes.
# ---------------------------------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Application source
# datasets/ is intentionally excluded — mount it as a volume at runtime
# so the image stays lean and datasets never get baked in.
# ---------------------------------------------------------------------------
COPY src/      ./src/
COPY scripts/  ./scripts/
COPY app.py    ./app.py

# ---------------------------------------------------------------------------
# Runtime directories
# Pre-create as root before switching to appuser so the non-root user owns
# them even when no volume is mounted.
# ---------------------------------------------------------------------------
RUN mkdir -p /app/chroma_db /app/logs /app/datasets

# ---------------------------------------------------------------------------
# Environment variables
#   PYTHONPATH        — makes "import agentic_rag" work from any directory
#   PYTHONUNBUFFERED  — flush stdout/stderr immediately (important for logs)
#   PYTHONDONTWRITEBYTECODE — no .pyc files in the container filesystem
# ---------------------------------------------------------------------------
ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# ---------------------------------------------------------------------------
# Non-root user (security best practice)
# ---------------------------------------------------------------------------
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8501

# Default: launch the Streamlit UI.
# Override for CLI use:
#   docker run ... agentic-rag python scripts/ingest.py
#   docker run ... agentic-rag python scripts/run_agent.py -q "..."
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
