# =============================================================================
# Agentic RAG — Dockerfile
# =============================================================================
#
# Build:
#   docker build -t agentic-rag .
#
# Step 1 — ingest datasets into ChromaDB (run once):
#   docker run --rm \
#     --env-file .env \
#     -v "$(pwd)/chroma_db:/app/chroma_db" \
#     agentic-rag python scripts/ingest.py
#
# Step 2 — run a query:
#   docker run --rm \
#     --env-file .env \
#     -v "$(pwd)/chroma_db:/app/chroma_db" \
#     -v "$(pwd)/logs:/app/logs" \
#     agentic-rag python scripts/run_agent.py -q "What causes Kawasaki disease?"
#
# Notes:
#   • Never bake secrets into the image — always pass via --env-file or -e.
#   • Mount chroma_db as a volume so the vector store persists between runs.
#   • Mount logs as a volume to access log files from the host machine.
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1 — dependency builder
# Install Python wheels into an isolated prefix so the final stage stays lean.
# -----------------------------------------------------------------------------
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build tools needed by some packages (e.g. chromadb's native deps).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements file first so Docker can cache this layer.
# The expensive pip install is only re-run when requirements.txt changes.
COPY requirements.txt .

RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# -----------------------------------------------------------------------------
# Stage 2 — runtime image
# Copies only the installed wheels and the application source — no build tools.
# -----------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

# Metadata labels (optional but good practice for production images).
LABEL org.opencontainers.image.title="Agentic RAG Pipeline"
LABEL org.opencontainers.image.description="LangGraph + ChromaDB + Groq medical Q&A agent"
LABEL org.opencontainers.image.version="0.1.0"

WORKDIR /app

# --- Copy installed Python packages from builder stage ---
COPY --from=builder /install /usr/local

# --- Copy application source ---
COPY src/       ./src/
COPY scripts/   ./scripts/
COPY datasets/  ./datasets/

# --- Persistent directories ---
# chroma_db and logs are intended to be mounted as volumes (see run commands
# at the top of this file).  We pre-create them here so the non-root user
# owns them even if no volume is mounted.
RUN mkdir -p /app/chroma_db /app/logs

# --- Environment variables ---
# PYTHONPATH: makes "import agentic_rag" work from any working directory.
# PYTHONUNBUFFERED: forces stdout/stderr to flush immediately (important for
#   Docker log streaming — otherwise log lines may appear delayed).
# PYTHONDONTWRITEBYTECODE: prevents .pyc files cluttering the container FS.
ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# --- Non-root user (security best practice) ---
# Running as root inside a container is a security risk if the container
# is ever compromised.  A dedicated user limits the blast radius.
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app

USER appuser

# --- Default command ---
# Shows the CLI help by default so running the image without arguments is safe.
# Override for normal use:
#   docker run ... agentic-rag python scripts/ingest.py
#   docker run ... agentic-rag python scripts/run_agent.py -q "..."
CMD ["python", "scripts/run_agent.py", "--help"]
