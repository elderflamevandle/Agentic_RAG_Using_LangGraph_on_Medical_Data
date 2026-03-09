"""Logging configuration for the Agentic RAG pipeline.

Call ``setup_logging()`` once at application startup (in ``scripts/ingest.py``
and ``scripts/run_agent.py``).  All other modules use the standard
``logging.getLogger(__name__)`` pattern — no further setup needed.

Output destinations
-------------------
Console (stdout)
    Human-readable, level-filtered stream.

Log file  (``logs/session_<UUID>_<timestamp>.log``)
    One log file per session/run. Each app launch creates a new file.
    The ``logs/`` directory is created automatically on first run.

Log format
----------
::

    2026-02-25 14:30:01 | INFO     | agentic_rag.graph.nodes | Router decision: web_search
"""
from __future__ import annotations

import logging
import sys
import uuid
from datetime import datetime
from pathlib import Path

# Project root is three levels above this file:
#   src/agentic_rag/logging_config.py → parents[2] = repo root
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = _PROJECT_ROOT / "logs"

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: str = "INFO") -> None:
    """Configure console + per-session file logging for the pipeline.

    Each call to this function (when no handlers exist yet) creates a
    unique log file named ``session_<short-uuid>_<timestamp>.log``.
    This means every app launch / session gets its own log file.

    Idempotent — subsequent calls only update the root logger's level;
    handlers are never added more than once.

    Parameters
    ----------
    level:
        Python logging level string: ``DEBUG | INFO | WARNING | ERROR | CRITICAL``.
        Case-insensitive.  Defaults to ``"INFO"``.

    Side effects
    ------------
    - Creates ``logs/`` directory if it does not exist.
    - Creates a new ``logs/session_<uuid>_<timestamp>.log`` per run.
    - Writes to ``stdout``.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger()

    # Guard: only add handlers once per process.
    if root.handlers:
        root.setLevel(numeric_level)
        return

    root.setLevel(numeric_level)
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # --- Console handler (stdout) ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    # --- Per-session file handler ---
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    session_id = uuid.uuid4().hex[:8]
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = LOG_DIR / f"session_{session_id}_{timestamp}.log"

    file_handler = logging.FileHandler(
        filename=str(log_file),
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    logging.getLogger(__name__).debug(
        "Logging initialised | level=%s | session=%s | log_file=%s",
        level.upper(), session_id, log_file,
    )
