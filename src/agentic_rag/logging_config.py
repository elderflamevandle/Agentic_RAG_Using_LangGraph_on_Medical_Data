"""Logging configuration for the Agentic RAG pipeline.

Call ``setup_logging()`` once at application startup (in ``scripts/ingest.py``
and ``scripts/run_agent.py``).  All other modules use the standard
``logging.getLogger(__name__)`` pattern — no further setup needed.

Output destinations
-------------------
Console (stdout)
    Human-readable, level-filtered stream.

Log file  (``logs/agentic_rag_YYYY-MM-DD.log``)
    Plain-text, rotated at midnight, 30 daily files retained before deletion.
    The ``logs/`` directory is created automatically on first run.

Log format
----------
::

    2026-02-25 14:30:01 | INFO     | agentic_rag.graph.nodes | Router decision: web_search
"""
from __future__ import annotations

import logging
import sys
from datetime import date
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

# Project root is three levels above this file:
#   src/agentic_rag/logging_config.py → parents[2] = repo root
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = _PROJECT_ROOT / "logs"

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: str = "INFO") -> None:
    """Configure console + rotating-file logging for the pipeline.

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
    - Appends to ``logs/agentic_rag_<today>.log`` (rotates at midnight).
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

    # --- Rotating file handler ---
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"agentic_rag_{date.today().isoformat()}.log"
    file_handler = TimedRotatingFileHandler(
        filename=str(log_file),
        when="midnight",       # rotate at midnight
        interval=1,            # every 1 day
        backupCount=30,        # keep 30 days of history
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    logging.getLogger(__name__).debug(
        "Logging initialised | level=%s | log_file=%s", level.upper(), log_file
    )
