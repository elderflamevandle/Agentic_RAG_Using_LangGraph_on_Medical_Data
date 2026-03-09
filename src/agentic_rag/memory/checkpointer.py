"""Checkpointer factory — swaps backend based on MEMORY_BACKEND setting.

Development  : SqliteSaver  → local .db file   (zero infra)
Docker       : SqliteSaver  → volume-mounted file (add -v flag)
Production   : PostgresSaver → Amazon RDS / any Postgres (set DATABASE_URL)

Switching from SQLite to Postgres requires only a config change:
    MEMORY_BACKEND=postgres
    DATABASE_URL=postgresql://user:pass@rds-host:5432/agenticrag
No application code changes needed.
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from ..config import Settings

logger = logging.getLogger(__name__)


def build_checkpointer(settings: Settings):
    """Return the appropriate LangGraph checkpointer for the current environment.

    Parameters
    ----------
    settings:
        Runtime configuration — reads ``MEMORY_BACKEND``, ``MEMORY_SQLITE_PATH``,
        and ``DATABASE_URL``.

    Returns
    -------
    BaseCheckpointSaver
        A compiled-ready checkpointer instance.

    Raises
    ------
    ImportError
        If ``MEMORY_BACKEND=postgres`` but the required packages are not installed.
    ValueError
        If ``MEMORY_BACKEND=postgres`` but ``DATABASE_URL`` is empty.
    """
    backend = settings.MEMORY_BACKEND.lower()

    if backend == "postgres":
        if not settings.DATABASE_URL:
            raise ValueError(
                "DATABASE_URL must be set when MEMORY_BACKEND=postgres. "
                "Example: postgresql://user:pass@host:5432/agenticrag"
            )
        try:
            from langgraph.checkpoint.postgres import PostgresSaver  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "PostgresSaver not available. Install with:\n"
                "  pip install langgraph-checkpoint-postgres psycopg2-binary"
            ) from exc

        logger.info("Memory checkpointer: PostgresSaver | url=***")
        return PostgresSaver.from_conn_string(settings.DATABASE_URL)

    # Default: SQLite — works locally and in Docker with a volume mount.
    # We create the sqlite3.Connection directly (with check_same_thread=False)
    # so the saver can be used across Streamlit's multiple threads without
    # "SQLite objects created in a thread can only be used in that same thread"
    # errors.  SqliteSaver.from_conn_string() is a @contextmanager and cannot
    # be returned directly.
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "SqliteSaver not available. Install with:\n"
            "  pip install langgraph-checkpoint-sqlite"
        ) from exc

    db_path = settings.MEMORY_SQLITE_PATH
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    logger.info("Memory checkpointer: SqliteSaver | path=%s", db_path)
    return SqliteSaver(conn)
