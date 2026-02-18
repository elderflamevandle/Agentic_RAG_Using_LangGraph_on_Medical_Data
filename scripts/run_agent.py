from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure local package imports work when running from repo root.
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agentic_rag.config import get_settings
from agentic_rag.logging_config import setup_logging
from agentic_rag.graph import build_agent

logger = logging.getLogger(__name__)


def main():
    settings = get_settings()
    setup_logging(settings.LOG_LEVEL)

    parser = argparse.ArgumentParser(description="Run Agentic RAG (LangGraph) locally.")
    parser.add_argument("--query", "-q", required=True, help="User query")
    args = parser.parse_args()

    agent = build_agent(settings)

    final = agent.invoke({"query": args.query})
    print("\n=== RESPONSE ===\n")
    print(final.get("response", ""))


if __name__ == "__main__":
    main()
