"""CLI entry point for the agentic RAG pipeline.

Run from the repository root::

    python scripts/run_agent.py --query "What are the treatments for Kawasaki disease?"
    python scripts/run_agent.py -q "What is an insulin pump?"

The agent will route the query to the most relevant source
(local Q&A collection, device manuals, or live web search),
check relevance, and return a concise answer.

Prerequisites
-------------
1. Copy .env.example → .env and fill in your API keys.
2. Run  ``python scripts/ingest.py``  to populate the vector store.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Make the src/ package importable when running as a script from repo root.
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agentic_rag.config import get_settings
from agentic_rag.graph import build_agent
from agentic_rag.logging_config import setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse and return CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run the Agentic RAG pipeline (LangGraph) from the command line.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            '  python scripts/run_agent.py -q "What causes Kawasaki disease?"\n'
            '  python scripts/run_agent.py --query "How does an insulin pump work?"\n'
        ),
    )
    parser.add_argument(
        "--query", "-q",
        required=True,
        help="The medical question to answer.",
    )
    return parser.parse_args()


def main() -> None:
    """Load config, build the agent, run the query, and print the answer."""
    settings = get_settings()
    setup_logging(settings.LOG_LEVEL)

    args = parse_args()
    query = args.query.strip()

    if not query:
        logger.error("Query cannot be empty.")
        sys.exit(1)

    logger.info("Building agent...")
    agent = build_agent(settings)

    logger.info("Running query | query=%r", query)
    result = agent.invoke({"query": query})

    response = result.get("response", "").strip()
    source = result.get("source", "unknown")

    print("\n" + "=" * 60)
    print(f"  Source : {source}")
    print("=" * 60)
    print(response)
    print("=" * 60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAborted by user.")
        sys.exit(0)
    except Exception as exc:
        logging.getLogger(__name__).exception("Agent run failed: %s", exc)
        sys.exit(1)
