"""Data ingestion script — loads CSV datasets into ChromaDB.

Run from the repository root::

    python scripts/ingest.py

The script reads both dataset CSVs, preprocesses each row into a single
searchable text string, then bulk-inserts them into the corresponding
ChromaDB collections.  Run this once before using the agent, or again
whenever the source datasets are updated.

Configuration is driven entirely by environment variables / .env:
    MEDICAL_QA_CSV, MEDICAL_DEVICE_CSV, SAMPLE_ROWS, CHROMA_PATH, etc.
See .env.example for the full list.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Make the src/ package importable when running as a script from repo root.
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agentic_rag.config import get_settings
from agentic_rag.logging_config import setup_logging
from agentic_rag.vectorstore.chroma_store import ChromaStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data preprocessing helpers
# ---------------------------------------------------------------------------

def _prep_medical_qa(df: pd.DataFrame) -> pd.DataFrame:
    """Combine Q&A columns into a single searchable ``combined_text`` string.

    The resulting string format is::

        Question: <text>. Answer: <text>. Type: <qtype>.

    Parameters
    ----------
    df:
        Raw medical Q&A DataFrame (columns: ``qtype``, ``Question``, ``Answer``).

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with an added ``combined_text`` column.
    """
    df = df.copy()
    df["combined_text"] = (
        "Question: " + df["Question"].astype(str) + ". "
        + "Answer: " + df["Answer"].astype(str) + ". "
        + "Type: " + df.get("qtype", pd.Series([""] * len(df))).astype(str) + "."
    )
    return df


def _prep_device(df: pd.DataFrame) -> pd.DataFrame:
    """Combine device-manual columns into a single searchable string.

    The resulting string format is::

        Device Name: <name>. Model: <model>. Manufacturer: <mfr>.
        Indications: <text>. Contraindications: <text>.

    Parameters
    ----------
    df:
        Raw medical device DataFrame (columns: ``Device_Name``, ``Model_Number``,
        ``Manufacturer``, ``Indications_for_Use``, ``Contraindications``).

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with an added ``combined_text`` column.
    """
    df = df.copy()
    df["combined_text"] = (
        "Device Name: " + df["Device_Name"].astype(str) + ". "
        + "Model: " + df["Model_Number"].astype(str) + ". "
        + "Manufacturer: " + df["Manufacturer"].astype(str) + ". "
        + "Indications: " + df["Indications_for_Use"].astype(str) + ". "
        + "Contraindications: " + df["Contraindications"].fillna("None").astype(str) + "."
    )
    return df


# ---------------------------------------------------------------------------
# Main ingestion logic
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the full ingestion pipeline."""
    settings = get_settings()
    setup_logging(settings.LOG_LEVEL)

    logger.info("Starting ingestion | chroma_path=%s", settings.CHROMA_PATH)

    # --- Vector store setup ---
    store = ChromaStore(settings.CHROMA_PATH)
    collections = store.ensure_collections(settings.QNA_COLLECTION, settings.DEVICE_COLLECTION)

    # --- Load datasets ---
    qa_path = Path(settings.MEDICAL_QA_CSV)
    dev_path = Path(settings.MEDICAL_DEVICE_CSV)

    if not qa_path.exists():
        logger.error("Medical Q&A CSV not found: %s", qa_path)
        sys.exit(1)
    if not dev_path.exists():
        logger.error("Medical device CSV not found: %s", dev_path)
        sys.exit(1)

    logger.info("Loading Q&A dataset | path=%s", qa_path)
    qa = pd.read_csv(qa_path)
    logger.info("Loading device dataset | path=%s", dev_path)
    dev = pd.read_csv(dev_path)

    # --- Optional sampling (for local testing with large files) ---
    if settings.SAMPLE_ROWS > 0:
        logger.info("Sampling %d rows from each dataset.", settings.SAMPLE_ROWS)
        qa = qa.sample(min(settings.SAMPLE_ROWS, len(qa)), random_state=0).reset_index(drop=True)
        dev = dev.sample(min(settings.SAMPLE_ROWS, len(dev)), random_state=0).reset_index(drop=True)

    logger.info("Q&A rows to ingest: %d", len(qa))
    logger.info("Device rows to ingest: %d", len(dev))

    # --- Preprocess ---
    qa = _prep_medical_qa(qa)
    dev = _prep_device(dev)

    # --- Insert into ChromaDB ---
    ChromaStore.add_documents(
        collections.qna,
        documents=qa["combined_text"].tolist(),
        metadatas=qa.drop(columns=["combined_text"], errors="ignore").to_dict(orient="records"),
        ids=[f"qa_{i}" for i in range(len(qa))],
        batch_size=settings.CHROMA_BATCH_SIZE,
    )
    ChromaStore.add_documents(
        collections.device,
        documents=dev["combined_text"].tolist(),
        metadatas=dev.drop(columns=["combined_text"], errors="ignore").to_dict(orient="records"),
        ids=[f"dev_{i}" for i in range(len(dev))],
        batch_size=settings.CHROMA_BATCH_SIZE,
    )

    logger.info("Ingestion complete | chroma_path=%s", settings.CHROMA_PATH)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nIngestion interrupted by user.")
        sys.exit(0)
    except Exception as exc:
        logging.getLogger(__name__).exception("Ingestion failed: %s", exc)
        sys.exit(1)
