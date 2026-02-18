from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

# Ensure local package imports work when running from repo root.
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agentic_rag.config import get_settings
from agentic_rag.logging_config import setup_logging
from agentic_rag.vectorstore.chroma_store import ChromaStore


logger = logging.getLogger(__name__)


def _prep_medical_qa(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["combined_text"] = (
        "Question: " + df["Question"].astype(str) + ". "
        "Answer: " + df["Answer"].astype(str) + ". "
        "Type: " + df.get("qtype", "").astype(str) + ". "
    )
    return df


def _prep_device(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["combined_text"] = (
        "Device Name: " + df["Device_Name"].astype(str) + ". "
        "Model: " + df["Model_Number"].astype(str) + ". "
        "Manufacturer: " + df["Manufacturer"].astype(str) + ". "
        "Indications: " + df["Indications_for_Use"].astype(str) + ". "
        "Contraindications: " + df["Contraindications"].fillna("None").astype(str)
    )
    return df


def main() -> None:
    settings = get_settings()
    setup_logging(settings.LOG_LEVEL)

    store = ChromaStore(settings.CHROMA_PATH)
    collections = store.ensure_collections(settings.QNA_COLLECTION, settings.DEVICE_COLLECTION)

    # Read CSVs
    qa = pd.read_csv(settings.MEDICAL_QA_CSV)
    dev = pd.read_csv(settings.MEDICAL_DEVICE_CSV)

    if settings.SAMPLE_ROWS and settings.SAMPLE_ROWS > 0:
        qa = qa.sample(settings.SAMPLE_ROWS, random_state=0).reset_index(drop=True)
        dev = dev.sample(settings.SAMPLE_ROWS, random_state=0).reset_index(drop=True)

    qa = _prep_medical_qa(qa)
    dev = _prep_device(dev)

    # Insert
    ChromaStore.add_documents(
        collections.qna,
        documents=qa["combined_text"].tolist(),
        metadatas=qa.drop(columns=["combined_text"], errors="ignore").to_dict(orient="records"),
        ids=[f"qa_{i}" for i in range(len(qa))],
    )
    ChromaStore.add_documents(
        collections.device,
        documents=dev["combined_text"].tolist(),
        metadatas=dev.drop(columns=["combined_text"], errors="ignore").to_dict(orient="records"),
        ids=[f"dev_{i}" for i in range(len(dev))],
    )

    logger.info("Ingestion complete. Chroma path: %s", settings.CHROMA_PATH)


if __name__ == "__main__":
    main()
