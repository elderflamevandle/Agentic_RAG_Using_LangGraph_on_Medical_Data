from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import chromadb

logger = logging.getLogger(__name__)


@dataclass
class ChromaCollections:
    qna: Any
    device: Any


class ChromaStore:
    """Thin wrapper around ChromaDB persistent client and collections."""

    def __init__(self, path: str):
        self.path = path
        self.client = chromadb.PersistentClient(path=path)

    def get_or_create(self, name: str):
        return self.client.get_or_create_collection(name=name)

    def ensure_collections(self, qna_name: str, device_name: str) -> ChromaCollections:
        return ChromaCollections(
            qna=self.get_or_create(qna_name),
            device=self.get_or_create(device_name),
        )

    @staticmethod
    def add_documents(
        collection: Any,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        batch_size: int = 256,
    ) -> None:
        """Batch insert to avoid large payloads."""
        if ids is None:
            ids = [str(i) for i in range(len(documents))]
        if metadatas is None:
            metadatas = [{} for _ in documents]

        for start in range(0, len(documents), batch_size):
            end = start + batch_size
            collection.add(
                documents=documents[start:end],
                metadatas=metadatas[start:end],
                ids=ids[start:end],
            )
        logger.info("Inserted %d docs into collection '%s'", len(documents), getattr(collection, "name", "unknown"))

    @staticmethod
    def query(collection: Any, query_text: str, top_k: int = 3) -> Tuple[str, List[str]]:
        """Return (joined_context, raw_docs)."""
        results = collection.query(query_texts=[query_text], n_results=top_k)
        docs = results.get("documents", [[]])[0] or []
        context = "\n".join(docs)
        return context, docs
