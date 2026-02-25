"""ChromaDB wrapper for persistent vector storage and retrieval.

Usage example::

    store = ChromaStore("./chroma_db")
    collections = store.ensure_collections("medical_q_n_a", "medical_device_manual")

    ChromaStore.add_documents(collections.qna, documents=["doc1", "doc2"])
    context, raw_docs = ChromaStore.query(collections.qna, "What is aspirin?", top_k=3)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import chromadb

from ..exceptions import VectorStoreError

logger = logging.getLogger(__name__)


@dataclass
class ChromaCollections:
    """Container for the two named ChromaDB collections used by this pipeline.

    Attributes
    ----------
    qna:
        Collection holding medical Q&A documents.
    device:
        Collection holding medical device manual documents.
    """

    qna: Any
    device: Any


class ChromaStore:
    """Thin wrapper around a ChromaDB ``PersistentClient``.

    Parameters
    ----------
    path:
        Filesystem path where ChromaDB persists its data files.
        The directory is created automatically if it does not exist.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        try:
            self.client = chromadb.PersistentClient(path=path)
        except Exception as exc:
            raise VectorStoreError(f"Failed to open ChromaDB at '{path}'") from exc
        logger.debug("ChromaDB client initialised | path=%s", path)

    def get_or_create(self, name: str) -> Any:
        """Get an existing collection by name, or create it if absent.

        Parameters
        ----------
        name:
            Collection name (valid ChromaDB identifier).

        Returns
        -------
        chromadb.Collection
        """
        collection = self.client.get_or_create_collection(name=name)
        logger.debug("Collection ready | name=%s", name)
        return collection

    def ensure_collections(self, qna_name: str, device_name: str) -> ChromaCollections:
        """Get or create both pipeline collections and return them together.

        Parameters
        ----------
        qna_name:
            Name for the medical Q&A collection.
        device_name:
            Name for the medical device manual collection.

        Returns
        -------
        ChromaCollections
        """
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
        """Batch-insert documents into a ChromaDB collection.

        Splits large document lists into smaller batches to avoid
        exceeding ChromaDB's payload limits.

        Parameters
        ----------
        collection:
            Target ChromaDB collection.
        documents:
            Plain-text strings to embed and store.
        metadatas:
            Optional per-document metadata dicts (same length as ``documents``).
            Defaults to empty dicts when omitted.
        ids:
            Optional unique string IDs (same length as ``documents``).
            Defaults to ``"0"``, ``"1"``, … when omitted.
        batch_size:
            Documents per ChromaDB ``add()`` call.
            Pass ``Settings.CHROMA_BATCH_SIZE`` from the caller to keep this
            driven by configuration.
        """
        total = len(documents)
        if ids is None:
            ids = [str(i) for i in range(total)]
        if metadatas is None:
            metadatas = [{} for _ in range(total)]

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            try:
                collection.add(
                    documents=documents[start:end],
                    metadatas=metadatas[start:end],
                    ids=ids[start:end],
                )
            except Exception as exc:
                raise VectorStoreError(
                    f"Batch insert failed | range=[{start}, {end}) "
                    f"| collection={getattr(collection, 'name', 'unknown')}"
                ) from exc
            logger.debug(
                "Batch inserted | range=[%d, %d) | collection=%s",
                start,
                end,
                getattr(collection, "name", "unknown"),
            )

        logger.info(
            "Ingestion complete | collection=%s | total_docs=%d",
            getattr(collection, "name", "unknown"),
            total,
        )

    @staticmethod
    def query(
        collection: Any,
        query_text: str,
        top_k: int = 3,
    ) -> Tuple[str, List[str]]:
        """Retrieve the top-k nearest documents for a query string.

        Parameters
        ----------
        collection:
            Source ChromaDB collection to search.
        query_text:
            The user query to embed and search with.
        top_k:
            Number of nearest-neighbour documents to return.

        Returns
        -------
        tuple[str, list[str]]
            ``(joined_context, raw_docs)`` — ``joined_context`` is all
            retrieved documents joined by newlines, ready to drop into a prompt.
        """
        try:
            results = collection.query(query_texts=[query_text], n_results=top_k)
        except Exception as exc:
            raise VectorStoreError(
                f"ChromaDB query failed | collection={getattr(collection, 'name', 'unknown')}"
            ) from exc
        docs: List[str] = results.get("documents", [[]])[0] or []
        context = "\n".join(docs)
        return context, docs
