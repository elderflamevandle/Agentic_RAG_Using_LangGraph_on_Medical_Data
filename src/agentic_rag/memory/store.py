"""Long-term episodic memory store backed by ChromaDB.

Stores past Q&A interactions in a dedicated ChromaDB collection so they
can be retrieved semantically on future queries and injected into the prompt.
User preferences (topics, interaction counts) are persisted as a JSON sidecar
file — cheap, human-readable, and works on any volume mount.

Production note
---------------
Both the ChromaDB data directory (``CHROMA_PATH``) and the preferences file
(``MEMORY_PREFS_PATH``) must be on a **persistent volume** in Docker/K8s:

  Docker  : ``-v $(pwd)/data:/app/data``
  K8s EFS : ``ReadWriteMany`` PVC shared across all pods
  K8s EBS : ``ReadWriteOnce`` PVC — fine for single-pod deployments

No code changes are required when moving between environments — only the
mount path in the container needs to match the config values.
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path

import chromadb

logger = logging.getLogger(__name__)

# Cosine distance threshold — memories beyond this are considered irrelevant
_DISTANCE_THRESHOLD = 1.5


class AgentMemoryStore:
    """ChromaDB-backed episodic memory store with a JSON preferences sidecar.

    Parameters
    ----------
    chroma_path:
        Directory where ChromaDB persists its data (same root as the main
        vector store; a separate collection keeps memories isolated).
    collection_name:
        ChromaDB collection name for agent memories (default: ``agent_memories``).
    prefs_path:
        Path to the JSON file that holds user preferences.
    """

    def __init__(self, chroma_path: str, collection_name: str, prefs_path: str) -> None:
        self._client = chromadb.PersistentClient(path=chroma_path)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._prefs_path = Path(prefs_path)
        logger.info(
            "AgentMemoryStore ready | collection=%s | docs=%d",
            collection_name,
            self._collection.count(),
        )

    # ------------------------------------------------------------------
    # Episodic memory — past Q&A retrieval
    # ------------------------------------------------------------------

    def retrieve_similar(self, query: str, top_k: int) -> list[dict]:
        """Return up to ``top_k`` past interactions semantically similar to ``query``.

        Results are filtered by ``_DISTANCE_THRESHOLD`` so only genuinely
        relevant memories are returned.  Returns an empty list when the
        collection is empty or no relevant memories exist.

        Parameters
        ----------
        query:
            The current user question — used as the embedding search vector.
        top_k:
            Maximum number of memories to return.

        Returns
        -------
        list[dict]
            Each dict has keys: ``query``, ``response``, ``source``, ``route``,
            ``timestamp``, ``thread_id``.
        """
        count = self._collection.count()
        if count == 0:
            return []

        n = min(top_k, count)
        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=n,
                include=["metadatas", "distances"],
            )
            memories: list[dict] = []
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]
            for meta, dist in zip(metadatas, distances):
                if dist < _DISTANCE_THRESHOLD:
                    memories.append(meta)
            logger.debug("Memory retrieval | query=%r | found=%d", query, len(memories))
            return memories
        except Exception as exc:
            logger.warning("Memory retrieval failed: %s", exc)
            return []

    def save_interaction(
        self,
        *,
        query: str,
        response: str,
        source: str,
        route: str,
        thread_id: str,
    ) -> None:
        """Persist a completed Q&A interaction to the memory collection.

        The document text is ``"{query} {response}"`` so ChromaDB's default
        sentence-transformer embeddings capture the full semantic content.

        Parameters
        ----------
        query, response:
            The user question and agent answer.
        source:
            Human-readable retrieval source label.
        route:
            Routing decision (``retrieve_qna`` / ``retrieve_device`` / ``web_search``).
        thread_id:
            Session identifier — stored as metadata for future filtering.
        """
        doc_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        try:
            self._collection.add(
                ids=[doc_id],
                documents=[f"{query} {response}"],
                metadatas=[{
                    "query":     query,
                    "response":  response[:500],   # cap to avoid bloat
                    "source":    source,
                    "route":     route,
                    "timestamp": timestamp,
                    "thread_id": thread_id,
                }],
            )
            logger.info("Memory saved | thread_id=%s | total=%d", thread_id, self._collection.count())
        except Exception as exc:
            logger.warning("Memory save failed: %s", exc)

    # ------------------------------------------------------------------
    # Semantic preferences — lightweight JSON sidecar
    # ------------------------------------------------------------------

    def get_preferences(self) -> dict:
        """Load user preferences from the JSON sidecar file.

        Returns an empty dict if the file does not exist yet.
        """
        if self._prefs_path.exists():
            try:
                return json.loads(self._prefs_path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("Preferences load failed: %s", exc)
        return {}

    def update_preferences(self, **kwargs) -> None:
        """Merge ``kwargs`` into the preferences JSON and write it back.

        Always updates ``last_active`` to the current timestamp.
        Silently swallows write errors so a preferences failure never
        crashes the main pipeline.
        """
        prefs = self.get_preferences()
        prefs.update(kwargs)
        prefs["last_active"] = datetime.now().isoformat()
        try:
            self._prefs_path.parent.mkdir(parents=True, exist_ok=True)
            self._prefs_path.write_text(
                json.dumps(prefs, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("Preferences save failed: %s", exc)

    def increment_feedback(self, rating: str) -> None:
        """Increment the positive or negative feedback counter in preferences.

        Parameters
        ----------
        rating:
            ``"up"`` increments ``feedback_positive_count``;
            ``"down"`` increments ``feedback_negative_count``.
        """
        prefs = self.get_preferences()
        if rating == "up":
            prefs["feedback_positive_count"] = prefs.get("feedback_positive_count", 0) + 1
        elif rating == "down":
            prefs["feedback_negative_count"] = prefs.get("feedback_negative_count", 0) + 1
        prefs["total_interactions"] = prefs.get("total_interactions", 0) + 1
        self.update_preferences(**prefs)
