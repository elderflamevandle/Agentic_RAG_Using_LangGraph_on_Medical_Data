"""Custom exceptions for the Agentic RAG pipeline.

Exception hierarchy
-------------------
::

    AgenticRAGError          ← base for all project exceptions
    ├── ConfigurationError   ← missing / invalid env vars or Settings fields
    ├── IngestionError       ← CSV → ChromaDB ingestion failures
    ├── VectorStoreError     ← ChromaDB read / write failures
    ├── LLMError             ← LLM initialisation or invocation failures
    └── SearchError          ← Serper web-search failures

Usage example::

    from agentic_rag.exceptions import VectorStoreError

    try:
        results = collection.query(...)
    except Exception as exc:
        raise VectorStoreError("ChromaDB query failed") from exc
"""
from __future__ import annotations


class AgenticRAGError(Exception):
    """Base exception for all Agentic RAG pipeline errors.

    Catch this class to handle any project-specific error in one place.
    """


class ConfigurationError(AgenticRAGError):
    """Raised when required configuration is missing or invalid.

    Examples
    --------
    - A required API key environment variable is not set.
    - A settings value falls outside its permitted range.
    """


class IngestionError(AgenticRAGError):
    """Raised when the CSV → ChromaDB ingestion pipeline fails.

    Examples
    --------
    - Source CSV file not found.
    - DataFrame preprocessing produces unexpected columns.
    - Batch insert to ChromaDB is rejected.
    """


class VectorStoreError(AgenticRAGError):
    """Raised when a ChromaDB operation fails.

    Examples
    --------
    - ``PersistentClient`` cannot open or create the database directory.
    - ``collection.add()`` rejects documents (type error, size limit, etc.).
    - ``collection.query()`` raises an internal ChromaDB error.
    """


class LLMError(AgenticRAGError):
    """Raised when LLM initialisation or invocation fails unexpectedly.

    Examples
    --------
    - ``ChatGroq`` cannot connect to the Groq API.
    - An LLM response cannot be parsed even after retries.
    """


class SearchError(AgenticRAGError):
    """Raised when the Serper web-search tool fails.

    Examples
    --------
    - Invalid or expired Serper API key.
    - Serper API returns an HTTP error response.
    - Network timeout during the search request.
    """
