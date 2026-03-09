from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized, environment-driven configuration.

    All values can be overridden via environment variables or a .env file.
    See .env.example for a full list of available options.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # -------------------------------------------------------------------------
    # API keys (required — must be present in environment or .env)
    # -------------------------------------------------------------------------
    GROQ_API_KEY: str = Field(..., description="Groq cloud API key")
    SERPER_API_KEY: str = Field(..., description="Google Serper search API key")

    # -------------------------------------------------------------------------
    # LLM parameters
    # -------------------------------------------------------------------------
    GROQ_MODEL: str = Field(
        default="llama-3.3-70b-versatile",
        description="Groq model identifier to use for all LLM calls",
    )
    TEMPERATURE: float = Field(
        default=0.3,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0 = deterministic, higher = more creative)",
    )

    # -------------------------------------------------------------------------
    # Vector store / ChromaDB
    # -------------------------------------------------------------------------
    CHROMA_PATH: str = Field(
        default="./chroma_db",
        description="Directory path where ChromaDB persists its data",
    )
    QNA_COLLECTION: str = Field(
        default="medical_q_n_a",
        description="ChromaDB collection name for medical Q&A data",
    )
    DEVICE_COLLECTION: str = Field(
        default="medical_device_manual",
        description="ChromaDB collection name for medical device manual data",
    )
    CHROMA_BATCH_SIZE: int = Field(
        default=256,
        ge=1,
        le=2048,
        description="Number of documents to insert per ChromaDB batch call",
    )

    # -------------------------------------------------------------------------
    # Retrieval settings
    # -------------------------------------------------------------------------
    TOP_K: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Number of nearest-neighbor documents to retrieve from the vector store",
    )

    # -------------------------------------------------------------------------
    # Prompting / answer limits
    # -------------------------------------------------------------------------
    ANSWER_WORD_LIMIT: int = Field(
        default=150,
        ge=10,
        le=300,
        description="Soft word-count target injected into the answer prompt",
    )
    MAX_ITERATIONS: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum relevance-check retries before forcing an answer",
    )

    # -------------------------------------------------------------------------
    # Dataset paths (used by scripts/ingest.py only)
    # -------------------------------------------------------------------------
    MEDICAL_QA_CSV: str = Field(
        default="datasets/medical_QnA_Dataset.csv",
        description="Relative or absolute path to the medical Q&A CSV dataset",
    )
    MEDICAL_DEVICE_CSV: str = Field(
        default="datasets/medical_device_manuals_dataset.csv",
        description="Relative or absolute path to the medical device manuals CSV dataset",
    )
    SAMPLE_ROWS: int = Field(
        default=0,
        ge=0,
        description="Rows to sample per dataset for testing (0 = use all rows)",
    )

    # -------------------------------------------------------------------------
    # Memory (short-term checkpointer + long-term ChromaDB store)
    # -------------------------------------------------------------------------
    MEMORY_BACKEND: str = Field(
        default="sqlite",
        description="Checkpointer backend: 'sqlite' (dev/Docker) or 'postgres' (production EKS)",
    )
    MEMORY_SQLITE_PATH: str = Field(
        default="./memory.db",
        description="SQLite file path for SqliteSaver — must be on a persistent volume in Docker/K8s",
    )
    DATABASE_URL: str = Field(
        default="",
        description="Postgres connection string — required only when MEMORY_BACKEND=postgres",
    )
    MEMORY_COLLECTION: str = Field(
        default="agent_memories",
        description="ChromaDB collection name for long-term episodic memory",
    )
    MEMORY_TOP_K: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of past interactions to retrieve from memory per query",
    )
    MEMORY_PREFS_PATH: str = Field(
        default="./user_preferences.json",
        description="Path to JSON file storing user preferences — must be on a persistent volume",
    )

    # -------------------------------------------------------------------------
    # Observability
    # -------------------------------------------------------------------------
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Python logging level: DEBUG | INFO | WARNING | ERROR | CRITICAL",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance.

    Using lru_cache ensures the .env file is read only once per process,
    which avoids repeated I/O and makes injection easier in tests
    (call get_settings.cache_clear() to reset).
    """
    return Settings()
