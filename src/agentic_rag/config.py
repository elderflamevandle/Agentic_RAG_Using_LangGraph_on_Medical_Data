from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized, environment-driven configuration.

    Set values via environment variables or a .env file.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- API keys ---
    GROQ_API_KEY: str = Field(..., description="Groq API key")
    SERPER_API_KEY: str = Field(..., description="Google Serper API key")

    # --- LLM params ---
    GROQ_MODEL: str = Field(default="llama-3.3-70b-versatile")
    TEMPERATURE: float = Field(default=0.3)

    # --- Vector store / Chroma ---
    CHROMA_PATH: str = Field(default="./chroma_db")
    QNA_COLLECTION: str = Field(default="medical_q_n_a")
    DEVICE_COLLECTION: str = Field(default="medical_device_manual")

    # --- Retrieval settings ---
    TOP_K: int = Field(default=3, ge=1, le=20)

    # --- Prompting / limits ---
    ANSWER_WORD_LIMIT: int = Field(default=80, ge=10, le=300)
    MAX_ITERATIONS: int = Field(default=3, ge=1, le=10)

    # --- Datasets (optional; used by ingestion script) ---
    MEDICAL_QA_CSV: str = Field(default="datasets/medical_QnA_Dataset.csv")
    MEDICAL_DEVICE_CSV: str = Field(default="datasets/medical_device_manuals_dataset.csv")
    SAMPLE_ROWS: int = Field(default=0, ge=0, description="0 means no sampling")

    # --- Observability ---
    LOG_LEVEL: str = Field(default="INFO")


def get_settings() -> Settings:
    # In production you may want caching; keep it simple here.
    return Settings()
