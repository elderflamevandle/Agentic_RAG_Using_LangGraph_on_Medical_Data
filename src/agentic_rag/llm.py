"""LLM factory and invocation helpers.

All LLM access in this project goes through this module so that the
model choice and call conventions are centralized in one place.
"""
from __future__ import annotations

import logging

from langchain_groq import ChatGroq

from .config import Settings
from .exceptions import LLMError

logger = logging.getLogger(__name__)


def build_llm(settings: Settings) -> ChatGroq:
    """Create and return a ``ChatGroq`` instance.

    The API key is passed explicitly from ``settings`` rather than relying
    on ``os.environ``, because pydantic-settings reads ``.env`` into the
    Settings model only — it does NOT populate ``os.environ``.

    Parameters
    ----------
    settings:
        Runtime configuration; provides ``GROQ_API_KEY``, ``GROQ_MODEL``,
        and ``TEMPERATURE``.

    Returns
    -------
    ChatGroq
    """
    logger.debug(
        "Building LLM | model=%s | temperature=%.2f",
        settings.GROQ_MODEL,
        settings.TEMPERATURE,
    )
    try:
        return ChatGroq(
            model=settings.GROQ_MODEL,
            temperature=settings.TEMPERATURE,
            groq_api_key=settings.GROQ_API_KEY,
        )
    except Exception as exc:
        raise LLMError(f"Failed to initialise ChatGroq | model={settings.GROQ_MODEL}") from exc


def invoke_text(llm: ChatGroq, prompt: str) -> str:
    """Invoke the model with a single user message and return the text content.

    Parameters
    ----------
    llm:
        A ``ChatGroq`` instance.
    prompt:
        The user-facing prompt string.

    Returns
    -------
    str
        The model's response text.
    """
    response = llm.invoke(input=[{"role": "user", "content": prompt}])
    return response.content
