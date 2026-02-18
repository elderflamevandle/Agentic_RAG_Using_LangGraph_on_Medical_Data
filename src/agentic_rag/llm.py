from __future__ import annotations

from langchain_groq import ChatGroq

from .config import Settings


def build_llm(settings: Settings) -> ChatGroq:
    """Create a Groq chat model instance."""
    return ChatGroq(model=settings.GROQ_MODEL, temperature=settings.TEMPERATURE)


def invoke_text(llm: ChatGroq, prompt: str) -> str:
    """Simple helper to invoke the model with a user prompt."""
    resp = llm.invoke(input=[{"role": "user", "content": prompt}])
    return resp.content
