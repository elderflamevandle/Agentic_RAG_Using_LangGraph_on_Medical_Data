from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RouteDecision(BaseModel):
    """Structured output schema for the router node.

    The LLM must return exactly one of the three route values.
    These strings double as the LangGraph node names in ``build.py``,
    so both files must stay in sync.
    """

    route: Literal["retrieve_qna", "retrieve_device", "web_search"] = Field(
        ...,
        description=(
            "Which source to query. "
            "retrieve_qna = general medical knowledge (symptoms, diseases, treatments); "
            "retrieve_device = medical device manuals (models, manufacturers, instructions); "
            "web_search = recent or external information not in local data."
        ),
    )


class RelevanceDecision(BaseModel):
    """Structured output schema for the relevance-checker node."""

    is_relevant: Literal["Yes", "No"] = Field(
        ...,
        description="Whether the provided context adequately addresses the user query.",
    )


class RewriteDecision(BaseModel):
    """Structured output schema for the query-rewriter node.

    ``needs_rewrite`` lets the LLM explicitly signal that the query is
    already standalone, avoiding silent echo-returns that are
    indistinguishable from a genuine no-op.
    """

    needs_rewrite: bool = Field(
        ...,
        description=(
            "True when the query contains pronouns or implicit references "
            "(its, they, it, the disease, the device, this condition) that "
            "require prior context to resolve. False when the query is "
            "already fully self-contained."
        ),
    )
    rewritten_query: str = Field(
        default="",
        description=(
            "A self-contained restatement of the query with all pronouns and "
            "implicit references replaced by their referents. "
            "Empty string when needs_rewrite is False."
        ),
    )
