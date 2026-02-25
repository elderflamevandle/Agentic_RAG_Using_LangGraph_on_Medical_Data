from __future__ import annotations

from typing import Literal, TypedDict


# Route values are what the LLM returns — kept lowercase to match
# the LangGraph node names defined in build.py (single source of truth).
Route = Literal["retrieve_qna", "retrieve_device", "web_search"]


class _RequiredState(TypedDict):
    """Fields that must be present before the graph starts."""

    query: str


class GraphState(_RequiredState, total=False):
    """Full state dict passed between every LangGraph node.

    Inherits the required ``query`` field from ``_RequiredState``;
    everything else is optional (``total=False``) because each node
    populates only the fields it owns.

    Fields
    ------
    query : str
        The user's original question — required at graph entry.
    route : Route
        Decision from the router node (which retriever to call).
    context : str
        Raw text retrieved from the chosen source.
    source : str
        Human-readable label for the retrieval source (for logs / UI).
    is_relevant : "Yes" | "No"
        Relevance verdict produced by the relevance-checker node.
    iteration_count : int
        Number of relevance-check loop iterations completed so far.
        Capped by ``Settings.MAX_ITERATIONS`` to prevent infinite loops.
    prompt : str
        Fully-assembled RAG prompt forwarded to the generator node.
    response : str
        Final answer text produced by the generator node.
    """

    route: Route
    context: str
    source: str
    is_relevant: Literal["Yes", "No"]
    iteration_count: int
    prompt: str
    response: str
