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
    # Performance tracking
    node_timings: dict
    prompt_tokens: int
    completion_tokens: int
    # Memory
    thread_id: str        # session ID passed in from app.py; used by save_memory node
    memory_context: str   # retrieved past Q&As injected into the augment prompt
    # Conversation history — list of {"role": "user"|"assistant", "content": str}
    # populated by app.py from st.session_state.messages so follow-up questions
    # ("What are its treatments?") resolve correctly against prior turns.
    conversation_history: list
    # Coreference-resolved form of query produced by query_rewriter_node.
    # Absent when the query is already standalone or there is no history.
    # All retrieval nodes use: state.get("rewritten_query") or state["query"]
    # build_prompt_node and save_memory_node always use the original query.
    rewritten_query: str
