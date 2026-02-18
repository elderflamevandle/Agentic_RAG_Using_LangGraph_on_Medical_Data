from __future__ import annotations

import logging
from langgraph.graph import StateGraph, START, END

from ..config import Settings
from ..llm import build_llm
from ..tools.web_search import build_serper
from ..vectorstore.chroma_store import ChromaStore
from .state import GraphState
from .nodes import (
    router_node,
    retrieve_qna_node,
    retrieve_device_node,
    web_search_node,
    relevance_checker_node,
    build_prompt_node,
    generate_node,
)

logger = logging.getLogger(__name__)


def build_agent(settings: Settings):
    """Build and compile the LangGraph agentic RAG workflow."""
    llm = build_llm(settings)
    store = ChromaStore(settings.CHROMA_PATH)
    collections = store.ensure_collections(settings.QNA_COLLECTION, settings.DEVICE_COLLECTION)
    search_tool = build_serper()

    workflow = StateGraph(GraphState)

    workflow.add_node("Router", router_node(llm))
    workflow.add_node("Retrieve_QnA", retrieve_qna_node(collections, settings))
    workflow.add_node("Retrieve_Device", retrieve_device_node(collections, settings))
    workflow.add_node("Web_Search", web_search_node(search_tool))
    workflow.add_node("Relevance_Checker", relevance_checker_node(llm))
    workflow.add_node("Augment", build_prompt_node(settings))
    workflow.add_node("Generate", generate_node(llm))

    # Edges
    workflow.add_edge(START, "Router")

    def _route(state: GraphState) -> str:
        return state["route"]

    workflow.add_conditional_edges(
        "Router",
        _route,
        {
            "Retrieve_QnA": "Retrieve_QnA",
            "Retrieve_Device": "Retrieve_Device",
            "Web_Search": "Web_Search",
        },
    )

    for n in ("Retrieve_QnA", "Retrieve_Device", "Web_Search"):
        workflow.add_edge(n, "Relevance_Checker")

    def _relevance_decision(state: GraphState) -> str:
        iteration = state.get("iteration_count", 0) + 1
        state["iteration_count"] = iteration
        if iteration >= settings.MAX_ITERATIONS:
            logger.warning("Max iterations reached (%d). Forcing relevance=Yes.", iteration)
            state["is_relevant"] = "Yes"
        return state.get("is_relevant", "Yes")

    workflow.add_conditional_edges(
        "Relevance_Checker",
        _relevance_decision,
        {
            "Yes": "Augment",
            "No": "Web_Search",
        },
    )

    workflow.add_edge("Augment", "Generate")
    workflow.add_edge("Generate", END)

    return workflow.compile()