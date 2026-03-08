"""LangGraph workflow assembly for the agentic RAG pipeline.

Call ``build_agent(settings)`` to get a compiled, ready-to-invoke graph.

Workflow topology
-----------------
::

    START
      │
      ▼
    router  ──────────────────────────────────────────────┐
      │ (conditional on state["route"])                   │
      ├─► retrieve_qna                                    │
      ├─► retrieve_device                                 │
      └─► web_search ◄──── (relevance retry loop) ────────┘
              │
              ▼
        relevance_checker
              │ (conditional on is_relevant + iteration_count)
              ├─► [Yes]  augment ──► generate ──► END
              └─► [No]   web_search  (up to MAX_ITERATIONS)
"""
from __future__ import annotations

import logging
import time

from langgraph.graph import END, START, StateGraph

from ..config import Settings
from ..llm import build_llm
from ..tools.web_search import build_serper
from ..vectorstore.chroma_store import ChromaStore
from .nodes import (
    build_prompt_node,
    generate_node,
    relevance_checker_node,
    retrieve_device_node,
    retrieve_qna_node,
    router_node,
    web_search_node,
)
from .state import GraphState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Node timing wrapper
# ---------------------------------------------------------------------------

def _timed(name: str, fn):
    """Wrap a node function to record per-node execution time in state.

    Reads the existing ``node_timings`` dict from state (if any), appends
    this node's elapsed milliseconds, and writes it back.  Safe to use on
    any node — the merge is non-destructive.

    Parameters
    ----------
    name:
        Node name used as the dict key (matches the LangGraph node constant).
    fn:
        The original node callable returned by a node factory.
    """
    def _wrapper(state: GraphState) -> GraphState:
        t0 = time.perf_counter()
        result = fn(state)
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        timings = dict((result or state).get("node_timings") or {})
        timings[name] = elapsed_ms
        return {**result, "node_timings": timings}

    _wrapper.__name__ = f"timed_{name}"
    return _wrapper


# Node name constants — single source of truth referenced by add_node and edges.
_ROUTER = "router"
_RETRIEVE_QNA = "retrieve_qna"
_RETRIEVE_DEVICE = "retrieve_device"
_WEB_SEARCH = "web_search"
_RELEVANCE_CHECKER = "relevance_checker"
_AUGMENT = "augment"
_GENERATE = "generate"


def build_agent(settings: Settings):
    """Build and compile the agentic RAG LangGraph workflow.

    Steps
    -----
    1. Instantiate shared dependencies (LLM, ChromaStore, Serper search).
    2. Register all seven nodes on a ``StateGraph``.
    3. Wire conditional and unconditional edges.
    4. Compile and return the runnable graph.

    Parameters
    ----------
    settings:
        Fully-populated ``Settings`` instance (from ``get_settings()``).

    Returns
    -------
    CompiledGraph
        A LangGraph compiled graph ready for ``.invoke({"query": "..."})`` calls.
    """
    logger.info("Building agentic RAG agent | model=%s", settings.GROQ_MODEL)

    # --- Shared dependencies ---
    llm = build_llm(settings)
    store = ChromaStore(settings.CHROMA_PATH)
    collections = store.ensure_collections(settings.QNA_COLLECTION, settings.DEVICE_COLLECTION)
    search_tool = build_serper(serper_api_key=settings.SERPER_API_KEY)

    # --- Graph construction ---
    workflow = StateGraph(GraphState)

    workflow.add_node(_ROUTER,           _timed(_ROUTER,           router_node(llm)))
    workflow.add_node(_RETRIEVE_QNA,     _timed(_RETRIEVE_QNA,     retrieve_qna_node(collections, settings)))
    workflow.add_node(_RETRIEVE_DEVICE,  _timed(_RETRIEVE_DEVICE,  retrieve_device_node(collections, settings)))
    workflow.add_node(_WEB_SEARCH,       _timed(_WEB_SEARCH,       web_search_node(search_tool)))
    workflow.add_node(_RELEVANCE_CHECKER,_timed(_RELEVANCE_CHECKER,relevance_checker_node(llm, settings)))
    workflow.add_node(_AUGMENT,          _timed(_AUGMENT,          build_prompt_node(settings)))
    workflow.add_node(_GENERATE,         _timed(_GENERATE,         generate_node(llm)))

    # --- Entry edge ---
    workflow.add_edge(START, _ROUTER)

    # --- Router → retriever (conditional on state["route"]) ---
    def _pick_retriever(state: GraphState) -> str:
        """Return the node name that matches the router's decision."""
        return state["route"]  # values are the same as node name constants above

    workflow.add_conditional_edges(
        _ROUTER,
        _pick_retriever,
        {
            _RETRIEVE_QNA: _RETRIEVE_QNA,
            _RETRIEVE_DEVICE: _RETRIEVE_DEVICE,
            _WEB_SEARCH: _WEB_SEARCH,
        },
    )

    # --- All retrievers feed into the relevance checker ---
    for retriever in (_RETRIEVE_QNA, _RETRIEVE_DEVICE, _WEB_SEARCH):
        workflow.add_edge(retriever, _RELEVANCE_CHECKER)

    # --- Relevance checker → augment or retry web_search ---
    def _relevance_routing(state: GraphState) -> str:
        """Read the relevance verdict written by the relevance_checker node."""
        return state.get("is_relevant", "Yes")

    workflow.add_conditional_edges(
        _RELEVANCE_CHECKER,
        _relevance_routing,
        {
            "Yes": _AUGMENT,
            "No": _WEB_SEARCH,
        },
    )

    # --- Linear tail of the pipeline ---
    workflow.add_edge(_AUGMENT, _GENERATE)
    workflow.add_edge(_GENERATE, END)

    compiled = workflow.compile()
    logger.info("Agent graph compiled successfully.")
    return compiled
