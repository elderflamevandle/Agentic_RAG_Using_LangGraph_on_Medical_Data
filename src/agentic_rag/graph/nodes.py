"""Node factory functions for the LangGraph agentic RAG workflow.

Each public function is a *factory* — it closes over its dependencies
(LLM, collections, settings, etc.) and returns a callable node that
accepts and returns a ``GraphState`` dict.

This factory pattern keeps nodes free of global state and makes them
easy to unit-test in isolation (inject mock dependencies).

Execution order (happy path)
----------------------------
router
  → retrieve_qna | retrieve_device | web_search
      → relevance_checker
          → [Yes]  augment → generate → END
          → [No]   web_search (retries up to MAX_ITERATIONS)
"""
from __future__ import annotations

import logging

from langchain_groq import ChatGroq

from ..config import Settings
from ..llm import invoke_text
from ..tools.web_search import run_search
from ..vectorstore.chroma_store import ChromaCollections, ChromaStore
from .schemas import RelevanceDecision, RouteDecision
from .state import GraphState, Route

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Router node
# ---------------------------------------------------------------------------

def router_node(llm: ChatGroq):
    """Factory: return a routing node bound to an LLM.

    The node prompts the LLM for a ``RouteDecision`` and writes the
    chosen route into ``state["route"]``.  Falls back to ``web_search``
    when structured-output parsing fails.

    Parameters
    ----------
    llm:
        ChatGroq instance used for routing decisions.
    """
    structured = llm.with_structured_output(RouteDecision)

    def _router(state: GraphState) -> GraphState:
        query = state["query"]
        prompt = (
            "You are a routing agent. Choose exactly one source to answer the query.\n\n"
            "Options:\n"
            "  retrieve_qna    — general medical knowledge: symptoms, diseases, treatments.\n"
            "  retrieve_device — medical device manuals: model numbers, manufacturers, instructions.\n"
            "  web_search      — recent news, brand names, or anything unlikely to be in local data.\n\n"
            f"Query: {query}"
        )
        try:
            decision = structured.invoke(prompt)
            route: Route = decision.route
        except Exception as exc:
            logger.warning(
                "Router structured-output failed (%s). Falling back to web_search.", exc
            )
            route = "web_search"

        logger.info("Router decision: %s | query=%r", route, query)
        return {**state, "route": route, "source": route}

    return _router


# ---------------------------------------------------------------------------
# Retriever nodes
# ---------------------------------------------------------------------------

def retrieve_qna_node(collections: ChromaCollections, settings: Settings):
    """Factory: return a node that retrieves from the medical Q&A collection.

    Parameters
    ----------
    collections:
        Named ChromaDB collections (qna + device).
    settings:
        Runtime configuration — provides ``TOP_K``.
    """
    def _retrieve(state: GraphState) -> GraphState:
        query = state["query"]
        logger.debug("Q&A retrieval | top_k=%d | query=%r", settings.TOP_K, query)
        ctx, docs = ChromaStore.query(collections.qna, query_text=query, top_k=settings.TOP_K)
        logger.info("Q&A retrieval complete | docs_returned=%d", len(docs))
        return {**state, "context": ctx, "source": "Medical Q&A Collection"}

    return _retrieve


def retrieve_device_node(collections: ChromaCollections, settings: Settings):
    """Factory: return a node that retrieves from the medical device collection.

    Parameters
    ----------
    collections:
        Named ChromaDB collections (qna + device).
    settings:
        Runtime configuration — provides ``TOP_K``.
    """
    def _retrieve(state: GraphState) -> GraphState:
        query = state["query"]
        logger.debug("Device retrieval | top_k=%d | query=%r", settings.TOP_K, query)
        ctx, docs = ChromaStore.query(collections.device, query_text=query, top_k=settings.TOP_K)
        logger.info("Device retrieval complete | docs_returned=%d", len(docs))
        return {**state, "context": ctx, "source": "Medical Device Manual"}

    return _retrieve


def web_search_node(search_tool):
    """Factory: return a node that performs a live Serper web search.

    Parameters
    ----------
    search_tool:
        A ``GoogleSerperAPIWrapper`` instance.
    """
    def _search(state: GraphState) -> GraphState:
        query = state["query"]
        logger.debug("Web search | query=%r", query)
        context = run_search(search_tool, query=query)
        logger.info("Web search complete | result_chars=%d", len(context))
        return {**state, "context": context, "source": "Web Search (Serper)"}

    return _search


# ---------------------------------------------------------------------------
# Relevance checker node
# ---------------------------------------------------------------------------

def relevance_checker_node(llm: ChatGroq):
    """Factory: return a node that judges whether context answers the query.

    Uses structured output (``RelevanceDecision``) for a strict Yes/No
    verdict.  Defaults to ``"Yes"`` on parse failure so the pipeline
    completes rather than looping indefinitely.

    Parameters
    ----------
    llm:
        ChatGroq instance used for relevance classification.
    """
    structured = llm.with_structured_output(RelevanceDecision)

    def _check(state: GraphState) -> GraphState:
        query = state["query"]
        context = state.get("context", "")
        prompt = (
            "Decide strictly whether the context below is relevant to the user query.\n"
            "Respond only in the required structured schema.\n\n"
            f"User Query: {query}\n\n"
            f"Context:\n{context}"
        )
        try:
            decision = structured.invoke(prompt)
            is_relevant = decision.is_relevant
        except Exception as exc:
            logger.warning(
                "Relevance checker structured-output failed (%s). Defaulting to Yes.", exc
            )
            is_relevant = "Yes"

        logger.info(
            "Relevance check: %s | source=%s", is_relevant, state.get("source", "unknown")
        )
        return {**state, "is_relevant": is_relevant}

    return _check


# ---------------------------------------------------------------------------
# Prompt builder node
# ---------------------------------------------------------------------------

def build_prompt_node(settings: Settings):
    """Factory: return a node that assembles the RAG prompt for the generator.

    Combines the source label, retrieved context, original query, and the
    configured word-count constraint into a single instruction string.

    Parameters
    ----------
    settings:
        Runtime configuration — provides ``ANSWER_WORD_LIMIT``.
    """
    def _build(state: GraphState) -> GraphState:
        query = state["query"]
        context = state.get("context", "")
        source = state.get("source", "unknown")
        prompt = (
            "You are a knowledgeable medical assistant. "
            "Answer the question using *only* the context provided below.\n"
            "If the context does not contain enough information, "
            "briefly state what is missing rather than guessing.\n\n"
            f"Source: {source}\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Constraint: Keep the answer within approximately {settings.ANSWER_WORD_LIMIT} words."
        )
        logger.debug("Prompt built | length=%d chars", len(prompt))
        return {**state, "prompt": prompt}

    return _build


# ---------------------------------------------------------------------------
# Generator node
# ---------------------------------------------------------------------------

def generate_node(llm: ChatGroq):
    """Factory: return a node that calls the LLM to produce the final answer.

    Parameters
    ----------
    llm:
        ChatGroq instance used for answer generation.
    """
    def _generate(state: GraphState) -> GraphState:
        logger.debug("Generating answer...")
        response = invoke_text(llm, state["prompt"])
        logger.info("Answer generated | length=%d chars", len(response))
        return {**state, "response": response}

    return _generate
