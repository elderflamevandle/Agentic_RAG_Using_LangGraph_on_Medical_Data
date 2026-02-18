from __future__ import annotations

import logging

from langchain_groq import ChatGroq

from ..config import Settings
from ..llm import invoke_text
from ..tools.web_search import run_search
from ..vectorstore.chroma_store import ChromaStore, ChromaCollections
from .state import GraphState, Route
from .schemas import RouteDecision, RelevanceDecision

logger = logging.getLogger(__name__)


def router_node(llm: ChatGroq) :
    """Factory: returns a node function bound to an LLM."""
    structured = llm.with_structured_output(RouteDecision)

    def _router(state: GraphState) -> GraphState:
        query = state["query"]
        prompt = (
            "You are a routing agent. Decide where to look for information.\n"
            "Options:\n"
            "- Retrieve_QnA: general medical knowledge, symptoms, disease, treatment.\n"
            "- Retrieve_Device: device manuals, device instructions, model/manufacturer.\n"
            "- Web_Search: recent info, brand names, external facts, anything not likely in local data.\n\n"
            f"Query: {query}"
        )
        try:
            decision = structured.invoke(prompt)
            route: Route = decision.route
        except Exception:
            # Safe fallback to Web_Search if structured parsing fails.
            route = "Web_Search"
        state["route"] = route
        state["source"] = route
        logger.info("Router decided: %s", route)
        return state

    return _router


def retrieve_qna_node(collections: ChromaCollections, settings: Settings):
    def _retrieve(state: GraphState) -> GraphState:
        query = state["query"]
        ctx, _ = ChromaStore.query(collections.qna, query_text=query, top_k=settings.TOP_K)
        state["context"] = ctx
        state["source"] = "Medical Q&A Collection"
        return state
    return _retrieve


def retrieve_device_node(collections: ChromaCollections, settings: Settings):
    def _retrieve(state: GraphState) -> GraphState:
        query = state["query"]
        ctx, _ = ChromaStore.query(collections.device, query_text=query, top_k=settings.TOP_K)
        state["context"] = ctx
        state["source"] = "Medical Device Manual"
        return state
    return _retrieve


def web_search_node(search_tool):
    def _search(state: GraphState) -> GraphState:
        query = state["query"]
        state["context"] = run_search(search_tool, query=query)
        state["source"] = "Web Search (Serper)"
        return state
    return _search


def relevance_checker_node(llm: ChatGroq):
    structured = llm.with_structured_output(RelevanceDecision)

    def _check(state: GraphState) -> GraphState:
        query = state["query"]
        context = state.get("context", "")
        prompt = (
            "Decide if the context is relevant to the user query.\n"
            "Answer strictly in the required structured schema.\n\n"
            f"User Query: {query}\n\n"
            f"Context:\n{context}"
        )
        try:
            decision = structured.invoke(prompt)
            state["is_relevant"] = decision.is_relevant
        except Exception:
            # Conservative: if parsing fails, treat as relevant to avoid loops.
            state["is_relevant"] = "Yes"
        return state

    return _check


def build_prompt_node(settings: Settings):
    def _build(state: GraphState) -> GraphState:
        query = state["query"]
        context = state.get("context", "")
        prompt = (
            "You are a helpful assistant. Use the given context to answer the question.\n"
            "If the context is insufficient, say what is missing briefly.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Constraints: Keep the answer within ~{settings.ANSWER_WORD_LIMIT} words."
        )
        state["prompt"] = prompt
        return state
    return _build


def generate_node(llm: ChatGroq):
    def _gen(state: GraphState) -> GraphState:
        answer = invoke_text(llm, state["prompt"])
        state["response"] = answer
        return state
    return _gen
