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

from ..tools.web_search import run_search
from ..vectorstore.chroma_store import ChromaCollections, ChromaStore
from .prompts import (
    AUGMENT_PROMPT,
    QUERY_REWRITER_PROMPT,
    RELEVANCE_CHECKER_PROMPT,
    ROUTER_PROMPT,
)
from .schemas import RelevanceDecision, RewriteDecision, RouteDecision
from .state import GraphState, Route

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Token extraction helper
# ---------------------------------------------------------------------------

def _extract_tokens(raw_msg) -> tuple[int, int]:
    """Extract (prompt_tokens, completion_tokens) from an AIMessage.

    Works with both raw ``llm.invoke()`` results and the ``"raw"`` key
    returned by ``with_structured_output(include_raw=True)``.
    Returns ``(0, 0)`` when usage metadata is unavailable.
    """
    usage = getattr(raw_msg, "usage_metadata", None) or {}
    return int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))


# ---------------------------------------------------------------------------
# Query rewriter node
# ---------------------------------------------------------------------------

def query_rewriter_node(llm: ChatGroq):
    """Factory: return a node that rewrites ambiguous follow-up queries.

    Resolves coreferences ("its", "they", "the disease") against
    ``state["conversation_history"]`` so that downstream retrieval nodes
    receive a fully self-contained query string.

    Is a zero-cost no-op when there is no conversation history (i.e. the
    first message in a session), skipping the LLM call entirely.

    Parameters
    ----------
    llm:
        ChatGroq instance — same model used by the router and relevance checker.
    """
    structured = llm.with_structured_output(RewriteDecision, include_raw=True)

    def _rewrite(state: GraphState) -> GraphState:
        history = state.get("conversation_history") or []
        if not history:
            logger.debug("Query rewriter: no history, skipping.")
            return state

        query = state["query"]

        history_lines = []
        for turn in history[-4:]:
            role = "User" if turn.get("role") == "user" else "Assistant"
            history_lines.append(f"  {role}: {turn.get('content', '')[:300]}")
        history_text = "\n".join(history_lines)

        prompt = QUERY_REWRITER_PROMPT.format(
            history_text=history_text, query=query,
        )

        try:
            result = structured.invoke(prompt)
            decision: RewriteDecision = result["parsed"]
            p_tok, c_tok = _extract_tokens(result["raw"])
            logger.info(
                "Query rewriter | needs_rewrite=%s | prompt_tokens=%d | completion_tokens=%d",
                decision.needs_rewrite, p_tok, c_tok,
            )
            if decision.needs_rewrite and decision.rewritten_query.strip():
                logger.info(
                    "Query rewritten | original=%r | rewritten=%r",
                    query, decision.rewritten_query,
                )
                return {**state, "rewritten_query": decision.rewritten_query.strip()}
            logger.debug("Query rewriter: standalone query, no rewrite | query=%r", query)
            return state
        except Exception as exc:
            logger.warning(
                "Query rewriter structured-output failed (%s). Using original query.", exc
            )
            return state

    return _rewrite


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
    structured = llm.with_structured_output(RouteDecision, include_raw=True)

    def _router(state: GraphState) -> GraphState:
        query = state.get("rewritten_query") or state["query"]
        prompt = ROUTER_PROMPT.format(query=query)
        try:
            result = structured.invoke(prompt)
            decision = result["parsed"]
            route: Route = decision.route
            p_tok, c_tok = _extract_tokens(result["raw"])
            logger.info(
                "Router decision: %s | query=%r | prompt_tokens=%d | completion_tokens=%d",
                route, query, p_tok, c_tok,
            )
        except Exception as exc:
            logger.warning(
                "Router structured-output failed (%s). Falling back to web_search.", exc
            )
            route = "web_search"

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
        query = state.get("rewritten_query") or state["query"]
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
        query = state.get("rewritten_query") or state["query"]
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
        query = state.get("rewritten_query") or state["query"]
        logger.debug("Web search | query=%r", query)
        context = run_search(search_tool, query=query)
        logger.info("Web search complete | result_chars=%d", len(context))
        return {**state, "context": context, "source": "Web Search (Serper)"}

    return _search


# ---------------------------------------------------------------------------
# Relevance checker node
# ---------------------------------------------------------------------------

def relevance_checker_node(llm: ChatGroq, settings: Settings):
    """Factory: return a node that judges whether context answers the query.

    Uses structured output (``RelevanceDecision``) for a strict Yes/No
    verdict.  Defaults to ``"Yes"`` on parse failure so the pipeline
    completes rather than looping indefinitely.

    Parameters
    ----------
    llm:
        ChatGroq instance used for relevance classification.
    settings:
        Runtime configuration — provides ``MAX_ITERATIONS``.
    """
    structured = llm.with_structured_output(RelevanceDecision, include_raw=True)

    def _check(state: GraphState) -> GraphState:
        query = state.get("rewritten_query") or state["query"]
        context = state.get("context", "")
        iteration = state.get("iteration_count", 0) + 1

        prompt = RELEVANCE_CHECKER_PROMPT.format(query=query, context=context)
        try:
            result = structured.invoke(prompt)
            decision = result["parsed"]
            is_relevant = decision.is_relevant
            p_tok, c_tok = _extract_tokens(result["raw"])
            logger.info(
                "Relevance check: %s | iteration=%d | source=%s | prompt_tokens=%d | completion_tokens=%d",
                is_relevant, iteration, state.get("source", "unknown"), p_tok, c_tok,
            )
        except Exception as exc:
            logger.warning(
                "Relevance checker structured-output failed (%s). Defaulting to No (retry).", exc
            )
            is_relevant = "No"

        if iteration >= settings.MAX_ITERATIONS:
            logger.warning(
                "MAX_ITERATIONS (%d) reached. Forcing answer generation.", settings.MAX_ITERATIONS
            )
            is_relevant = "Yes"

        return {**state, "is_relevant": is_relevant, "iteration_count": iteration}

    return _check


# ---------------------------------------------------------------------------
# Prompt builder node
# ---------------------------------------------------------------------------

def build_prompt_node(settings: Settings, memory_store=None):
    """Factory: return a node that assembles the RAG prompt for the generator.

    Uses conversation history from the current session for follow-up context.
    Long-term memory (``memory_store``) is supported but disabled by default
    — pass ``None`` to skip it (session-only mode).

    Parameters
    ----------
    settings:
        Runtime configuration — provides ``ANSWER_WORD_LIMIT``, ``MEMORY_TOP_K``.
    memory_store:
        Optional ``AgentMemoryStore`` instance.  When ``None`` the node
        uses only current-session conversation history (no cross-session memory).
    """
    def _build(state: GraphState) -> GraphState:
        query   = state["query"]
        context = state.get("context", "")
        source  = state.get("source", "unknown")

        # --- Current-session conversation history ---
        history_section = ""
        history = state.get("conversation_history") or []
        if history:
            lines = ["Conversation History:"]
            for turn in history:
                role = "User" if turn.get("role") == "user" else "Assistant"
                lines.append(f"  {role}: {turn.get('content', '')[:300]}")
            history_section = "\n".join(lines) + "\n\n"

        # --- Long-term memory (disabled when memory_store is None) ---
        memory_section = ""
        if memory_store is not None:
            try:
                prefs    = memory_store.get_preferences()
                past_qas = memory_store.retrieve_similar(query, top_k=settings.MEMORY_TOP_K)

                parts: list[str] = []
                topics = prefs.get("topics_of_interest", [])
                if topics:
                    parts.append(f"User interests: {', '.join(topics[:5])}")

                if past_qas:
                    parts.append("Relevant past interactions:")
                    for pa in past_qas:
                        parts.append(f"  Past Q: {pa['query']}")
                        parts.append(f"  Past A: {pa['response'][:200]}")

                if parts:
                    memory_section = "\n".join(parts) + "\n\n"
            except Exception as exc:
                logger.warning("Memory injection failed (non-fatal): %s", exc)

        prompt = AUGMENT_PROMPT.format(
            history_section=history_section + (f"Memory Context:\n{memory_section}" if memory_section else ""),
            source=source,
            context=context,
            query=query,
            word_limit=settings.ANSWER_WORD_LIMIT,
        )
        logger.debug(
            "Prompt built | chars=%d | history_turns=%d | memory=%s",
            len(prompt), len(history), bool(memory_section),
        )
        return {**state, "prompt": prompt, "memory_context": memory_section}

    return _build


# ---------------------------------------------------------------------------
# Memory save node
# ---------------------------------------------------------------------------

def save_memory_node(memory_store=None):
    """Factory: return a node that persists the completed Q&A to long-term memory.

    Runs as the final graph node (after generate).  Failures are caught and
    logged so a memory write error never crashes the pipeline.

    Parameters
    ----------
    memory_store:
        ``AgentMemoryStore`` instance.  When ``None`` the node is a no-op
        (memory disabled), keeping the graph wiring intact.
    """
    def _save(state: GraphState) -> GraphState:
        if memory_store is None:
            return state
        try:
            memory_store.save_interaction(
                query=state["query"],
                response=state.get("response", ""),
                source=state.get("source", ""),
                route=state.get("route", ""),
                thread_id=state.get("thread_id", "default"),
            )
        except Exception as exc:
            logger.warning("Memory save failed (non-fatal): %s", exc)
        return state

    return _save


# ---------------------------------------------------------------------------
# Generator node
# ---------------------------------------------------------------------------

def generate_node(llm: ChatGroq):
    """Factory: return a node that calls the LLM to produce the final answer.

    Calls ``llm.invoke()`` directly (rather than the ``invoke_text`` helper)
    so that Groq's ``usage_metadata`` is accessible on the returned
    ``AIMessage``.  Token counts are written into state for downstream
    performance evaluation.

    Parameters
    ----------
    llm:
        ChatGroq instance used for answer generation.
    """
    def _generate(state: GraphState) -> GraphState:
        logger.debug("Generating answer...")
        msg = llm.invoke(state["prompt"])
        response = (msg.content or "").strip()

        # Extract token counts from Groq usage_metadata
        # (AIMessage.usage_metadata keys: input_tokens, output_tokens, total_tokens)
        usage = getattr(msg, "usage_metadata", None) or {}
        prompt_tokens     = int(usage.get("input_tokens",  0))
        completion_tokens = int(usage.get("output_tokens", 0))

        logger.info(
            "Answer generated | chars=%d | prompt_tokens=%d | completion_tokens=%d",
            len(response), prompt_tokens, completion_tokens,
        )
        return {
            **state,
            "response":          response,
            "prompt_tokens":     prompt_tokens,
            "completion_tokens": completion_tokens,
        }

    return _generate
