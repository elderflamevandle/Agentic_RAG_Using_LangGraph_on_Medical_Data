"""Streamlit front-end for the Agentic RAG Medical Q&A pipeline.

Launch with::

    streamlit run app.py

Prerequisites
-------------
1. Copy .env.example -> .env and fill in GROQ_API_KEY and SERPER_API_KEY.
2. Run  ``python scripts/ingest.py``  to populate the ChromaDB vector store.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Make src/ importable when running from the repo root.
# ---------------------------------------------------------------------------
_ROOT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _ROOT_DIR / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import dataclasses
import time

import streamlit as st

from agentic_rag.config import get_settings
from agentic_rag.graph import build_agent
from agentic_rag.llm import build_llm
from agentic_rag.logging_config import setup_logging
from agentic_rag.metrics import EvaluationResult, evaluate

# ---------------------------------------------------------------------------
# Page config — must be the first Streamlit call in the script.
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Medical Q&A Agent",
    page_icon=":hospital:",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Logging — idempotent; setup_logging guards against duplicate handlers.
# ---------------------------------------------------------------------------
setup_logging(get_settings().LOG_LEVEL)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_GROQ_MODELS = [
    "llama-3.3-70b-versatile",   # default — best quality
    "llama-3.1-8b-instant",      # fast, lightweight
    "gemma2-9b-it",              # Google Gemma via Groq
    "mixtral-8x7b-32768",        # large context window
]

_SOURCE_COLOURS: dict[str, str] = {
    "Medical Q&A Collection": "#1f77b4",   # blue
    "Medical Device Manual":  "#2ca02c",   # green
    "Web Search (Serper)":    "#d62728",   # red
}
_FALLBACK_COLOUR = "#7f7f7f"


def _source_badge(source: str) -> str:
    """Return an HTML badge string for the given source label."""
    colour = _SOURCE_COLOURS.get(source, _FALLBACK_COLOUR)
    return (
        f'<span style="background-color:{colour};color:#ffffff;'
        f'padding:2px 10px;border-radius:4px;'
        f'font-size:0.72rem;font-weight:600;">'
        f"{source}</span>"
    )


# ---------------------------------------------------------------------------
# Agent cache — keyed on every tunable parameter.
# A new combination triggers a rebuild; the same combination reuses the cache.
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Applying settings and loading agent…")
def _load_agent(
    model: str,
    temperature: float,
    top_k: int,
    max_iterations: int,
    answer_word_limit: int,
):
    """Build and cache the LangGraph agent for a specific parameter combination.

    ``st.cache_resource`` keys the cache on all arguments, so changing any
    slider or selectbox produces a fresh agent while keeping previously-used
    configurations cached in memory.

    Parameters
    ----------
    model, temperature, top_k, max_iterations, answer_word_limit:
        User-selected values from the sidebar widgets.

    Returns
    -------
    tuple[CompiledGraph, Settings, ChatGroq]
        The compiled agent graph, the active settings, and an LLM instance
        for the LLM-as-judge evaluation step.
    """
    base = get_settings()                       # reads API keys + infra from .env
    settings = base.model_copy(update={         # override only the tunable fields
        "GROQ_MODEL":         model,
        "TEMPERATURE":        temperature,
        "TOP_K":              top_k,
        "MAX_ITERATIONS":     max_iterations,
        "ANSWER_WORD_LIMIT":  answer_word_limit,
    })
    agent   = build_agent(settings)
    eval_llm = build_llm(settings)              # reused for LLM-as-judge evaluation
    logger.info(
        "Agent built | model=%s temp=%.1f top_k=%d max_iter=%d word_limit=%d",
        model, temperature, top_k, max_iterations, answer_word_limit,
    )
    return agent, settings, eval_llm


# ---------------------------------------------------------------------------
# Sidebar — renders widgets and returns the chosen parameter values.
# ---------------------------------------------------------------------------
def _render_sidebar() -> tuple[str, float, int, int, int]:
    """Render all interactive controls and return the current parameter values.

    Returns
    -------
    tuple of (model, temperature, top_k, max_iterations, answer_word_limit)
    """
    defaults = get_settings()   # used only to seed widget default values

    with st.sidebar:
        st.header("Settings")
        st.caption("Adjust parameters below. The agent rebuilds automatically when anything changes.")
        st.markdown("---")

        # --- Model selection ---
        default_model_idx = (
            _GROQ_MODELS.index(defaults.GROQ_MODEL)
            if defaults.GROQ_MODEL in _GROQ_MODELS
            else 0
        )
        model = st.selectbox(
            "Model",
            options=_GROQ_MODELS,
            index=default_model_idx,
            help="Groq-hosted model used for routing, relevance checking, and answer generation.",
        )

        st.markdown("---")

        # --- Numeric sliders ---
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=float(defaults.TEMPERATURE),
            step=0.1,
            help="Controls randomness. 0 = deterministic, higher = more creative.",
        )

        top_k = st.slider(
            "Top-K retrieval",
            min_value=1,
            max_value=20,
            value=int(defaults.TOP_K),
            step=1,
            help="Number of nearest-neighbour documents fetched from ChromaDB.",
        )

        max_iterations = st.slider(
            "Max iterations",
            min_value=1,
            max_value=10,
            value=int(defaults.MAX_ITERATIONS),
            step=1,
            help="How many times the relevance-check loop can retry via web search before forcing an answer.",
        )

        answer_word_limit = st.slider(
            "Answer word limit",
            min_value=10,
            max_value=300,
            value=int(defaults.ANSWER_WORD_LIMIT),
            step=10,
            help="Soft target injected into the generation prompt to control answer length.",
        )

        st.markdown("---")

        # --- Source legend ---
        st.markdown("**Source legend**")
        for label, colour in _SOURCE_COLOURS.items():
            st.markdown(_source_badge(label), unsafe_allow_html=True)
            st.write("")

        st.markdown("---")

        if st.button("Clear chat history", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        # --- Feedback summary ---
        msgs = st.session_state.get("messages", [])
        up_count   = sum(1 for m in msgs if m.get("feedback_rating") == "up")
        down_count = sum(1 for m in msgs if m.get("feedback_rating") == "down")
        rated = up_count + down_count
        if rated:
            st.markdown("---")
            st.markdown("**Session feedback**")
            st.markdown(f"👍 **{up_count}** helpful &nbsp;&nbsp; 👎 **{down_count}** not helpful")
            satisfaction = round(up_count / rated * 100)
            st.progress(satisfaction / 100, text=f"{satisfaction}% satisfaction")

        st.caption("Run `python scripts/ingest.py` once before starting the app.")

    return model, temperature, top_k, max_iterations, answer_word_limit


# ---------------------------------------------------------------------------
# Chat session helpers
# ---------------------------------------------------------------------------
def _init_session() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "active_config" not in st.session_state:
        st.session_state.active_config = None
    # feedback_rating and feedback_comment are stored per-message in messages[i]


def _render_history() -> None:
    for idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                if msg.get("source"):
                    st.markdown(_source_badge(msg["source"]), unsafe_allow_html=True)
                if msg.get("metrics"):
                    _render_metrics(msg["metrics"])
                _render_feedback(idx)


def _run_query(agent, query: str) -> tuple[str, str, dict, float]:
    """Invoke the agent graph and return the response, source, full state, and wall-clock latency.

    Parameters
    ----------
    agent:
        Compiled LangGraph agent.
    query:
        User question string.

    Returns
    -------
    tuple of (response, source, full_state, latency_ms)
    """
    t0 = time.perf_counter()
    full_state = agent.invoke({"query": query})
    latency_ms = (time.perf_counter() - t0) * 1000
    response = (full_state.get("response") or "").strip() or "No response was generated."
    source   = full_state.get("source", "unknown")
    return response, source, full_state, latency_ms


# ---------------------------------------------------------------------------
# Metrics rendering
# ---------------------------------------------------------------------------

def _score_indicator(score: float) -> str:
    """Return a coloured emoji indicator for a 0–1 score."""
    if score >= 0.75:
        return "🟢"
    if score >= 0.5:
        return "🟡"
    return "🔴"


def _render_metrics(metrics: dict) -> None:
    """Render the five-category performance metrics dashboard below each answer.

    Categories
    ----------
    1. Outcome Quality      — goal fulfillment, answer relevance, completeness, 1st-pass
    2. Reasoning Quality    — reasoning score, routing confidence, route, retries
    3. Data Groundedness    — groundedness, citation faithfulness, hallucination, ctx %, reliability
    4. Operational Efficiency — latency, token counts, estimated cost
    5. Feedback             — rendered separately by _render_feedback()

    Parameters
    ----------
    metrics:
        ``dataclasses.asdict(EvaluationResult)`` stored in session state.
    """
    with st.expander("Performance Metrics", expanded=True):
        c1, c2, c3, c4 = st.columns(4)

        # --- 1. Outcome Quality ---
        with c1:
            gf  = metrics.get("goal_fulfillment_score", 0.0)
            ar  = metrics.get("answer_relevance_score", 0.0)
            cmp = metrics.get("answer_completeness_pct", 0.0)
            fps = metrics.get("first_pass_success", False)
            avg_oq = (gf + ar) / 2
            st.markdown(f"**Outcome Quality** {_score_indicator(avg_oq)}")
            st.metric("Goal Fulfillment",    f"{gf:.0%}")
            st.metric("Answer Relevance",    f"{ar:.0%}")
            st.metric("Answer Completeness", f"{cmp:.0f}%")
            st.metric("1st-Pass Success",    "Yes ✓" if fps else "No ✗")

        # --- 2. Reasoning Quality ---
        with c2:
            rs  = metrics.get("reasoning_score", 0.0)
            rc  = metrics.get("routing_confidence", 0.0)
            route = metrics.get("routing_decision", "—").replace("_", " ").title()
            tries = metrics.get("retrieval_attempts", 0)
            avg_rq = (rs + rc) / 2
            st.markdown(f"**Reasoning Quality** {_score_indicator(avg_rq)}")
            st.metric("Reasoning Score",    f"{rs:.0%}")
            st.metric("Routing Confidence", f"{rc:.0%}")
            st.metric("Route Chosen",       route)
            st.metric("Retrieval Attempts", tries)

        # --- 3. Data Groundedness ---
        with c3:
            gs   = metrics.get("groundedness_score", 0.0)
            cf   = metrics.get("citation_faithfulness_score", 0.0)
            hall = metrics.get("hallucination_flag", False)
            cu   = metrics.get("context_utilization_pct", 0.0)
            sr   = metrics.get("source_reliability", 0.0)
            avg_dg = (gs + cf) / 2
            st.markdown(f"**Data Groundedness** {_score_indicator(avg_dg)}")
            st.metric("Groundedness",          f"{gs:.0%}")
            st.metric("Citation Faithfulness", f"{cf:.0%}")
            st.metric("Hallucination",         "Detected ⚠️" if hall else "None ✓")
            st.metric("Context Utilized",      f"{cu:.0f}%")
            st.metric("Source Reliability",    f"{sr:.0%}")

        # --- 4. Operational Efficiency ---
        with c4:
            lat   = metrics.get("total_latency_ms", 0.0)
            ptoks = metrics.get("prompt_tokens", 0)
            ctoks = metrics.get("completion_tokens", 0)
            cost  = metrics.get("estimated_cost_usd", 0.0)
            st.markdown("**Operational Efficiency** ⚙️")
            st.metric("Total Latency",   f"{lat / 1000:.2f}s")
            st.metric("Tokens In",       f"{ptoks:,}")
            st.metric("Tokens Out",      f"{ctoks:,}")
            st.metric("Est. Cost",       f"${cost:.5f}")

        # --- Node timing breakdown ---
        node_timings: dict = metrics.get("node_timings") or {}
        if node_timings:
            st.markdown("---")
            st.markdown("**Node Timings (ms)**")
            cols = st.columns(len(node_timings))
            for col, (node, ms) in zip(cols, node_timings.items()):
                col.metric(node.replace("_", " ").title(), f"{ms:.0f} ms")


def _render_feedback(msg_idx: int) -> None:
    """Render thumbs-up / thumbs-down feedback buttons for a chat message.

    Feedback is stored directly on the message dict:
      ``messages[msg_idx]["feedback_rating"]``  — "up" | "down" | None
      ``messages[msg_idx]["feedback_comment"]`` — optional free-text string

    Only renders for assistant messages that exist in session state.
    Clicking a button updates session state and triggers a rerun so the
    button style and status text update immediately.

    Parameters
    ----------
    msg_idx:
        Index of the assistant message in ``st.session_state.messages``.
    """
    msgs = st.session_state.messages
    if msg_idx >= len(msgs) or msgs[msg_idx]["role"] != "assistant":
        return

    msg    = msgs[msg_idx]
    rating = msg.get("feedback_rating")

    st.markdown("---")
    st.caption("**Was this answer helpful?**")
    col_up, col_down, col_status = st.columns([1, 1, 5])

    with col_up:
        up_type = "primary" if rating == "up" else "secondary"
        if st.button("👍 Yes", key=f"fb_up_{msg_idx}", type=up_type):
            st.session_state.messages[msg_idx]["feedback_rating"] = "up"
            st.rerun()

    with col_down:
        down_type = "primary" if rating == "down" else "secondary"
        if st.button("👎 No", key=f"fb_down_{msg_idx}", type=down_type):
            st.session_state.messages[msg_idx]["feedback_rating"] = "down"
            st.rerun()

    with col_status:
        if rating == "up":
            st.success("Marked as helpful ✓", icon="✅")
        elif rating == "down":
            st.warning("Marked as not helpful", icon="⚠️")

    # Show optional comment box once rated
    if rating is not None:
        comment = st.text_input(
            "Add a comment (optional)",
            value=msg.get("feedback_comment", ""),
            key=f"fb_comment_{msg_idx}",
            placeholder="What could be improved?",
            label_visibility="collapsed",
        )
        if comment != msg.get("feedback_comment", ""):
            st.session_state.messages[msg_idx]["feedback_comment"] = comment


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    _init_session()

    # Render sidebar first — returns live widget values.
    model, temperature, top_k, max_iterations, answer_word_limit = _render_sidebar()
    current_config = (model, temperature, top_k, max_iterations, answer_word_limit)

    # Load (or retrieve from cache) the agent for the current config.
    try:
        agent, settings, eval_llm = _load_agent(*current_config)
    except Exception as exc:
        st.error(
            "**Could not load the agent.** Check the points below and restart the app.\n\n"
            "- Is your `.env` present with `GROQ_API_KEY` and `SERPER_API_KEY`?\n"
            "- Has `python scripts/ingest.py` been run?\n\n"
            f"Error: `{exc}`"
        )
        st.stop()

    # Show a banner when the config changes mid-session.
    if st.session_state.active_config is not None and st.session_state.active_config != current_config:
        st.info("Settings updated — agent rebuilt with the new configuration.")
    st.session_state.active_config = current_config

    # --- Page header ---
    st.title("Medical Q&A Agent")
    st.caption(
        f"Model: `{settings.GROQ_MODEL}` · "
        f"Temp: `{settings.TEMPERATURE}` · "
        f"Top-K: `{settings.TOP_K}` · "
        f"Max iter: `{settings.MAX_ITERATIONS}` · "
        f"Word limit: `{settings.ANSWER_WORD_LIMIT}`"
    )
    st.divider()

    # --- Chat ---
    _render_history()

    if query := st.chat_input("Ask a medical question…"):
        query = query.strip()
        if not query:
            st.warning("Please enter a non-empty question.")
            st.stop()

        st.session_state.messages.append({"role": "user", "content": query, "source": None})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    response, source, full_state, latency_ms = _run_query(agent, query)
                    logger.info("Query answered | source=%s | latency_ms=%.0f", source, latency_ms)
                except Exception as exc:
                    logger.exception("Agent invocation failed: %s", exc)
                    response   = f"An error occurred: `{exc}`"
                    source     = ""
                    full_state = {}
                    latency_ms = 0.0

            st.markdown(response)
            if source:
                st.markdown(_source_badge(source), unsafe_allow_html=True)

            # --- Evaluate and render performance metrics ---
            metrics_dict: dict = {}
            if full_state:
                with st.spinner("Evaluating response quality…"):
                    try:
                        ev = evaluate(
                            llm               = eval_llm,
                            query             = query,
                            context           = full_state.get("context", ""),
                            response          = response,
                            source            = source,
                            route             = full_state.get("route", ""),
                            iteration_count   = full_state.get("iteration_count", 0),
                            prompt_tokens     = full_state.get("prompt_tokens", 0),
                            completion_tokens = full_state.get("completion_tokens", 0),
                            total_latency_ms  = latency_ms,
                            node_timings      = full_state.get("node_timings", {}),
                            answer_word_limit = settings.ANSWER_WORD_LIMIT,
                        )
                        metrics_dict = dataclasses.asdict(ev)
                    except Exception as exc:
                        logger.warning("Metrics evaluation failed: %s", exc)

            if metrics_dict:
                _render_metrics(metrics_dict)

        new_msg_idx = len(st.session_state.messages)
        st.session_state.messages.append({
            "role":             "assistant",
            "content":          response,
            "source":           source,
            "metrics":          metrics_dict,
            "feedback_rating":  None,   # set by _render_feedback
            "feedback_comment": "",
        })
        # Show feedback buttons inline for the just-answered message
        _render_feedback(new_msg_idx)


if __name__ == "__main__":
    main()