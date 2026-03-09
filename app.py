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
import uuid

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
    page_icon=":stethoscope:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for compact metrics and clean UI
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Tighten main content padding */
    .block-container { padding-top: 2rem; max-width: 900px; }

    /* Compact metric cards */
    .metric-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 8px 12px;
        margin-bottom: 6px;
        border-left: 3px solid #dee2e6;
    }
    .metric-card.green  { border-left-color: #28a745; }
    .metric-card.yellow { border-left-color: #ffc107; }
    .metric-card.red    { border-left-color: #dc3545; }

    .metric-label {
        font-size: 0.7rem;
        font-weight: 600;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 2px;
    }
    .metric-value {
        font-size: 0.85rem;
        font-weight: 700;
        color: #212529;
    }
    .metric-sub {
        font-size: 0.65rem;
        color: #868e96;
    }

    /* Source badge */
    .source-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.7rem;
        font-weight: 600;
        color: #ffffff;
        margin-top: 4px;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] { background-color: #fafbfc; }
    [data-testid="stSidebar"] .stMarkdown h2 { font-size: 1.1rem; }

    /* Chat input */
    .stChatInput { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Logging — idempotent; setup_logging guards against duplicate handlers.
# ---------------------------------------------------------------------------
setup_logging(get_settings().LOG_LEVEL)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_GROQ_MODELS = [
    "llama-3.3-70b-versatile",   # default — best quality, 131K context
    "llama-3.1-8b-instant",      # fast, lightweight, 131K context
    "openai/gpt-oss-120b",       # OpenAI open-source, 131K context
    "openai/gpt-oss-20b",        # OpenAI open-source mid-size, 131K context
]

_SOURCE_COLOURS: dict[str, str] = {
    "Medical Q&A Collection": "#1f77b4",
    "Medical Device Manual":  "#2ca02c",
    "Web Search (Serper)":    "#d62728",
}
_FALLBACK_COLOUR = "#7f7f7f"


def _source_badge(source: str) -> str:
    """Return an HTML badge string for the given source label."""
    colour = _SOURCE_COLOURS.get(source, _FALLBACK_COLOUR)
    return f'<span class="source-badge" style="background-color:{colour};">{source}</span>'


def _metric_card(label: str, value: str, sub: str = "", score: float | None = None) -> str:
    """Return HTML for a compact metric card with optional color coding."""
    if score is not None:
        if score >= 0.75:
            cls = "green"
        elif score >= 0.5:
            cls = "yellow"
        else:
            cls = "red"
    else:
        cls = ""
    sub_html = f'<div class="metric-sub">{sub}</div>' if sub else ""
    return (
        f'<div class="metric-card {cls}">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'{sub_html}</div>'
    )


# ---------------------------------------------------------------------------
# Agent cache
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading agent...")
def _load_agent(
    model: str,
    temperature: float,
    top_k: int,
    max_iterations: int,
    answer_word_limit: int,
):
    """Build and cache the LangGraph agent for a specific parameter combination."""
    base = get_settings()
    settings = base.model_copy(update={
        "GROQ_MODEL":         model,
        "TEMPERATURE":        temperature,
        "TOP_K":              top_k,
        "MAX_ITERATIONS":     max_iterations,
        "ANSWER_WORD_LIMIT":  answer_word_limit,
    })
    # Session-only mode: no long-term memory, no checkpointer.
    # Conversation context comes from st.session_state.messages (current session).
    # Long-term memory (AgentMemoryStore + SqliteSaver) can be re-enabled for deployment.
    agent    = build_agent(settings, checkpointer=None, memory_store=None)
    eval_llm = build_llm(settings)
    logger.info(
        "Agent built | model=%s temp=%.1f top_k=%d max_iter=%d word_limit=%d",
        model, temperature, top_k, max_iterations, answer_word_limit,
    )
    return agent, settings, eval_llm


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def _render_sidebar() -> tuple[str, float, int, int, int]:
    """Render sidebar controls and return current parameter values."""
    defaults = get_settings()

    with st.sidebar:
        st.markdown("## Settings")
        st.caption("Agent rebuilds automatically on change.")

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
            help="Groq-hosted model for routing, relevance checking, and generation.",
        )

        # --- Sliders in two columns ---
        col1, col2 = st.columns(2)
        with col1:
            temperature = st.slider(
                "Temperature", 0.0, 2.0,
                value=float(defaults.TEMPERATURE), step=0.1,
                help="0 = deterministic, higher = creative.",
            )
            top_k = st.slider(
                "Top-K", 1, 20,
                value=int(defaults.TOP_K), step=1,
                help="Documents fetched from ChromaDB.",
            )
        with col2:
            max_iterations = st.slider(
                "Max retries", 1, 10,
                value=int(defaults.MAX_ITERATIONS), step=1,
                help="Relevance-check retry limit.",
            )
            answer_word_limit = st.slider(
                "Word limit", 10, 300,
                value=int(defaults.ANSWER_WORD_LIMIT), step=10,
                help="Soft word-count target for answers.",
            )

        st.divider()

        # --- Source legend (compact) ---
        st.markdown("**Sources**")
        legend_html = " &nbsp; ".join(
            _source_badge(label) for label in _SOURCE_COLOURS
        )
        st.markdown(legend_html, unsafe_allow_html=True)

        st.divider()

        if st.button("Clear chat", use_container_width=True, type="secondary"):
            st.session_state.messages = []
            st.rerun()

        # --- Feedback summary ---
        msgs = st.session_state.get("messages", [])
        up_count   = sum(1 for m in msgs if m.get("feedback_rating") == "up")
        down_count = sum(1 for m in msgs if m.get("feedback_rating") == "down")
        rated = up_count + down_count
        if rated:
            satisfaction = round(up_count / rated * 100)
            st.divider()
            st.caption(f"**Feedback:** {up_count} helpful / {down_count} not &middot; {satisfaction}%")
            st.progress(satisfaction / 100)

    return model, temperature, top_k, max_iterations, answer_word_limit


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------
def _init_session() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "active_config" not in st.session_state:
        st.session_state.active_config = None
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())


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
    """Invoke the agent and return (response, source, full_state, latency_ms)."""
    thread_id = st.session_state.get("thread_id", "default")
    config    = {"configurable": {"thread_id": thread_id}}

    history = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in st.session_state.get("messages", [])[-6:]
        if msg["role"] in ("user", "assistant")
    ]

    t0 = time.perf_counter()
    full_state = agent.invoke(
        {"query": query, "thread_id": thread_id, "conversation_history": history},
        config=config,
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    response = (full_state.get("response") or "").strip() or "No response was generated."
    source   = full_state.get("source", "unknown")
    return response, source, full_state, latency_ms


# ---------------------------------------------------------------------------
# Metrics rendering — compact cards with small text
# ---------------------------------------------------------------------------
def _render_metrics(metrics: dict) -> None:
    """Render compact performance metrics below each answer."""
    with st.expander("Performance Metrics", expanded=False):
        gs   = metrics.get("groundedness_score", 0.0)
        gf   = metrics.get("goal_fulfillment_score", 0.0)
        ar   = metrics.get("answer_relevance_score", 0.0)
        rs   = metrics.get("reasoning_score", 0.0)
        rc   = metrics.get("routing_confidence", 0.0)
        cf   = metrics.get("citation_faithfulness_score", 0.0)
        hall = metrics.get("hallucination_flag", False)
        lat  = metrics.get("total_latency_ms", 0.0)
        ptoks = metrics.get("prompt_tokens", 0)
        ctoks = metrics.get("completion_tokens", 0)
        ttoks = metrics.get("total_tokens", 0)
        cost  = metrics.get("estimated_cost_usd", 0.0)
        comp  = metrics.get("answer_completeness_pct", 0.0)
        ctx_u = metrics.get("context_utilization_pct", 0.0)
        src_r = metrics.get("source_reliability", 0.0)
        route = metrics.get("routing_decision", "")
        retries = metrics.get("retrieval_attempts", 0)
        timings = metrics.get("node_timings", {})

        # --- Row 1: Quality scores ---
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(_metric_card(
                "Groundedness", f"{gs:.0%}",
                "Hallucination!" if hall else "No hallucination",
                score=gs,
            ), unsafe_allow_html=True)
        with c2:
            st.markdown(_metric_card(
                "Goal Fulfillment", f"{gf:.0%}", score=gf,
            ), unsafe_allow_html=True)
        with c3:
            st.markdown(_metric_card(
                "Answer Relevance", f"{ar:.0%}", score=ar,
            ), unsafe_allow_html=True)
        with c4:
            st.markdown(_metric_card(
                "Reasoning", f"{rs:.0%}", score=rs,
            ), unsafe_allow_html=True)

        # --- Row 2: Routing + Grounding ---
        c5, c6, c7, c8 = st.columns(4)
        with c5:
            st.markdown(_metric_card(
                "Routing Confidence", f"{rc:.0%}",
                f"Route: {route}", score=rc,
            ), unsafe_allow_html=True)
        with c6:
            st.markdown(_metric_card(
                "Citation Faithfulness", f"{cf:.0%}", score=cf,
            ), unsafe_allow_html=True)
        with c7:
            st.markdown(_metric_card(
                "Context Utilization", f"{ctx_u:.0f}%",
                f"Source reliability: {src_r:.0%}",
            ), unsafe_allow_html=True)
        with c8:
            st.markdown(_metric_card(
                "Completeness", f"{comp:.0f}%",
                f"Retries: {retries}" + (" (first pass)" if retries <= 1 else ""),
            ), unsafe_allow_html=True)

        # --- Row 3: Efficiency (single row) ---
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Efficiency</div>'
            f'<div class="metric-value">'
            f'{lat / 1000:.2f}s latency &nbsp;&middot;&nbsp; '
            f'{ptoks} in / {ctoks} out ({ttoks} total) &nbsp;&middot;&nbsp; '
            f'${cost:.5f}'
            f'</div>'
            f'<div class="metric-sub">'
            f'{" → ".join(f"{k}: {v:.0f}ms" for k, v in timings.items()) if timings else "No node timings"}'
            f'</div></div>',
            unsafe_allow_html=True,
        )


def _render_feedback(msg_idx: int) -> None:
    """Render compact feedback buttons for an assistant message."""
    msgs = st.session_state.messages
    if msg_idx >= len(msgs) or msgs[msg_idx]["role"] != "assistant":
        return

    msg    = msgs[msg_idx]
    rating = msg.get("feedback_rating")

    col_up, col_down, col_status = st.columns([1, 1, 6])
    with col_up:
        up_type = "primary" if rating == "up" else "secondary"
        if st.button("👍", key=f"fb_up_{msg_idx}", type=up_type):
            st.session_state.messages[msg_idx]["feedback_rating"] = "up"
            st.rerun()
    with col_down:
        down_type = "primary" if rating == "down" else "secondary"
        if st.button("👎", key=f"fb_down_{msg_idx}", type=down_type):
            st.session_state.messages[msg_idx]["feedback_rating"] = "down"
            st.rerun()
    with col_status:
        if rating == "up":
            st.caption("Marked helpful")
        elif rating == "down":
            st.caption("Marked not helpful")

    if rating is not None:
        comment = st.text_input(
            "Comment",
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

    model, temperature, top_k, max_iterations, answer_word_limit = _render_sidebar()
    current_config = (model, temperature, top_k, max_iterations, answer_word_limit)

    try:
        agent, settings, eval_llm = _load_agent(*current_config)
    except Exception as exc:
        st.error(
            "**Could not load the agent.** Check below and restart.\n\n"
            "- Is `.env` present with `GROQ_API_KEY` and `SERPER_API_KEY`?\n"
            "- Has `python scripts/ingest.py` been run?\n\n"
            f"Error: `{exc}`"
        )
        st.stop()

    if st.session_state.active_config is not None and st.session_state.active_config != current_config:
        st.toast("Settings updated — agent rebuilt.", icon="🔄")
    st.session_state.active_config = current_config

    # --- Header ---
    st.markdown("## Medical Q&A Agent")
    st.caption(
        f"`{settings.GROQ_MODEL}` · "
        f"temp {settings.TEMPERATURE} · "
        f"top-k {settings.TOP_K} · "
        f"retries {settings.MAX_ITERATIONS} · "
        f"~{settings.ANSWER_WORD_LIMIT} words"
    )

    # --- Chat ---
    _render_history()

    if query := st.chat_input("Ask a question..."):
        query = query.strip()
        if not query:
            st.warning("Please enter a non-empty question.")
            st.stop()

        st.session_state.messages.append({"role": "user", "content": query, "source": None})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
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

            # --- Evaluate and render metrics ---
            metrics_dict: dict = {}
            if full_state:
                with st.spinner("Evaluating..."):
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
                        # --- Log full pipeline summary ---
                        logger.info(
                            "=== PIPELINE SUMMARY ===\n"
                            "  Query:                %s\n"
                            "  Route:                %s\n"
                            "  Source:               %s\n"
                            "  Iterations:           %d\n"
                            "  First pass success:   %s\n"
                            "  --- Token Usage ---\n"
                            "  Prompt tokens:        %d\n"
                            "  Completion tokens:    %d\n"
                            "  Total tokens:         %d\n"
                            "  Estimated cost:       $%.6f\n"
                            "  --- Latency ---\n"
                            "  Total latency:        %.1f ms\n"
                            "  Node timings:         %s\n"
                            "  --- Quality Scores (LLM-as-judge) ---\n"
                            "  Goal fulfillment:     %.2f\n"
                            "  Answer relevance:     %.2f\n"
                            "  Reasoning:            %.2f\n"
                            "  Routing confidence:   %.2f\n"
                            "  Groundedness:         %.2f\n"
                            "  Citation faithfulness: %.2f\n"
                            "  Hallucination flag:   %s\n"
                            "  --- Deterministic Metrics ---\n"
                            "  Answer completeness:  %.1f%%\n"
                            "  Context utilization:  %.1f%%\n"
                            "  Source reliability:   %.2f\n"
                            "========================",
                            query,
                            ev.routing_decision,
                            source,
                            ev.retrieval_attempts,
                            ev.first_pass_success,
                            ev.prompt_tokens,
                            ev.completion_tokens,
                            ev.total_tokens,
                            ev.estimated_cost_usd,
                            ev.total_latency_ms,
                            ev.node_timings,
                            ev.goal_fulfillment_score,
                            ev.answer_relevance_score,
                            ev.reasoning_score,
                            ev.routing_confidence,
                            ev.groundedness_score,
                            ev.citation_faithfulness_score,
                            ev.hallucination_flag,
                            ev.answer_completeness_pct,
                            ev.context_utilization_pct,
                            ev.source_reliability,
                        )
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
            "feedback_rating":  None,
            "feedback_comment": "",
        })
        _render_feedback(new_msg_idx)


if __name__ == "__main__":
    main()
