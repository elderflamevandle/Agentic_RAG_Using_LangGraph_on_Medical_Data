"""LLM-as-judge evaluation for the agentic RAG pipeline.

Five metric categories are computed for every pipeline run:

1. **Outcome Quality**      — goal_fulfillment_score, answer_relevance_score,
                               answer_completeness_pct, first_pass_success
2. **Reasoning Quality**    — reasoning_score, routing_confidence,
                               routing_decision, retrieval_attempts
3. **Data Groundedness**    — groundedness_score, citation_faithfulness_score,
                               hallucination_flag, context_utilization_pct,
                               source_reliability
4. **Operational Efficiency**— total_latency_ms, node_timings,
                               prompt_tokens, completion_tokens, estimated_cost_usd
5. **Feedback**             — user_rating (👍/👎) and user_comment are stored
                               in the Streamlit session state (app.py), not here.

A single LLM-as-judge call (structured output with six fields) produces all
qualitative scores in one round-trip to minimise latency and cost.
All other metrics are computed deterministically.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Groq llama-3.3-70b-versatile pricing (USD per token, as of 2026-Q1)
# ---------------------------------------------------------------------------
_INPUT_COST_PER_TOKEN  = 0.59 / 1_000_000   # $0.59 / 1M input  tokens
_OUTPUT_COST_PER_TOKEN = 0.79 / 1_000_000   # $0.79 / 1M output tokens

# Groundedness threshold below which a hallucination is flagged
_HALLUCINATION_THRESHOLD = 0.5

# Source reliability scores — curated local datasets are more trustworthy than live web
_SOURCE_RELIABILITY: dict[str, float] = {
    "Medical Q&A Collection": 1.0,   # curated, domain-specific dataset
    "Medical Device Manual":  1.0,   # curated, domain-specific dataset
    "Web Search (Serper)":    0.6,   # live web — less verified
}
_DEFAULT_RELIABILITY = 0.5

# Medical keyword sets used for tool-usage accuracy heuristic
_DEVICE_KEYWORDS = {
    "device", "model", "manual", "manufacturer", "serial", "equipment",
    "machine", "instrument", "calibrate", "specification", "sensor", "probe",
}
_MEDICAL_KEYWORDS = {
    "symptom", "disease", "treatment", "diagnosis", "medicine", "drug",
    "therapy", "condition", "patient", "health", "infection", "pain",
    "surgery", "vaccine", "dose", "prescription",
}


# ---------------------------------------------------------------------------
# Pydantic schema for LLM-as-judge structured output  (single round-trip)
# ---------------------------------------------------------------------------

class _JudgeScores(BaseModel):
    """Six-dimensional structured output from the LLM-as-judge call.

    All scores are in [0.0, 1.0].  A single prompt asks the LLM to score
    all six dimensions simultaneously to minimise API calls and latency.
    """

    # Outcome Quality
    goal_fulfillment_score: float = Field(
        ..., ge=0.0, le=1.0,
        description=(
            "How completely does the answer address the query? "
            "0 = not at all, 1 = fully answered."
        ),
    )
    answer_relevance_score: float = Field(
        ..., ge=0.0, le=1.0,
        description=(
            "Is the answer relevant and on-topic for the user's query? "
            "0 = completely off-topic, 1 = directly on-topic."
        ),
    )

    # Reasoning Quality
    reasoning_score: float = Field(
        ..., ge=0.0, le=1.0,
        description=(
            "Is the answer logically coherent and well-structured with clear reasoning? "
            "0 = incoherent, 1 = excellent step-by-step logic."
        ),
    )
    routing_confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description=(
            "Given the query, does the chosen retrieval route seem like the right tool? "
            "0 = clearly wrong tool, 1 = clearly the best tool for this query."
        ),
    )

    # Data Groundedness
    groundedness_score: float = Field(
        ..., ge=0.0, le=1.0,
        description=(
            "Is every factual claim in the answer directly supported by the retrieved context? "
            "0 = fully hallucinated, 1 = every claim is grounded."
        ),
    )
    citation_faithfulness_score: float = Field(
        ..., ge=0.0, le=1.0,
        description=(
            "Does the answer accurately represent what the context says, without distortion? "
            "0 = misrepresents context, 1 = perfectly faithful to context."
        ),
    )


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class EvaluationResult:
    """All performance metrics for a single pipeline run.

    Fields are grouped into five evaluation categories.
    The Feedback category (user_rating, user_comment) is stored separately
    in Streamlit session state (app.py) and not included here.
    """

    # --- 1. Outcome Quality ---
    goal_fulfillment_score:   float = 0.0   # 0–1   LLM-as-judge
    answer_relevance_score:   float = 0.0   # 0–1   LLM-as-judge
    answer_completeness_pct:  float = 0.0   # 0–100 actual_words / word_limit
    first_pass_success:       bool  = False  # True  if relevance passed on first try

    # --- 2. Reasoning Quality ---
    reasoning_score:          float = 0.0   # 0–1   LLM-as-judge
    routing_confidence:       float = 0.0   # 0–1   LLM-as-judge
    routing_decision:         str   = ""    # e.g. "retrieve_qna"
    retrieval_attempts:       int   = 0     # iteration_count from state

    # --- 3. Data Groundedness ---
    groundedness_score:           float = 0.0   # 0–1   LLM-as-judge
    citation_faithfulness_score:  float = 0.0   # 0–1   LLM-as-judge
    hallucination_flag:           bool  = False  # True when groundedness < threshold
    context_utilization_pct:      float = 0.0   # 0–100 word-overlap %
    source_reliability:           float = 0.0   # 0–1   lookup table

    # --- 4. Operational Efficiency ---
    total_latency_ms:     float = 0.0
    node_timings:         dict  = field(default_factory=dict)  # {node: ms}
    prompt_tokens:        int   = 0
    completion_tokens:    int   = 0
    total_tokens:         int   = 0
    estimated_cost_usd:   float = 0.0


# ---------------------------------------------------------------------------
# Deterministic helper functions
# ---------------------------------------------------------------------------

def _context_utilization(context: str, response: str) -> float:
    """Return the % of answer words that appear verbatim in the retrieved context.

    Proxy for how much the answer draws on retrieved material vs. parametric
    model knowledge.
    """
    ctx_words  = set(re.findall(r"\w+", context.lower()))
    resp_words = re.findall(r"\w+", response.lower())
    if not resp_words:
        return 0.0
    hits = sum(1 for w in resp_words if w in ctx_words)
    return round(hits / len(resp_words) * 100, 1)


def _source_reliability(source: str) -> float:
    """Map the human-readable source label to a reliability score (0–1)."""
    return _SOURCE_RELIABILITY.get(source, _DEFAULT_RELIABILITY)


# ---------------------------------------------------------------------------
# Main evaluation entry point
# ---------------------------------------------------------------------------

def evaluate(
    *,
    llm: ChatGroq,
    query: str,
    context: str,
    response: str,
    source: str,
    route: str,
    iteration_count: int,
    prompt_tokens: int,
    completion_tokens: int,
    total_latency_ms: float,
    node_timings: dict,
    answer_word_limit: int,
) -> EvaluationResult:
    """Compute all five metric categories for one pipeline run.

    Deterministic metrics (efficiency, completeness, context overlap, source
    reliability) are calculated immediately.  A single LLM-as-judge call with
    structured output scores all six qualitative dimensions in one round-trip.

    Parameters
    ----------
    llm:
        ChatGroq instance reused from the agent — no additional configuration.
    query:
        The user's original question.
    context:
        Raw text retrieved by the pipeline (``state["context"]``).
    response:
        Final answer text (``state["response"]``).
    source:
        Human-readable retrieval source label (e.g. "Medical Q&A Collection").
    route:
        Routing decision string (e.g. ``"retrieve_qna"``).
    iteration_count:
        Number of relevance-check loop iterations completed.
    prompt_tokens, completion_tokens:
        Token counts from Groq ``usage_metadata`` (written by generate node).
    total_latency_ms:
        Wall-clock time for the full graph invocation.
    node_timings:
        Per-node latency dict written by the ``_timed()`` wrapper in build.py.
    answer_word_limit:
        Configured word limit — used to compute completeness %.

    Returns
    -------
    EvaluationResult
        All metrics populated.  LLM-judge scores default to 0.0 on failure
        (error logged, never raised) so the UI degrades gracefully.
    """
    result = EvaluationResult()

    # =========================================================================
    # 4. Operational Efficiency — pure arithmetic, always succeeds first
    # =========================================================================
    total_tokens = prompt_tokens + completion_tokens
    result.total_latency_ms   = round(total_latency_ms, 1)
    result.node_timings       = dict(node_timings)
    result.prompt_tokens      = prompt_tokens
    result.completion_tokens  = completion_tokens
    result.total_tokens       = total_tokens
    result.estimated_cost_usd = round(
        prompt_tokens * _INPUT_COST_PER_TOKEN
        + completion_tokens * _OUTPUT_COST_PER_TOKEN,
        6,
    )

    # =========================================================================
    # 1. Outcome Quality — deterministic part
    # =========================================================================
    answer_words = len(response.split())
    result.answer_completeness_pct = round(
        min(answer_words / max(answer_word_limit, 1) * 100, 100), 1
    )
    result.first_pass_success = iteration_count <= 1

    # =========================================================================
    # 2. Reasoning Quality — deterministic part
    # =========================================================================
    result.routing_decision   = route
    result.retrieval_attempts = max(iteration_count, 1)

    # =========================================================================
    # 3. Data Groundedness — deterministic part
    # =========================================================================
    result.context_utilization_pct = _context_utilization(context, response)
    result.source_reliability       = _source_reliability(source)

    # =========================================================================
    # LLM-as-judge — ONE call, SIX scores, covering categories 1, 2, and 3
    # =========================================================================
    try:
        judge = llm.with_structured_output(_JudgeScores)
        judge_prompt = (
            "You are an impartial evaluator of RAG (Retrieval-Augmented Generation) pipeline outputs.\n"
            "Score the ANSWER on SIX dimensions using the required structured schema.\n"
            "Be strict and objective — scores should reflect actual quality, not optimism.\n\n"
            f"QUERY:\n{query}\n\n"
            f"RETRIEVAL ROUTE USED: {route}\n\n"
            f"RETRIEVED CONTEXT (may be truncated to 2000 chars):\n{context[:2000]}\n\n"
            f"ANSWER:\n{response}\n\n"
            "Scoring instructions (all scores 0.0–1.0):\n"
            "  goal_fulfillment_score    : Does the answer fully address what the user asked?\n"
            "  answer_relevance_score    : Is the answer on-topic and directly useful?\n"
            "  reasoning_score           : Is the logic clear, structured, and step-by-step?\n"
            "  routing_confidence        : Was the retrieval route a sensible tool choice for this query?\n"
            "  groundedness_score        : Is every claim supported by the retrieved context?\n"
            "  citation_faithfulness_score: Does the answer accurately represent the context without distortion?\n"
        )
        scores: _JudgeScores = judge.invoke(judge_prompt)

        # Outcome Quality
        result.goal_fulfillment_score = round(scores.goal_fulfillment_score, 2)
        result.answer_relevance_score = round(scores.answer_relevance_score, 2)

        # Reasoning Quality
        result.reasoning_score   = round(scores.reasoning_score, 2)
        result.routing_confidence = round(scores.routing_confidence, 2)

        # Data Groundedness
        result.groundedness_score          = round(scores.groundedness_score, 2)
        result.citation_faithfulness_score = round(scores.citation_faithfulness_score, 2)
        result.hallucination_flag          = scores.groundedness_score < _HALLUCINATION_THRESHOLD

        logger.info(
            "LLM-as-judge | goal=%.2f relevance=%.2f reasoning=%.2f "
            "routing_conf=%.2f grounded=%.2f citation=%.2f",
            result.goal_fulfillment_score,
            result.answer_relevance_score,
            result.reasoning_score,
            result.routing_confidence,
            result.groundedness_score,
            result.citation_faithfulness_score,
        )

    except Exception as exc:
        logger.warning("LLM-as-judge evaluation failed: %s", exc)
        # All judge scores remain 0.0 — UI will display N/A gracefully

    return result
