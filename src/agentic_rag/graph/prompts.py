"""Centralised prompt templates for every LangGraph node.

Each constant is a format-string with named placeholders (e.g. ``{query}``,
``{history_text}``).  Node factories in ``nodes.py`` call ``.format()`` at
runtime to inject dynamic values.

Keeping prompts here makes them easy to find, compare, and tune without
touching node logic.
"""

# ---------------------------------------------------------------------------
# QUERY_REWRITER — resolves pronouns / coreferences in follow-up queries
# ---------------------------------------------------------------------------
QUERY_REWRITER_PROMPT = (
    "You are a query resolution assistant.\n\n"
    "Your ONLY job is to resolve coreferences — pronouns or implicit "
    "references that point back to something mentioned earlier.\n\n"
    "Rewrite the query ONLY if it contains explicit coreference markers such as:\n"
    "  - Pronouns: it, its, they, their, them, this, that, these, those\n"
    "  - Implicit references: 'the disease', 'the condition', 'the device', "
    "'the treatment', 'the same', 'the above'\n\n"
    "IMPORTANT rules:\n"
    "  - If the query introduces a completely new topic with no pronouns, "
    "set needs_rewrite=False — even if there is prior conversation history.\n"
    "  - Do NOT add context from history to a standalone question. "
    "A question like 'What is UMB?' or 'Tell me about Bitcoin' is already "
    "self-contained and must NOT be rewritten.\n"
    "  - Only rewrite when the query is genuinely ambiguous without history.\n\n"
    "Conversation history:\n{history_text}\n\n"
    "Follow-up query: {query}"
)

# ---------------------------------------------------------------------------
# ROUTER — classifies the query and picks the right retriever
# ---------------------------------------------------------------------------
ROUTER_PROMPT = (
    "You are a routing agent. Choose exactly one data source to answer the query.\n\n"
    "Options:\n"
    "  retrieve_qna    — use ONLY for well-known medical knowledge: named diseases\n"
    "                     (e.g. diabetes, hypertension, asthma, cancer), symptoms,\n"
    "                     treatments, medications, anatomy, medical procedures.\n"
    "  retrieve_device — use ONLY for medical device manuals: specific device model\n"
    "                     numbers, manufacturers, device operating instructions,\n"
    "                     calibration, maintenance of medical equipment.\n"
    "  web_search      — use for EVERYTHING else. This includes:\n"
    "                     - Unknown acronyms or abbreviations (e.g. UMB, FAANG)\n"
    "                     - Company names, people, organisations\n"
    "                     - Financial, legal, or business topics\n"
    "                     - Current events, news, recent developments\n"
    "                     - Technology, programming, software\n"
    "                     - General knowledge not related to medicine\n"
    "                     - Anything you are NOT 100%% certain is a medical topic\n\n"
    "Decision rules:\n"
    "  1. If the query mentions a specific, well-known disease or medical term\n"
    "     (diabetes, hypertension, pneumonia, etc.) → retrieve_qna\n"
    "  2. If the query mentions a medical device brand/model → retrieve_device\n"
    "  3. For ANYTHING else, or if you are even slightly unsure → web_search\n\n"
    "Examples:\n"
    "  'What is diabetes?'           → retrieve_qna\n"
    "  'Symptoms of hypertension'    → retrieve_qna\n"
    "  'How to calibrate Philips X3' → retrieve_device\n"
    "  'What is UMB?'               → web_search\n"
    "  'Tell me about Tesla'         → web_search\n"
    "  'Latest COVID news'           → web_search\n"
    "  'What is machine learning?'   → web_search\n\n"
    "Query: {query}"
)

# ---------------------------------------------------------------------------
# RELEVANCE_CHECKER — strict judgment: does context ANSWER the query?
# ---------------------------------------------------------------------------
RELEVANCE_CHECKER_PROMPT = (
    "You are a strict relevance judge.\n\n"
    "Decide whether the retrieved context below contains a DIRECT, SPECIFIC\n"
    "answer to the user's query.\n\n"
    "Rules:\n"
    "  - Answer 'Yes' ONLY if the context provides concrete information that\n"
    "    directly addresses what the user asked.\n"
    "  - Answer 'No' if:\n"
    "    * The context only mentions the topic without answering the question.\n"
    "    * The context is about a different (even related) subject.\n"
    "    * The context is too vague or generic to be useful.\n"
    "    * The context does not contain factual information relevant to the query.\n\n"
    "User Query: {query}\n\n"
    "Retrieved Context:\n{context}"
)

# ---------------------------------------------------------------------------
# AUGMENT — assembles the final RAG prompt for answer generation
# ---------------------------------------------------------------------------
AUGMENT_PROMPT = (
    "You are a knowledgeable assistant. "
    "Answer the question using *only* the context provided below.\n"
    "If the context does not contain enough information, "
    "briefly state what is missing rather than guessing.\n\n"
    "{history_section}"
    "Source: {source}\n\n"
    "Context:\n{context}\n\n"
    "Question: {query}\n\n"
    "Constraint: Keep the answer within approximately {word_limit} words."
)
