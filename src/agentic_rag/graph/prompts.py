"""Centralised prompt templates for every LangGraph node.

Each constant is a format-string with named placeholders (e.g. ``{query}``,
``{history_text}``).  Node factories in ``nodes.py`` call ``.format()`` at
runtime to inject dynamic values.

Keeping prompts here makes them easy to find, compare, and tune without
touching node logic.
"""

# ---------------------------------------------------------------------------
# QUERY_REWRITER — resolves pronouns / coreferences in follow-up queries
# Uses chain-of-thought with an explicit decision gate.
# ---------------------------------------------------------------------------
QUERY_REWRITER_PROMPT = (
    "Role: You are a coreference resolution specialist.\n\n"
    "Task: Examine the follow-up query and decide if it needs rewriting.\n\n"
    "Step 1 — Scan for coreference markers:\n"
    "  Does the query contain any of these?\n"
    "    Pronouns: it, its, they, their, them, this, that, these, those\n"
    "    Implicit refs: 'the disease', 'the condition', 'the device',\n"
    "                   'the treatment', 'the same', 'the above'\n\n"
    "Step 2 — If YES markers found:\n"
    "  Look at conversation history and identify what each marker refers to.\n"
    "  Rewrite the query by replacing every marker with its referent.\n\n"
    "Step 3 — If NO markers found:\n"
    "  The query is standalone. Do NOT rewrite it, even if history exists.\n"
    "  'What is Bitcoin?' after a diabetes conversation → standalone, no rewrite.\n"
    "  'What is UMB?' after any conversation → standalone, no rewrite.\n\n"
    "Conversation history:\n{history_text}\n\n"
    "Follow-up query: {query}"
)

# ---------------------------------------------------------------------------
# ROUTER — classifies the query and picks the right retriever
# Uses elimination-based routing with a confidence gate.
# ---------------------------------------------------------------------------
ROUTER_PROMPT = (
    "Role: You are a query classifier. Route the query to exactly one source.\n\n"
    "Sources:\n"
    "  retrieve_qna    → Medical encyclopedia. Contains: named diseases\n"
    "                     (diabetes, cancer, asthma, pneumonia, hypertension...),\n"
    "                     symptoms, treatments, drug information, anatomy.\n\n"
    "  retrieve_device → Device manuals. Contains: medical equipment specs,\n"
    "                     operating instructions, model numbers (e.g. Philips X3,\n"
    "                     Medtronic 780G).\n\n"
    "  web_search      → Live internet search. The DEFAULT choice.\n\n"
    "Classification logic:\n"
    "  Ask yourself: 'Am I 100%% certain this is about a specific, well-known\n"
    "  medical condition or a specific medical device?'\n\n"
    "  - If YES and it is a disease/treatment → retrieve_qna\n"
    "  - If YES and it is a device model      → retrieve_device\n"
    "  - If NO, or even 1%% uncertain         → web_search\n\n"
    "  web_search is the SAFE DEFAULT. Choosing it is never wrong.\n"
    "  Choosing retrieve_qna for a non-medical query IS wrong.\n\n"
    "Examples of WRONG routing (avoid these):\n"
    "  'What is UMB?'        → retrieve_qna  WRONG (UMB is not a disease)\n"
    "  'Tell me about Apple' → retrieve_qna  WRONG (company, not medical)\n"
    "  'Latest COVID stats'  → retrieve_qna  WRONG (current events → web)\n\n"
    "Examples of CORRECT routing:\n"
    "  'What is diabetes?'           → retrieve_qna   CORRECT\n"
    "  'Symptoms of pneumonia'       → retrieve_qna   CORRECT\n"
    "  'Philips X3 calibration'      → retrieve_device CORRECT\n"
    "  'What is UMB?'               → web_search     CORRECT\n"
    "  'Who is Elon Musk?'          → web_search     CORRECT\n"
    "  'What is machine learning?'   → web_search     CORRECT\n\n"
    "Query: {query}"
)

# ---------------------------------------------------------------------------
# RELEVANCE_CHECKER — strict judgment: does context ANSWER the query?
# Uses an "answer extraction test" — could a stranger answer from this text?
# ---------------------------------------------------------------------------
RELEVANCE_CHECKER_PROMPT = (
    "Role: You are a strict quality gate in a retrieval pipeline.\n\n"
    "Task: Determine if the retrieved context contains a DIRECT ANSWER\n"
    "to the user's query. Not just related text — an actual answer.\n\n"
    "Judgment criteria:\n"
    "  Answer YES only if ALL of these are true:\n"
    "    1. The context contains specific facts that address the query\n"
    "    2. A human reading this context could write a correct answer\n"
    "    3. The information is about the EXACT topic asked, not a related one\n\n"
    "  Answer NO if ANY of these are true:\n"
    "    1. The context only mentions the topic name without useful details\n"
    "    2. The context discusses a related but different subject\n"
    "    3. The context is generic filler text that does not answer anything\n"
    "    4. You would need external knowledge to bridge the gap\n\n"
    "Think of it this way: If you handed this context to someone who\n"
    "knows nothing about the topic, could THEY answer the question\n"
    "using ONLY this text? If not → No.\n\n"
    "User Query: {query}\n\n"
    "Retrieved Context:\n{context}"
)

# ---------------------------------------------------------------------------
# AUGMENT — assembles the final RAG prompt for answer generation
# Source-aware answering with strict grounding.
# ---------------------------------------------------------------------------
AUGMENT_PROMPT = (
    "Role: You are an answer synthesis assistant.\n"
    "You must ONLY use the provided context. You have NO other knowledge.\n\n"
    "Instructions:\n"
    "  1. Read the context carefully\n"
    "  2. Extract the relevant facts that answer the question\n"
    "  3. Compose a clear, concise answer using ONLY those facts\n"
    "  4. If the context partially answers the question, answer what you\n"
    "     can and state what specific information is missing\n"
    "  5. NEVER invent facts, statistics, or details not in the context\n\n"
    "{history_section}"
    "Source: {source}\n\n"
    "Context:\n{context}\n\n"
    "Question: {query}\n\n"
    "Answer in approximately {word_limit} words."
)
