"""A small StateGraph-style coordinator for classifying intent and retrieving answers.

This module implements a simple graph with three nodes:
- classify_intent: decide whether the question is a policy question
- retrieve_and_answer: run a real retrieval against ChromaDB and return a mock answer
- direct_answer: return a canned direct answer when classify says so

The graph uses a MOCK_LLM environment variable to choose between "mock" and
"real-LLM" branches. By default MOCK_LLM is true (mock mode).

Note: the real-LLM branch contains a placeholder showing where a Groq or other
LLM call would be made. The mock branches fully run locally and do not call
external LLM services.
"""

from __future__ import annotations

import os
from typing import TypedDict, List, Dict, Any

from sentence_transformers import SentenceTransformer
import chromadb

MOCK_LLM = os.getenv("MOCK_LLM", "1") != "0"

# Simple state type for the graph handlers
class State(TypedDict, total=False):
    query: str
    intent: str
    retrieved: List[Dict[str, Any]]
    answer: str
    sources: List[str]
    confidence: float


# Keywords to treat as policy questions
_POLICY_KEYWORDS = [
    "delivery",
    "return",
    "refund",
    "membership",
    "tracking",
    "cancel",
    "gift card",
    "support hours",
]

_COLLECTION_NAME = "zepto_docs"

# Instantiate embedding model and chroma client once
_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
_CLIENT = chromadb.Client()

try:
    _COLLECTION = _CLIENT.get_collection(name=_COLLECTION_NAME)
except Exception:
    # If it doesn't exist, create an empty collection with the requested metadata
    try:
        _COLLECTION = _CLIENT.create_collection(name=_COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    except Exception:
        _COLLECTION = None


def classify_intent(state: State) -> State:
    """Classify the intent as 'policy_question' or 'general_question'.

    Mock policy: if the query contains any keyword from _POLICY_KEYWORDS -> policy_question
    Otherwise general_question.
    """
    q = state.get("query", "").lower()
    if MOCK_LLM:
        if any(k in q for k in _POLICY_KEYWORDS):
            state["intent"] = "policy_question"
        else:
            state["intent"] = "general_question"
        return state

    # Real-LLM branch (placeholder): call a Groq LLM or other model and parse intent
    # The real branch should also validate and retry if output is malformed.
    # For brevity we fall back to the simple rule above if the LLM flow fails.
    try:
        # Example placeholder pseudocode:
        # from groq import Groq
        # llm = Groq(api_key=os.getenv('GROQ_API_KEY'))
        # prompt = f"Classify intent: {state['query']}"
        # resp = llm.complete(prompt)
        # parsed = resp.text.strip()
        # state['intent'] = parsed
        state["intent"] = "policy_question" if any(k in q for k in _POLICY_KEYWORDS) else "general_question"
    except Exception:
        state["intent"] = "general_question"
    return state


def retrieve_and_answer(state: State) -> State:
    """Run retrieval against the Chroma collection and return a mock answer.

    This function always performs a real vector retrieval (top 3 by cosine).
    The answer branch is mocked: returns the first ~200 characters of the top
    retrieved chunk with sources and confidence=1.0.
    """
    if _COLLECTION is None:
        state["answer"] = "No knowledge collection is available. Please run ingest.py first."
        state["sources"] = []
        state["confidence"] = 0.0
        return state

    query = state.get("query", "")
    if not query:
        state["answer"] = "Empty query received."
        state["sources"] = []
        state["confidence"] = 0.0
        return state

    # Embed the query and run a cosine-based search
    q_emb = _MODEL.encode([query], show_progress_bar=False)[0].tolist()

    try:
        result = _COLLECTION.query(query_embeddings=[q_emb], n_results=3, include=["documents", "ids", "metadatas"]) 
    except Exception as exc:
        # Some chroma clients use slightly different method names; try a safe wrapper
        try:
            result = _COLLECTION.query(queries=[query], n_results=3)
        except Exception:
            state["answer"] = f"Retrieval failed: {exc}"
            state["sources"] = []
            state["confidence"] = 0.0
            return state

    # result is typically a dict with list entries per query
    docs = []
    ids = []
    if isinstance(result, dict):
        docs = result.get("documents", [[]])[0]
        ids = result.get("ids", [[]])[0]
    else:
        # fallback: try to parse
        try:
            docs = result["documents"][0]
            ids = result["ids"][0]
        except Exception:
            docs = []
            ids = []

    if not docs:
        state["answer"] = "No relevant documents were found in the knowledge base."
        state["sources"] = []
        state["confidence"] = 0.0
        return state

    top = docs[0]
    snippet = top[:200] if len(top) > 200 else top
    state["answer"] = f"Based on the retrieved context: {snippet}"
    state["sources"] = ids
    state["confidence"] = 1.0
    state["retrieved"] = [{"id": i, "doc": d} for i, d in zip(ids, docs)]

    return state


def direct_answer(state: State) -> State:
    """Return a canned direct answer used for non-retrieval responses."""
    state["answer"] = "I can only answer questions about Zepto policies right now."
    state["sources"] = []
    # Per spec, mock direct_answer should return confidence = 1.0
    state["confidence"] = 1.0
    return state


def run_graph(query: str) -> Dict[str, Any]:
    """Execute the small decision graph for a single query and return the state.

    Returns a dict suitable for the API response.
    """
    state: State = {"query": query}
    state = classify_intent(state)
    intent = state.get("intent")
    if intent == "policy_question":
        state = retrieve_and_answer(state)
    else:
        # For general questions return direct answer (mock) — could be extended
        state = direct_answer(state)
    return dict(state)


if __name__ == "__main__":
    # Quick local demo when run as a script
    q = input("Enter a question: ")
    out = run_graph(q)
    print("Answer:", out.get("answer"))
    print("Sources:", out.get("sources"))
    print("Confidence:", out.get("confidence"))
