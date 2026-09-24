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

import json
import os
from pathlib import Path
from typing import TypedDict, List, Dict, Any

from sentence_transformers import SentenceTransformer
import chromadb

MOCK_LLM = os.getenv("MOCK_LLM", "1") != "0"

BASE_DIR = Path(__file__).resolve().parent
PERSIST_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "zepto_docs"

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

# Instantiate embedding model and persistent Chroma client once.
# Keep the same path and collection name as ingest.py to ensure retrieval sees the same data.
_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
try:
    _CLIENT = chromadb.PersistentClient(path=str(PERSIST_DIR))
    print(f'Connected to persistent Chroma at {PERSIST_DIR}')
except Exception as exc:
    print('PersistentClient failed; falling back to in-memory client:', exc)
    _CLIENT = chromadb.Client()

try:
    _COLLECTION = _CLIENT.get_collection(name=COLLECTION_NAME)
except Exception:
    try:
        _COLLECTION = _CLIENT.create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    except Exception as exc:
        print('create_collection failed:', exc)
        _COLLECTION = None

# Build an in-memory fallback by preferring a persisted fallback.json (written by ingest.py),
# or else by loading local docs and embedding them.
_FALLBACK_DOCS = None
try:
    fallback_path = PERSIST_DIR / 'fallback.json'
    if fallback_path.exists():
        with open(fallback_path, 'r', encoding='utf-8') as fh:
            data = json.load(fh)
        _FALLBACK_DOCS = []
        for item in data:
            emb = item.get('embedding')
            _FALLBACK_DOCS.append({"id": item.get('id'), "text": item.get('text'), "embedding": emb})
        print(f"Loaded {_FALLBACK_DOCS.__len__()} fallback docs from {fallback_path}")
    else:
        docs_dir = BASE_DIR / "docs"
        _FALLBACK_DOCS = []
        for p in sorted(docs_dir.glob('doc_*.txt')):
            text = p.read_text(encoding='utf-8')
            try:
                emb = _MODEL.encode([text], show_progress_bar=False)[0].tolist()
            except Exception:
                emb = None
            _FALLBACK_DOCS.append({"id": p.stem, "text": text, "embedding": emb})
        if _FALLBACK_DOCS:
            print(f"Built {_FALLBACK_DOCS.__len__()} fallback docs by embedding local files")
except Exception as exc:
    print('Fallback in-memory doc load failed:', exc)
    _FALLBACK_DOCS = None


def _validate_intent(candidate: str) -> bool:
    """Return True if candidate intent is one of the allowed intents."""
    return candidate in ("policy_question", "general_question")


def _simulate_groq_response(query: str, attempt: int = 0) -> str:
    """Simulate a Groq LLM response for testing when GROQ_SIMULATE=1.

    This intentionally mirrors the mock keyword rule but can be expanded to
    model occasional malformed outputs to test the retry loop.
    """
    q = query.lower()
    # On the first attempt, produce a clean answer; on attempt==1, optionally
    # produce a malformed output to test the retry logic.
    if attempt == 1:
        # malformed example
        return "I am not sure"
    return "policy_question" if any(k in q for k in _POLICY_KEYWORDS) else "general_question"


def _call_real_llm_for_intent(query: str, attempt: int = 0) -> str:
    """Call an external LLM (Groq) to classify intent.

    Behavior:
    - If GROQ_SIMULATE=1, returns a simulated response (safe for offline/test).
    - If GROQ_API_KEY is present, this is the place to implement a real call.
    - Otherwise raises a NotImplementedError to indicate a real integration is needed.
    """
    if os.getenv("GROQ_SIMULATE", "0") == "1":
        return _simulate_groq_response(query, attempt)

    # Placeholder for real Groq integration. Not implemented here to avoid
    # using external paid services or requiring an API key.
    if os.getenv("GROQ_API_KEY"):
        # Real integration would go here. For now raise to indicate
        # the code path is reachable but requires a proper implementation.
        raise NotImplementedError("Groq integration not implemented in this environment")

    raise NotImplementedError("No Groq API key set; set GROQ_SIMULATE=1 to test the real-LLM branch locally")


def classify_intent(state: State) -> State:
    """Classify the intent as 'policy_question' or 'general_question'.

    - Mock path (default when MOCK_LLM is true): rule-based keyword check.
    - Real-LLM path (when MOCK_LLM is false): call out to a Groq LLM, validate
      the response, and retry up to 2 times with a corrective instruction if the
      LLM returns an unexpected format.
    """
    q = state.get("query", "").strip()
    if MOCK_LLM:
        _q = q.lower()
        state["intent"] = "policy_question" if any(k in _q for k in _POLICY_KEYWORDS) else "general_question"
        return state

    # Real-LLM branch with validation + retry
    max_attempts = 3  # initial + 2 retries
    last_error = None
    for attempt in range(max_attempts):
        try:
            resp = _call_real_llm_for_intent(q, attempt=attempt)
            resp = (resp or "").strip()
            if _validate_intent(resp):
                state["intent"] = resp
                return state
            # If invalid, prepare corrective instruction on next iteration
            last_error = f"Invalid intent format: {resp}"
        except Exception as exc:
            last_error = str(exc)
            # If the call failed and we're on the last attempt, we'll break and fallback
            if attempt == max_attempts - 1:
                break
            # otherwise continue to retry
    # Fallback to robust rule if LLM fails or returns invalid output
    _q = q.lower()
    state["intent"] = "policy_question" if any(k in _q for k in _POLICY_KEYWORDS) else "general_question"
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
    q_emb = _MODEL.encode([query], show_progress_bar=False)[0]

    # If a real chroma collection exists, use it
    if _COLLECTION is not None:
        try:
            # Use a conservative include list; some chroma versions reject 'ids' as an include
            result = _COLLECTION.query(query_embeddings=[q_emb.tolist()], n_results=3, include=["documents", "metadatas"]) 
        except Exception as exc:
            # If chroma query fails, fall back to the in-memory docs if available
            if _FALLBACK_DOCS:
                # use in-memory fallback
                import numpy as np
                qv = np.array(q_emb)
                sims = []
                for d in _FALLBACK_DOCS:
                    dv = np.array(d['embedding'])
                    denom = (np.linalg.norm(qv) * np.linalg.norm(dv))
                    sim = float(np.dot(qv, dv) / denom) if denom != 0 else 0.0
                    sims.append((sim, d))
                sims.sort(key=lambda x: x[0], reverse=True)
                topk = sims[:3]
                docs = [d['text'] for _, d in topk]
                ids = [d['id'] for _, d in topk]
            else:
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

        # If chroma returned no documents but we have a fallback, use it
        if (not docs) and _FALLBACK_DOCS:
            import numpy as np
            qv = np.array(q_emb)
            sims = []
            for d in _FALLBACK_DOCS:
                dv = np.array(d['embedding'])
                denom = (np.linalg.norm(qv) * np.linalg.norm(dv))
                sim = float(np.dot(qv, dv) / denom) if denom != 0 else 0.0
                sims.append((sim, d))
            sims.sort(key=lambda x: x[0], reverse=True)
            try:
                top_debug = [(round(float(s),4), dd['id']) for s, dd in sims[:5]]
                print('Chroma returned empty; fallback retrieval sims (top):', top_debug)
            except Exception:
                pass
            topk = sims[:3]
            docs = [d['text'] for _,d in topk]
            ids = [d['id'] for _,d in topk]
    else:
        # Use in-memory fallback retrieval based on cosine similarity with precomputed embeddings
        if not _FALLBACK_DOCS:
            state["answer"] = "No knowledge collection is available. Please run ingest.py first."
            state["sources"] = []
            state["confidence"] = 0.0
            return state
        import numpy as np
        qv = np.array(q_emb)
        sims = []
        for d in _FALLBACK_DOCS:
            dv = np.array(d['embedding'])
            # cosine similarity
            denom = (np.linalg.norm(qv) * np.linalg.norm(dv))
            sim = float(np.dot(qv, dv) / denom) if denom != 0 else 0.0
            sims.append((sim, d))
        sims.sort(key=lambda x: x[0], reverse=True)
        # Log top similarities for debugging
        try:
            top_debug = [(round(float(s),4), dd['id']) for s, dd in sims[:5]]
            print('Retrieval sims (top):', top_debug)
        except Exception:
            pass
        topk = sims[:3]
        docs = [d['text'] for _,d in topk]
        ids = [d['id'] for _,d in topk]

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
