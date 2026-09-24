# Support Assistant

This module ingests a small set of policy documents, embeds them with a
sentence-transformers model, stores embeddings in ChromaDB, and exposes a
FastAPI POST /ask endpoint that returns answers with sources and confidence.

Architecture (high level):
- Ingestion: support_assistant/ingest.py — reads docs/*, embeds with all-MiniLM-L6-v2, creates ChromaDB collection `zepto_docs` with metadata {"hnsw:space": "cosine"}.
- Retrieval: support_assistant/graph.py -> retrieve_and_answer uses the ChromaDB collection to find top-3 chunks by cosine.
- Generation/Policy branching: support_assistant/graph.py -> classify_intent decides whether to run retrieval or return a canned message. This branch is controlled by environment variable MOCK_LLM (default "1"). If MOCK_LLM=0 the code contains placeholders where a real LLM (Groq) would be called.
- API: support_assistant/main.py — FastAPI app exposing POST /ask using Pydantic models from schemas.py.

Example curl calls (mock responses shown):

1) Policy question (retrieval path)

curl -X POST http://localhost:7860/ask -H "Content-Type: application/json" -d '{"query": "How long does delivery take?"}'

Response (example):
{
  "answer": "Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation...",
  "sources": ["doc_01"],
  "confidence": 1.0
}

2) Non-policy/general question (direct answer path)

curl -X POST http://localhost:7860/ask -H "Content-Type: application/json" -d '{"query": "Tell me about your product catalog."}'

Response (example):
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 0.0
}

Running locally (quick verification):
1. Install dependencies: python3 -m pip install -r requirements.txt
2. Ingest docs: python3 support_assistant/ingest.py
3. Start the API: uvicorn support_assistant.main:app --host 0.0.0.0 --port 7860
4. Try the curl examples above.
