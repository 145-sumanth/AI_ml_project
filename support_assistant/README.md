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
1. Create and activate venv (recommended):
   python3.11 -m venv .venv
   source .venv/bin/activate
2. Install dependencies: pip install -r support_assistant/requirements.txt
3. Ingest docs (builds an in-memory index or persistent collection if supported):
   python support_assistant/ingest.py
4. Start the API: uvicorn support_assistant.main:app --host 0.0.0.0 --port 7860
5. Try the curl examples above.

Docker (lightweight) quick guide (recommended):
1. Build the lightweight image (defers model download to runtime):
   cd /path/to/AI_ml_project
   docker build -f support_assistant/Dockerfile.light -t zepto-assistant:light support_assistant

2. Create host folder to persist the index and run ingest at first start:
   mkdir -p support_assistant/chroma_db
   docker run --rm -it -p 7860:7860 \
     -v $(pwd)/support_assistant/chroma_db:/app/support_assistant/chroma_db \
     -e RUN_INGEST=1 \
     zepto-assistant:light

3. Subsequent runs (index persists):
   docker run --rm -it -p 7860:7860 \
     -v $(pwd)/support_assistant/chroma_db:/app/support_assistant/chroma_db \
     zepto-assistant:light

Notes:
- If Hugging Face rate limits model downloads, set HF_TOKEN in the docker run environment: -e HF_TOKEN=your_token
- Docker run with RUN_INGEST=1 downloads the embedding model and builds the index at runtime; use the mounted chroma_db to avoid repeating the download on subsequent runs.
