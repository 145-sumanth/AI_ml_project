# Support Assistant

This module ingests a small set of policy documents (support_assistant/docs/doc_01.txt .. doc_08.txt), embeds them with a sentence-transformers model, stores embeddings (ChromaDB when available) or a fallback JSON, and exposes a FastAPI POST /ask endpoint that returns answers with sources and confidence.

Architecture (detailed)

- Ingestion (file: support_assistant/ingest.py)
  - Reads each document from support_assistant/docs/*.txt (one chunk per doc in this project).
  - Loads the sentence-transformers/all-MiniLM-L6-v2 model and computes 384-d embeddings.
  - Attempts to create a ChromaDB collection (metadata {"hnsw:space":"cosine"}) and upsert the embeddings.
  - For environments where Chroma persistence or binary indexes are not available, ingest.py writes a fallback JSON at support_assistant/chroma_db/fallback.json containing ids, texts and embeddings so the API can retrieve by cosine similarity without Chroma.

- Retrieval (file: support_assistant/graph.py -> retrieve_and_answer)
  - Primary path: query ChromaDB collection for top-k (k=3) similar chunks using cosine similarity.
  - Fallback path: when Chroma is not usable or returns empty, load support_assistant/chroma_db/fallback.json and compute cosine similarity locally with numpy; return top-k ids and texts.
  - Retrieval returns the top chunks (text and ids) for use by the generator or for constructing the mock response.

- Intent classification and generation (file: support_assistant/graph.py)
  - classify_intent: a lightweight keyword-based classifier (mock) that decides whether a query is a policy/support question (e.g., contains words like "delivery", "refund", "return", "tracking", "membership", "gift card", "cancel", "support hours") or a general question.
  - If classified as a policy question, the retrieve_and_answer node runs; the generator node (mock or real LLM) composes an answer using the retrieved context.
  - If classified as a general/non-policy question, the direct_answer node returns a canned message: "I can only answer questions about Zepto policies right now." with empty sources.

- API (file: support_assistant/main.py)
  - FastAPI app exposing POST /ask with Pydantic request/response models (support_assistant/schemas.py).
  - Routes validate JSON and call the graph StateGraph to run classify -> retrieve -> answer nodes.

MOCK_LLM branching

- The behavior is gated by the environment variable MOCK_LLM (default is the string "1" which evaluates to True in the code as written).
- MOCK_LLM=1 (default):
  - classify_intent runs the mock keyword classifier.
  - retrieve_and_answer runs the real retrieval (Chroma or fallback) and returns a mock answer of the form:
    "Based on the retrieved context: {first ~200 chars of top chunk}"
    sources = retrieved ids, confidence = 1.0
  - direct_answer returns the canned message with empty sources.
- MOCK_LLM=0:
  - Code paths exist for a real-LLM branch (placeholder shows Groq integration). When MOCK_LLM=0 the graph will invoke the LLM code paths instead of mock responses. These LLM branches include a validate-and-retry loop (up to 2 retries) to ensure the LLM output matches the expected schema and constraints.

Note: The default MOCK_LLM value and response schema are unchanged by this README — code behavior remains the same unless you explicitly set MOCK_LLM=0.

Quick HTTP checks performed (raw JSON outputs from local run)

1) POST {"query": "What is the delivery fee?"}

Raw JSON response (retrieval path, returned sources include doc_01):

{"answer":"Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard del","sources":["doc_01","doc_05","doc_02"],"confidence":1.0}

2) POST {"query": "How do refunds work?"}

Raw JSON response (retrieval path, returned sources include doc_02):

{"answer":"Based on the retrieved context: Grocery and perishable items may be reported for a return within 24 hours of delivery if damaged, spoiled, or incorrect; non-perishable packaged items may be returned within 7 days of delivery in unop","sources":["doc_02","doc_06","doc_05"],"confidence":1.0}

3) POST {"query": "What is the capital of France?"}

Raw JSON response (direct/canned path, no retrieval):

{"answer":"I can only answer questions about Zepto policies right now.","sources":[],"confidence":1.0}

HTTP method validation

- GET /ask returns 405 Method Not Allowed with body {"detail":"Method Not Allowed"} — this is expected because /ask accepts POST only.
- POST /ask returns 200 and the JSON schema shown above.

Confirming /docs

- GET /docs returns the OpenAPI UI (Swagger UI) and can be used to "Try it out" — the route /docs loads on the running server and the interactive POST works (same POST behavior as curl shown above).

Docker build & run (exact commands)

# Build (top-level Dockerfile if present)
# If you want the lightweight image (recommended):
cd /path/to/AI_ml_project
# Build the lightweight image (uses support_assistant/Dockerfile.light):
docker build -f support_assistant/Dockerfile.light -t zepto-assistant:light support_assistant

# Or build the main image (if a top-level Dockerfile exists):
docker build -t zepto-assistant .

# Run the container (first run: optionally ingest at startup with RUN_INGEST=1 and mount a host folder to persist index):
mkdir -p support_assistant/chroma_db
docker run --rm -p 7860:7860 \
  -v $(pwd)/support_assistant/chroma_db:/app/support_assistant/chroma_db \
  -e RUN_INGEST=1 \
  zepto-assistant:light

# Subsequent runs (index persists):
docker run --rm -p 7860:7860 \
  -v $(pwd)/support_assistant/chroma_db:/app/support_assistant/chroma_db \
  zepto-assistant:light

Notes and troubleshooting

- The project includes a fallback JSON (support_assistant/chroma_db/fallback.json) so retrieval works even without a Chroma binary index. This makes the API reliably testable in minimal environments.
- If you plan to use a real LLM (MOCK_LLM=0), ensure credentials/config for that LLM are available and that any rate/usage limits are acceptable.

If you'd like, I can:
- Commit this updated README and push it to origin/main.
- Run the Docker build here and attach the build log and the curl output against the running container (I can do that now if you confirm — note: image build/download may be large and take time).
