"""FastAPI application exposing a POST /ask endpoint for the support assistant.

- Validates requests with Pydantic models from schemas.py
- Runs the small decision graph in graph.py
- Returns an AskResponse
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

try:
    from support_assistant.schemas import AskRequest, AskResponse
    from support_assistant.graph import run_graph
except ImportError:  # pragma: no cover - supports running from the module directory
    from schemas import AskRequest, AskResponse
    from graph import run_graph

app = FastAPI(title="Support Assistant (Zepto policies)")

# Allow all origins for local testing — adjust in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    """Accept a query and return an answer with sources and a confidence score."""
    try:
        state = run_graph(req.query)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    answer = state.get("answer", "")
    sources = state.get("sources", []) or []
    confidence = float(state.get("confidence", 0.0) or 0.0)

    return AskResponse(answer=answer, sources=sources, confidence=confidence)


@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse("""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8" />
      <title>Zepto Support Assistant</title>
      <style>
        body { font-family: Arial, sans-serif; margin: 2rem; background: #f5f5f5; }
        .box { max-width: 700px; margin: auto; background: white; padding: 2rem; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,.08); }
        textarea { width: 100%; min-height: 100px; font-size: 1rem; }
        button { padding: 0.7rem 1.2rem; font-size: 1rem; margin-top: 0.5rem; }
        #result { margin-top: 1rem; white-space: pre-wrap; }
      </style>
    </head>
    <body>
      <div class="box">
        <h2>Zepto Support Assistant</h2>
        <textarea id="query" placeholder="Ask a policy question...">What is the delivery fee?</textarea>
        <button id="askBtn">Ask</button>
        <div id="result"></div>
      </div>
      <script>
        document.getElementById('askBtn').addEventListener('click', async () => {
          const q = document.getElementById('query').value.trim();
          const result = document.getElementById('result');
          result.textContent = 'Loading...';
          try {
            const response = await fetch('/ask', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ query: q })
            });
            const data = await response.json();
            if (!response.ok) {
              throw new Error(data.detail || 'Request failed');
            }
            result.textContent = JSON.stringify({
              answer: data.answer,
              sources: data.sources,
              confidence: data.confidence
            }, null, 2);
          } catch (err) {
            result.textContent = 'Error: ' + err.message;
          }
        });
      </script>
    </body>
    </html>
    """)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=True)
