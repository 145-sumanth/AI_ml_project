"""FastAPI application exposing a POST /ask endpoint for the support assistant.

- Validates requests with Pydantic models from schemas.py
- Runs the small decision graph in graph.py
- Returns an AskResponse
"""

from __future__ import annotations

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from support_assistant.schemas import AskRequest, AskResponse
from support_assistant.graph import run_graph

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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=True)
