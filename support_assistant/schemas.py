"""Pydantic schemas for the support assistant API.

Request: {query: str}
Response: {answer: str, sources: list[str], confidence: float}
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from typing import List


class AskRequest(BaseModel):
    query: str = Field(..., description="User's natural language query")


class AskResponse(BaseModel):
    answer: str = Field(..., description="The assistant's answer text")
    sources: List[str] = Field(default_factory=list, description="List of source ids or filenames used to derive the answer")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Assistant's confidence in the answer (0-1)")
