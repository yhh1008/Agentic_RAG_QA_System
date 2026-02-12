from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.agent.graph import run_agent


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1)
    session_id: str | None = None


class Citation(BaseModel):
    doc_id: str
    chunk_id: str
    source: str
    quote: str


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    trace_id: str
    attempts: int
    used_tools: list[str]


app = FastAPI(title="Agentic RAG QA System", version="0.1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    result = run_agent(req.query, req.session_id)
    return AskResponse(**result)
