from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str
    top_k: int = 20
    filters: Optional[Dict[str, Any]] = None
    with_rerank: bool = False


class Chunk(BaseModel):
    text: str
    score: float
    source: str
    title: Optional[str] = None
    section: Optional[str] = None
    anchor: Optional[str] = None
    updated_at: Optional[str] = None
    tags: Optional[Dict[str, Any]] = None


class SearchResponse(BaseModel):
    results: List[Chunk]


class AnswerRequest(BaseModel):
    query: str
    top_k: int = 24
    max_context_tokens: int = 3500
    with_rerank: bool = True
    filters: Optional[Dict[str, Any]] = None


class Citation(BaseModel):
    title: Optional[str] = None
    source: str
    anchor: Optional[str] = None


class AnswerResponse(BaseModel):
    answer: str
    citations: List[Citation]
    related: List[Citation] = Field(default_factory=list)
    used_chunks: List[Chunk]
    meta: Dict[str, Any] = Field(default_factory=dict)


# Async job models
class AnswerAsyncStartResponse(BaseModel):
    job_id: str


class AnswerJobStatus(BaseModel):
    status: str  # pending | running | done | error
    result: Optional[AnswerResponse] = None
    error: Optional[str] = None 