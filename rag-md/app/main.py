from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException

from .config import get_settings
from .deps import get_llm
from .models import AnswerRequest, AnswerResponse, Citation, SearchRequest, SearchResponse, AnswerAsyncStartResponse, AnswerJobStatus
from .prompts import ANSWER_TEMPLATE, SYSTEM_PROMPT
from .retriever import search
from .utils import all_json_fences_valid, trim_context

app = FastAPI(title="RAG over Markdown")

# In-memory async jobs store
_JOBS: Dict[str, Dict[str, Any]] = {}


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/search", response_model=SearchResponse)
async def post_search(req: SearchRequest) -> SearchResponse:
    results = search(
        query=req.query,
        top_k=req.top_k,
        filters=req.filters,
        with_rerank=req.with_rerank,
        collection=get_settings().qdrant_collection,
    )
    return SearchResponse(results=results)  # type: ignore[arg-type]


async def _build_answer(req: AnswerRequest) -> AnswerResponse:
    t0 = time.time()
    settings = get_settings()
    results = search(
        query=req.query,
        top_k=req.top_k,
        filters=req.filters,
        with_rerank=req.with_rerank,
        collection=settings.qdrant_collection,
    )
    # basic no-answer policy: if empty or low scores
    if not results:
        return AnswerResponse(
            answer=("Недостаточно контекста для ответа. Попробуйте переформулировать запрос. "
                    "См. также связанные разделы ниже."),
            citations=[],
            related=[],
            used_chunks=[],
            meta={"latency_ms": int((time.time() - t0) * 1000)},
        )

    # Simple relevance threshold: adapt if rerank is enabled (CrossEncoder scores are usually higher)
    top_score = results[0]["score"]
    threshold = 0.2 if not req.with_rerank else 0.05
    if top_score < threshold:
        related = [
            Citation(title=r.get("section") or r.get("title"), source=r["source"], anchor=r.get("anchor"))
            for r in results[:3]
        ]
        return AnswerResponse(
            answer=("Не нашёл достаточно релевантный контекст для ответа. Попробуйте уточнить запрос."),
            citations=[],
            related=related,
            used_chunks=[],
            meta={"latency_ms": int((time.time() - t0) * 1000)},
        )

    # Build context: dedup by source + anchor, keep top 6-8 after rerank
    top_chunks = results[:8] if req.with_rerank else results[:6]
    seen: set[tuple[str, str | None]] = set()
    context_parts: List[str] = []
    citations: List[Citation] = []
    used_chunks: List[Dict[str, Any]] = []
    for r in top_chunks:
        key = (r["source"], r.get("anchor"))
        if key in seen:
            continue
        seen.add(key)
        title = r.get("section") or r.get("title") or ""
        header = f"## {title}" if title else ""
        context_parts.append("\n\n".join([header, r["text"]]).strip())
        citations.append(Citation(title=title or None, source=r["source"], anchor=r.get("anchor")))
        used_chunks.append(r)

    context, used_tokens = trim_context(context_parts, max_tokens=req.max_context_tokens)

    # If context is too small, return a graceful no-answer
    if used_tokens < 50:
        related = [Citation(title=r.get("section") or r.get("title"), source=r["source"], anchor=r.get("anchor")) for r in results[:3]]
        return AnswerResponse(
            answer=("Недостаточно релевантного контекста для уверенного ответа. "
                    "Попробуйте уточнить вопрос."),
            citations=[],
            related=related,
            used_chunks=[],
            meta={"latency_ms": int((time.time() - t0) * 1000)},
        )

    llm = get_llm()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": ANSWER_TEMPLATE.format(context=context, question=req.query)},
    ]

    answer_text = await llm.acomplete(messages, temperature=0.2, max_tokens=800)

    # Validate JSON fences if any
    if not all_json_fences_valid(answer_text):
        answer_text += "\n\nПримечание: Обнаружен невалидный JSON во фрагменте ответа; исправьте синтаксис."

    # Ensure we do not cite sources not used
    used_sources = {(c.source, c.anchor) for c in citations}
    related = []
    for r in results[8:12]:
        k = (r["source"], r.get("anchor"))
        if k in used_sources:
            continue
        related.append(Citation(title=r.get("section") or r.get("title"), source=r["source"], anchor=r.get("anchor")))

    out = AnswerResponse(
        answer=answer_text.strip(),
        citations=citations,
        related=related,
        used_chunks=used_chunks,  # type: ignore[arg-type]
        meta={"latency_ms": int((time.time() - t0) * 1000)},
    )
    return out


@app.post("/answer", response_model=AnswerResponse)
async def post_answer(req: AnswerRequest) -> AnswerResponse:
    return await _build_answer(req)


@app.post("/answer_async/start", response_model=AnswerAsyncStartResponse)
async def post_answer_async_start(req: AnswerRequest) -> AnswerAsyncStartResponse:
    job_id = str(uuid.uuid4())
    _JOBS[job_id] = {"status": "pending", "result": None, "error": None}

    async def _runner() -> None:
        _JOBS[job_id]["status"] = "running"
        try:
            result = await _build_answer(req)
            _JOBS[job_id]["result"] = result
            _JOBS[job_id]["status"] = "done"
        except Exception as e:  # noqa: BLE001
            _JOBS[job_id]["status"] = "error"
            _JOBS[job_id]["error"] = str(e)

    asyncio.create_task(_runner())
    return AnswerAsyncStartResponse(job_id=job_id)


@app.get("/answer_async/status/{job_id}", response_model=AnswerJobStatus)
async def get_answer_async_status(job_id: str) -> AnswerJobStatus:
    data = _JOBS.get(job_id)
    if not data:
        raise HTTPException(status_code=404, detail="job not found")
    return AnswerJobStatus(status=data["status"], result=data.get("result"), error=data.get("error")) 