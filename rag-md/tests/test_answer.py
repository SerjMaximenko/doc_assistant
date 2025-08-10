from __future__ import annotations

import asyncio
from typing import Any, Dict, List

import pytest

from app.main import post_answer
from app.models import AnswerRequest


class DummyLLM:
    async def acomplete(self, messages: List[Dict[str, Any]], *, temperature: float = 0.2, max_tokens: int = 512) -> str:
        return "dummy answer"


@pytest.mark.asyncio
async def test_answer_no_context(monkeypatch: pytest.MonkeyPatch) -> None:
    from app import main as main_mod

    # Monkeypatch search to return empty
    monkeypatch.setattr(main_mod, "search", lambda **kwargs: [])
    # Monkeypatch get_llm
    monkeypatch.setattr(main_mod, "get_llm", lambda: DummyLLM())

    req = AnswerRequest(query="What is X?", with_rerank=True)
    resp = await post_answer(req)
    assert "Недостаточно" in resp.answer 