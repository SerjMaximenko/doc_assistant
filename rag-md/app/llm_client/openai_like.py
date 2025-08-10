from __future__ import annotations

from typing import List
from openai import AsyncOpenAI

from .base import ChatMessage, LLMClient


class OpenAILikeClient(LLMClient):
    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self._model = model

    async def acomplete(self, messages: List[ChatMessage], *, temperature: float = 0.2,
                        max_tokens: int = 512) -> str:
        resp = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or "" 