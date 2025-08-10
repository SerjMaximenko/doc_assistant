from __future__ import annotations

from typing import List
import asyncio
import httpx

from .base import ChatMessage, LLMClient


class LlamaCppClient(LLMClient):
    def __init__(self, base_url: str, model: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        # Increase timeouts for large local models; allow long generation
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(6000.0, connect=300.0))

    async def acomplete(self, messages: List[ChatMessage], *, temperature: float = 0.2,
                        max_tokens: int = 512) -> str:
        # Try OpenAI-compatible path first
        url = f"{self._base_url}/v1/chat/completions"
        payload = {
            "model": self._model,
            "messages": messages,  # type: ignore[arg-type]
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        try:
            r = await self._client.post(url, json=payload)
            if r.status_code == 200:
                data = r.json()
                return data["choices"][0]["message"].get("content", "")
        except Exception:
            # Fall through to native endpoint on any client error/timeout
            pass

        # Fallback to llama.cpp native /completion
        url2 = f"{self._base_url}/completion"
        prompt = "\n".join([m["content"] for m in messages if m["role"] in {"system", "user"}])
        r2 = await self._client.post(url2, json={
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": temperature,
        })
        r2.raise_for_status()
        data2 = r2.json()
        return data2.get("content", data2.get("completion", "")) 