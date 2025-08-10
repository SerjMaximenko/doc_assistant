from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional


class ChatMessage(dict):
    @staticmethod
    def system(content: str) -> "ChatMessage":
        return {"role": "system", "content": content}

    @staticmethod
    def user(content: str) -> "ChatMessage":
        return {"role": "user", "content": content}


class LLMClient(ABC):
    @abstractmethod
    async def acomplete(self, messages: List[ChatMessage], *, temperature: float = 0.2,
                        max_tokens: int = 512) -> str:
        raise NotImplementedError 