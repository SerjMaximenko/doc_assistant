from __future__ import annotations

import json
import re
from typing import Iterable, List, Tuple

import tiktoken


def get_tokenizer() -> tiktoken.Encoding:
    # Approximate tokenizer for Mistral (cl100k_base)
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    enc = get_tokenizer()
    return len(enc.encode(text))


def trim_text_tokens(text: str, max_tokens: int) -> str:
    enc = get_tokenizer()
    ids = enc.encode(text)
    if len(ids) <= max_tokens:
        return text
    trimmed = ids[:max_tokens]
    return enc.decode(trimmed)


def trim_context(chunks: Iterable[str], max_tokens: int) -> Tuple[str, int]:
    total = 0
    selected: List[str] = []
    for c in chunks:
        n = count_tokens(c)
        if total + n > max_tokens:
            break
        selected.append(c)
        total += n
    return "\n\n".join(selected), total


def is_valid_json(text: str) -> bool:
    try:
        json.loads(text)
        return True
    except Exception:
        return False


_JSON_FENCE_RE = re.compile(r"```json\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json_fences(text: str) -> List[str]:
    return [m.group(1).strip() for m in _JSON_FENCE_RE.finditer(text)]


def all_json_fences_valid(text: str) -> bool:
    blocks = extract_json_fences(text)
    if not blocks:
        return True
    return all(is_valid_json(b) for b in blocks) 