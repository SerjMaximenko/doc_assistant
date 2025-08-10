from __future__ import annotations

import re
from typing import Dict, Iterable, List, Tuple

from .md_loader import MDSection
from .utils import count_tokens, trim_text_tokens


_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def chunk_sections(
    sections: Iterable[MDSection], *, target_tokens_min: int = 500, target_tokens_max: int = 800,
    overlap_tokens: int = 100,
) -> List[Tuple[str, Dict[str, str]]]:
    chunks: List[Tuple[str, Dict[str, str]]] = []
    for sec in sections:
        text = sec.text
        paragraphs = _split_preserving_blocks(text)
        current: List[str] = []
        current_tokens = 0

        def push_chunk(*, keep_overlap: bool = True) -> None:
            nonlocal current, current_tokens
            if not current:
                return
            content = "\n\n".join(current).strip()
            if count_tokens(content) > target_tokens_max:
                content = trim_text_tokens(content, target_tokens_max)
            if not content:
                current = []
                current_tokens = 0
                return
            meta = {
                "source": sec.source,
                "title": sec.title or "",
                "section": sec.section or "",
                "anchor": sec.anchor or "",
                **sec.meta,
            }
            chunks.append((content, meta))

            if keep_overlap and overlap_tokens > 0:
                back, t = [], 0
                for para in reversed(current):
                    n = count_tokens(para)
                    if t + n > overlap_tokens:
                        break
                    back.append(para)
                    t += n
                current = list(reversed(back))
                current_tokens = sum(count_tokens(p) for p in current)
            else:
                current = []
                current_tokens = 0


        for para in paragraphs:
            n = count_tokens(para)
            if n >= target_tokens_max:
                if _is_code_or_table_block(para):
                    if current:
                        push_chunk()
                    current = [para]
                    current_tokens = n
                    push_chunk()
                    continue
                pieces = _split_text_to_max_tokens(para, target_tokens_max)
                for piece in pieces:
                    pn = count_tokens(piece)
                    while current and current_tokens + pn > target_tokens_max:
                        push_chunk(keep_overlap=False)
                    current.append(piece)
                    current_tokens += pn
                    if current_tokens >= target_tokens_max:
                        push_chunk()
                continue

            if current_tokens + n <= target_tokens_max:
                current.append(para)
                current_tokens += n
            else:
                if current_tokens < target_tokens_min:
                    current.append(para)
                    current_tokens += n
                    push_chunk()
                else:
                    push_chunk()
                    current = [para]
                    current_tokens = n
        push_chunk()

    return chunks


def _is_code_or_table_block(block: str) -> bool:
    if "```" in block:
        return True
    lines = [ln for ln in block.splitlines() if ln.strip()]
    if not lines:
        return False
    leading_pipes = sum(1 for ln in lines if ln.lstrip().startswith("|"))
    return leading_pipes >= max(2, len(lines) // 2)


def _split_text_to_max_tokens(text: str, max_tokens: int) -> List[str]:
    sentences = _SENT_SPLIT_RE.split(text)
    out: List[str] = []
    buf: List[str] = []
    tokens = 0
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        n = count_tokens(s)
        if n >= max_tokens:
            parts = _hard_wrap_by_tokens(s, max_tokens)
            for p in parts:
                if buf:
                    out.append(" ".join(buf))
                    buf = []
                out.append(p)
            tokens = 0
            continue
        if tokens + n > max_tokens and buf:
            out.append(" ".join(buf))
            buf = [s]
            tokens = n
        else:
            buf.append(s)
            tokens += n
    if buf:
        out.append(" ".join(buf))
    return out


def _hard_wrap_by_tokens(text: str, max_tokens: int) -> List[str]:
    approx_chars = max_tokens * 4
    out = []
    i = 0
    while i < len(text):
        out.append(text[i : i + approx_chars])
        i += approx_chars
    return out


def _split_preserving_blocks(text: str) -> List[str]:
    lines = text.splitlines()
    blocks: List[str] = []
    buf: List[str] = []
    in_code = False
    fence = None
    in_table = False

    def flush() -> None:
        nonlocal buf
        if buf:
            blocks.append("\n".join(buf).strip())
            buf = []

    for line in lines:
        if line.lstrip().startswith("|"):
            in_table = True
        if not line.strip():
            if not in_code and not in_table:
                flush()
                continue
        if line.strip().startswith("```"):
            if not in_code:
                in_code = True
                fence = line.strip()
            elif in_code and line.strip() == fence:
                in_code = False
                fence = None
                buf.append(line)
                flush()
                continue
        buf.append(line)
        if in_table and not line.lstrip().startswith("|"):
            in_table = False
            flush()
    flush()
    return [b for b in blocks if b] 