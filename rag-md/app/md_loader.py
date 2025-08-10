from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import frontmatter
from markdown_it import MarkdownIt
from markdown_it.token import Token


@dataclass
class MDSection:
    source: str
    title: Optional[str]
    section: Optional[str]
    anchor: Optional[str]
    text: str
    meta: Dict[str, str]


def _slugify(text: str) -> str:
    s = text.strip().lower().replace(" ", "-")
    return "#" + "".join(ch for ch in s if ch.isalnum() or ch in {"-", "_", "#"})


def parse_markdown(file_path: Path) -> Tuple[List[MDSection], Dict[str, str]]:
    post = frontmatter.load(file_path)
    raw = post.content
    fm = post.metadata or {}

    md = MarkdownIt("commonmark", options_update={"linkify": True, "typographer": False})
    tokens = md.parse(raw)

    sections: List[MDSection] = []
    current_title: Optional[str] = None
    current_section: Optional[str] = None
    current_anchor: Optional[str] = None
    buf: List[str] = []

    def flush() -> None:
        nonlocal buf, current_section, current_anchor
        if not buf:
            return
        text = "".join(buf).strip()
        if text:
            sections.append(
                MDSection(
                    source=str(file_path.as_posix()),
                    title=current_title,
                    section=current_section,
                    anchor=current_anchor,
                    text=text,
                    meta={k: str(v) for k, v in fm.items()},
                )
            )
        buf = []

    # Build content grouped by h1-h4
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t.type == "heading_open" and t.tag in {"h1", "h2", "h3", "h4"}:
            # when a new header starts, flush previous buffer
            flush()
            # next token is inline with content
            content_token = tokens[i + 1]
            heading_text = content_token.content.strip()
            if t.tag == "h1" and current_title is None:
                current_title = heading_text
            current_section = heading_text
            current_anchor = _slugify(heading_text)
            i += 3
            continue
        # Preserve code blocks and tables by copying token content verbatim
        if t.type in {"fence", "code_block"}:
            buf.append(t.content)
            i += 1
            continue
        if t.type == "inline":
            buf.append(t.content + "\n")
        i += 1

    flush()

    # add updated_at
    updated = dt.datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
    for s in sections:
        s.meta.setdefault("updated_at", updated)

    return sections, {k: str(v) for k, v in fm.items()}


def load_md_folder(folder: Path) -> Iterable[Tuple[Path, List[MDSection], Dict[str, str]]]:
    for p in folder.rglob("*.md"):
        sections, fm = parse_markdown(p)
        yield p, sections, fm 