from __future__ import annotations

from pathlib import Path

from app.md_loader import parse_markdown
from app.chunking import chunk_sections


def test_parse_and_chunk(tmp_path: Path) -> None:
    md = tmp_path / "doc.md"
    md.write_text("""---\nservice: stock\napi_version: v1\n---\n# Title\n\n## Section A\nText A\n\n```json\n{"a": 1}\n```\n\n| col | val |\n| --- | --- |\n| x | 1 |\n\n## Section B\nText B\n""")

    sections, fm = parse_markdown(md)
    assert any(s.section == "Section A" for s in sections)
    chunks = chunk_sections(sections, target_tokens_min=10, target_tokens_max=50, overlap_tokens=5)
    assert len(chunks) >= 1
    # ensure metadata propagated
    content, meta = chunks[0]
    assert meta["source"].endswith("doc.md")
    assert meta.get("service") == "stock" 