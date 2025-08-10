from __future__ import annotations

import os

import pytest
from qdrant_client import QdrantClient

from app.retriever import embed_query


@pytest.mark.integration
def test_qdrant_available() -> None:
    url = os.getenv("QDRANT_URL", "http://localhost:6333")
    client = QdrantClient(url=url)
    try:
        _ = client.get_collections()
    except Exception as e:
        pytest.skip(f"Qdrant not available: {e}")


@pytest.mark.integration
def test_embed_query() -> None:
    vec = embed_query("test query")
    assert isinstance(vec, list)
    assert len(vec) == 1024 