from __future__ import annotations

from functools import lru_cache
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from sentence_transformers import CrossEncoder, SentenceTransformer

from .config import get_settings
from .llm_client.base import LLMClient
from .llm_client.llama_cpp import LlamaCppClient
from .llm_client.openai_like import OpenAILikeClient


@lru_cache(maxsize=1)
def get_qdrant() -> QdrantClient:
    cfg = get_settings()
    return QdrantClient(url=cfg.qdrant_url, api_key=cfg.qdrant_api_key)


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    cfg = get_settings()
    model = SentenceTransformer(cfg.embedding_model, device=cfg.embedding_device)
    # important for bge-m3: normalize embeddings on encode
    return model


@lru_cache(maxsize=1)
def get_reranker() -> Optional[CrossEncoder]:
    cfg = get_settings()
    if not cfg.enable_rerank or not cfg.reranker_model:
        return None
    return CrossEncoder(cfg.reranker_model, device=cfg.reranker_device)


@lru_cache(maxsize=1)
def get_llm() -> LLMClient:
    cfg = get_settings()
    if cfg.openai_base_url and cfg.openai_api_key:
        return OpenAILikeClient(cfg.openai_base_url, cfg.openai_api_key, cfg.openai_model or "")
    if cfg.llama_base_url:
        return LlamaCppClient(cfg.llama_base_url, cfg.llama_model or "mistral")
    raise RuntimeError("No LLM backend configured. Set OPENAI_* or LLAMA_* env vars.")


def ensure_collection(collection: str, vector_size: int = 1024) -> None:
    client = get_qdrant()
    collections = client.get_collections().collections
    if any(c.name == collection for c in collections):
        return
    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    ) 