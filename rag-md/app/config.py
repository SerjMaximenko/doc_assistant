from __future__ import annotations

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    log_level: str = "INFO"

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "api_docs"

    # Embeddings
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "cpu"

    # Reranker
    reranker_model: str | None = "BAAI/bge-reranker-large"
    reranker_device: str = "cpu"
    enable_rerank: bool = True

    # LLM (OpenAI-like first)
    openai_base_url: str | None = None
    openai_api_key: str | None = None
    openai_model: str | None = "mistralai/Mistral-7B-Instruct-v0.2"

    # llama.cpp server
    llama_base_url: str | None = None
    llama_model: str | None = "mistral"

    # General
    default_language: str = "ru"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings() 