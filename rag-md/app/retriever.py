from __future__ import annotations

from typing import Any, Dict, List, Optional

from qdrant_client.models import FieldCondition, Filter as QFilter, MatchValue, ScoredPoint

from .deps import get_embedder, get_qdrant, get_reranker


def _to_filter(filters: Optional[Dict[str, Any]]) -> Optional[QFilter]:
    if not filters:
        return None
    conditions = []
    for k, v in filters.items():
        conditions.append(FieldCondition(key=k, match=MatchValue(value=v)))
    return QFilter(must=conditions)


def embed_query(query: str) -> List[float]:
    model = get_embedder()
    # Prefer a query-specific prompt if available; fall back gracefully
    prompts = getattr(model, "prompts", {}) or {}
    prompt_name: Optional[str] = None
    if isinstance(prompts, dict):
        if "query" in prompts:
            prompt_name = "query"
        elif "passage" in prompts:
            prompt_name = "passage"
        elif "document" in prompts:
            prompt_name = "document"
    try:
        if prompt_name:
            emb = model.encode([query], normalize_embeddings=True, prompt_name=prompt_name)[0]
        else:
            raise TypeError
    except TypeError:
        emb = model.encode([query], normalize_embeddings=True)[0]
    return emb.tolist()  # type: ignore[return-value]


def search(
    query: str,
    *,
    top_k: int = 20,
    filters: Optional[Dict[str, Any]] = None,
    with_rerank: bool = False,
    collection: str = "api_docs",
) -> List[Dict[str, Any]]:
    client = get_qdrant()
    vector = embed_query(query)
    flt = _to_filter(filters)
    hits: List[ScoredPoint] = client.search(
        collection_name=collection,
        query_vector=vector,
        limit=top_k,
        query_filter=flt,
        with_payload=True,
    )

    results: List[Dict[str, Any]] = []
    for h in hits:
        payload = h.payload or {}
        results.append(
            {
                "text": payload.get("text", ""),
                "score": float(h.score or 0.0),
                "source": payload.get("source", ""),
                "title": payload.get("title"),
                "section": payload.get("section"),
                "anchor": payload.get("anchor"),
                "updated_at": payload.get("updated_at"),
                "tags": {
                    k: v
                    for k, v in payload.items()
                    if k
                    not in {"text", "source", "title", "section", "anchor", "updated_at"}
                },
            }
        )

    if with_rerank:
        rr = get_reranker()
        if rr is not None and results:
            pairs = [[query, r["text"]] for r in results]
            scores = rr.predict(pairs)  # type: ignore[assignment]
            # Normalize logits to probabilities if needed using sigmoid
            try:
                import math

                def _sigmoid(x: float) -> float:
                    # numerically stable sigmoid
                    if x >= 0:
                        z = math.exp(-x)
                        return 1.0 / (1.0 + z)
                    z = math.exp(x)
                    return z / (1.0 + z)

                norm_scores = [_sigmoid(float(s)) for s in scores]
            except Exception:
                norm_scores = [float(s) for s in scores]
            rescored = [(*r.items(), ("score", float(s))) for r, s in zip(results, norm_scores)]
            results = [dict(items) for items in rescored]
            results.sort(key=lambda x: x["score"], reverse=True)

    return results 