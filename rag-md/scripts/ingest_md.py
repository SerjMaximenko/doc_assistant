from __future__ import annotations

import argparse
import time
import uuid
from pathlib import Path
from typing import Dict, List, Tuple

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from tqdm import tqdm

from app.chunking import chunk_sections
from app.config import get_settings
from app.md_loader import parse_markdown


def ensure_collection(client, name: str, vector_size: int = 1024, recreate: bool = False) -> None:
    existing = [c.name for c in client.get_collections().collections]
    if name in existing and recreate:
        client.delete_collection(name)
    if recreate or name not in existing:
        client.create_collection(name, VectorParams(size=vector_size, distance=Distance.COSINE))


def batched(iterable: List, batch_size: int):
    for i in range(0, len(iterable), batch_size):
        yield iterable[i : i + batch_size]


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Markdown docs into Qdrant")
    parser.add_argument("--docs", type=str, required=True, help="Path to docs folder or single file")
    parser.add_argument("--only", type=str, default="", help="Ingest only this file (overrides --docs directory scan)")
    parser.add_argument("--collection", type=str, default="api_docs", help="Qdrant collection name")
    parser.add_argument("--recreate", action="store_true", help="Drop and recreate collection")
    parser.add_argument("--batch-size", type=int, default=16, help="Embedding/upsert batch size")
    parser.add_argument("--max-files", type=int, default=0, help="Limit number of files for ingestion (0 = no limit)")
    args = parser.parse_args()

    docs_path = Path(args.docs)
    assert docs_path.exists(), f"Docs path not found: {docs_path}"

    cfg = get_settings()
    client = QdrantClient(url=cfg.qdrant_url, api_key=cfg.qdrant_api_key, timeout=60)
    ensure_collection(client, args.collection, vector_size=1024, recreate=args.recreate)

    from sentence_transformers import SentenceTransformer  # local import for faster startup

    embedder = SentenceTransformer(cfg.embedding_model, device=cfg.embedding_device)
    # Choose best available prompt name for documents depending on model presets
    prompts = getattr(embedder, "prompts", {}) or {}
    doc_prompt = None
    if isinstance(prompts, dict):
        if "document" in prompts:
            doc_prompt = "document"
        elif "passage" in prompts:
            doc_prompt = "passage"

    if args.only:
        file_list = [Path(args.only)]
    else:
        if docs_path.is_file():
            file_list = [docs_path]
        else:
            file_list = sorted(docs_path.rglob("*.md"))
            if args.max_files and args.max_files > 0:
                file_list = file_list[: args.max_files]

    use_progress = len(file_list) > 1 and not args.only
    pbar = tqdm(total=len(file_list), desc="Files", unit="file") if use_progress else None
    total_chunks = 0

    for file_path in file_list:
        start_file = time.time()
        sections, _fm = parse_markdown(file_path)
        chunks = chunk_sections(sections)
        texts = [c[0] for c in chunks]
        metas = [c[1] for c in chunks]
        print(f"{file_path.name}: {len(chunks)} chunks", flush=True)
        if not texts:
            if pbar:
                pbar.update(1)
            continue

        for text_batch, meta_batch in zip(batched(texts, args.batch_size), batched(metas, args.batch_size)):
            t0 = time.time()
            embeddings = None
            # Try with selected document prompt if available; fall back gracefully
            try:
                if doc_prompt:
                    embeddings = embedder.encode(
                        text_batch,
                        normalize_embeddings=True,
                        batch_size=min(args.batch_size, 8),
                        show_progress_bar=False,
                        prompt_name=doc_prompt,
                    )
            except Exception:
                embeddings = None
            if embeddings is None:
                embeddings = embedder.encode(
                    text_batch,
                    normalize_embeddings=True,
                    batch_size=min(args.batch_size, 8),
                    show_progress_bar=False,
                )
            t1 = time.time()
            points = []
            for vec, text, meta in zip(embeddings, text_batch, meta_batch):
                payload = {"text": text, **meta}
                points.append({
                    "id": str(uuid.uuid4()),
                    "vector": vec.tolist(),
                    "payload": payload,
                })
            client.upsert(collection_name=args.collection, points=points)
            t2 = time.time()
            total_chunks += len(points)
            print(f"{file_path.name}: batch {len(points)} emb {t1-t0:.2f}s upsert {t2-t1:.2f}s", flush=True)
        if pbar:
            pbar.update(1)
        print(f"Done {file_path.name} in {time.time()-start_file:.2f}s (total {total_chunks})", flush=True)

    if pbar:
        pbar.close()
    print(f"Ingested {total_chunks} chunks into collection '{args.collection}'.")


if __name__ == "__main__":
    main()