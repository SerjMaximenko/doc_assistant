# RAG over Markdown (FastAPI + Qdrant)

Minimal but production-ready-ish RAG service answering questions about internal API docs stored as Markdown. Uses bge-m3 embeddings, optional bge-reranker, and Mistral Instruct via OpenAI-compatible (vLLM) or llama.cpp server.

## Quickstart

1. Copy env and adjust as needed:

```bash
cp .env.example .env
```

2. Start Qdrant:

```bash
docker compose up -d qdrant
```

3. Create a `docs/` folder at repo root and place your `.md` files there.

4. Install Python deps using uv:

```bash
uv sync --all-extras
```

5. Ingest Markdown into Qdrant:

```bash
python scripts/ingest_md.py --docs ./docs --collection ${QDRANT_COLLECTION:-api_docs} --recreate
```

6. Run the API:

```bash
uvicorn app.main:app --reload
```

- Search: `POST /search` with `{ "query": "...", "top_k": 20, "filters": {"service": "..."} }`
- Answer: `POST /answer` with `{ "query": "...", "with_rerank": true }`

## LLM Backends
- OpenAI-compatible (vLLM): set `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `OPENAI_MODEL`
- llama.cpp server: set `LLAMA_BASE_URL`, `LLAMA_MODEL`

If both are set, OpenAI-compatible is preferred.

## Notes
- Embeddings: `BAAI/bge-m3` with normalize embeddings enabled
- Reranker (optional): `BAAI/bge-reranker-large`
- Collection: `api_docs`, vectors: 1024-dim, cosine

## Testing

```bash
pytest
```

Some tests are integration and will skip if Qdrant is not reachable.

## Local LLM (llama.cpp) quick scripts

- Build llama.cpp (CPU):
```bash
sudo apt-get update -y && sudo apt-get install -y cmake build-essential libcurl4-openssl-dev
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
cmake -S . -B build -D CMAKE_BUILD_TYPE=Release && cmake --build build -j$(nproc)
```

- Download a GGUF model (choose one):
```bash
# Install CLI (inside your venv if needed)
python -m pip install -U 'huggingface_hub[cli]'

# Mixtral 8x7B Instruct (Q3_K_M ~20GB)
hf download TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF \
  mixtral-8x7b-instruct-v0.1.Q3_K_M.gguf --local-dir ./models

# or smaller Mistral 7B Instruct (Q4_K_M ~4-5GB)
hf download TheBloke/Mistral-7B-Instruct-v0.2-GGUF \
  mistral-7b-instruct-v0.2.Q4_K_M.gguf --local-dir ./models
```

- Start llama.cpp server:
```bash
nohup ./llama.cpp/build/bin/llama-server \
  -m ./models/<MODEL_FILE>.gguf -c 2048 -t $(nproc) \
  --host 0.0.0.0 --port 8080 >/tmp/llama.log 2>&1 &
```

- Configure API to use llama.cpp:
```bash
cat > rag-md/.env <<EOF
LLAMA_BASE_URL=http://localhost:8080
LLAMA_MODEL=mixtral
EOF
```

- Start Qdrant and ingest docs:
```bash
docker compose -f rag-md/docker-compose.yml up -d qdrant
cd rag-md
uv sync --all-extras
python scripts/ingest_md.py --docs ./docs --collection api_docs --recreate
```

- Launch API:
```bash
./.venv/bin/uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Async answers (non-blocking)
- Start an async job:
```bash
curl -s -X POST http://localhost:8000/answer_async/start \
  -H 'Content-Type: application/json' \
  -d '{"query":"Как создать товар","top_k":12,"max_context_tokens":200,"with_rerank":false}'
# => {"job_id":"..."}
```
- Poll status / get result:
```bash
curl -s http://localhost:8000/answer_async/status/<JOB_ID>
```

### Filters note
To use `filters` (e.g. `{ "service": "sales" }`) you must add such keys in the Markdown frontmatter and re-ingest the docs. 