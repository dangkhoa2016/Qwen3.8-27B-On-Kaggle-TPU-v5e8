# CPU API development

Development endpoints use FastAPI with a clearly marked **MOCK** backend.
They do **not** load or execute the Qwen3.8-27B weights.
Do not publicly expose this development server; no authentication is implemented.

Run locally from the repository root:

```bash
python -m pip install fastapi uvicorn python-multipart httpx pytest
QWEN38_BACKEND=mock python -m uvicorn api.app:app --host 127.0.0.1 --port 8090
python -m pytest -q tests/test_api.py
```

Routes: `GET /health`, `GET /metrics`, `POST /v1/chat/completions`,
`POST /v1/vision/analyze` (multipart). All completion responses include `mock: true`.
The future TPU adapter needs a **separate**, explicitly qualified deployment gate,
a single loaded MaxEngine, bounded request concurrency, and shutdown hooks.
