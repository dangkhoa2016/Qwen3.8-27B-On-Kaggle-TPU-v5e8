# From a local API playground to TPU-backed inference

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](TPU-INTEGRATION.vi.md)

This repository provides an approachable **FastAPI + Vue CDN playground** to demonstrate a text/vision request contract. It also contains an **opt-in TPU adapter design**. The two are intentionally separate: the default browser demo does **not** load Qwen3.8-27B or make live TPU inference requests.

## Get the CPU playground running

```bash
python -m pip install -r requirements-cpu.txt
QWEN38_BACKEND=mock python -m uvicorn api.app:app --host 127.0.0.1 --port 8090
```

Open `http://127.0.0.1:8090`. The UI explicitly labels the backend as a mock. This path is useful to explore response envelopes, image uploads, error states and browser behavior while studying TPU code separately.

| Route | Purpose |
|---|---|
| `GET /health` | Runtime health |
| `GET /metrics` | Local runtime counters |
| `POST /v1/chat/completions` | Text request contract |
| `POST /v1/vision/analyze` | Multipart image analysis contract |

Mock responses expose their mock status. Do not quote them as outputs of the 27B model.

## How a real TPU API would be structured

| Component | Responsibility |
|---|---|
| `api/app.py` | Input validation, HTTP routes and errors |
| `api/serving_config.py` | Environment and security settings |
| `api/runtime.py` | READY/FAILED/CLOSED lifecycle, serialized jobs and shutdown |
| `api/tpu_asgi.py` | Explicit opt-in ASGI entry point |
| `api/tpu_factory.py` | Model verification and MaxEngine construction |
| `api/maxengine_adapter.py` | Text/Vision preprocess, prefill and generation |

A 27B BF16 engine should be loaded **once**, after checkpoint and eight-device validation. Do not spawn a new MaxEngine for every HTTP request. Readiness should reflect model state, not merely a listening port.

## Security and capacity design

The experimental TPU path is guarded for **loopback** use and a **single worker**, with explicit configuration and an API key of at least **24 characters**. Source-level request bounds include **4,000 prompt characters**, **8 MiB images** and **160 generated tokens**. Busy/unavailable requests are designed to return **429/503** rather than start unlimited workloads.

These are **source contracts and CPU test observations**, not evidence of a hardened production TPU service. Do not expose an unqualified TPU endpoint publicly.

## Important integration difference

The version-controlled notebook configures **320/768 prefill/target tokens** and 512 × 384 image preprocessing. The independent HTTP adapter currently has different token and image settings. Its outputs and serving performance must therefore be validated separately; the notebook measurements do not qualify this adapter.

## What would qualify TPU HTTP serving?

1. Align model, support manifest, image processing, token budgets and MRoPE handling with an accepted TPU notebook.
2. Run authenticated Text and Vision requests against **actual TPU devices**.
3. Measure startup, readiness, single-request latency and bounded queue behavior.
4. Test invalid images, oversize prompts, unauthorized requests, concurrency, 429/503 and graceful shutdown.
5. Preserve real generated tokens, EOS, input SHA256 and errors; compare against independent notebook results.

[CPU API quickstart](README.md) · [TPU architecture](../docs/ARCHITECTURE.md) · [Evaluation](../docs/ACCEPTANCE.md)
