# Try the Text + Vision API playground

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](README.vi.md)

The API playground demonstrates request validation and browser interaction with **FastAPI and Vue CDN**. It runs on a normal CPU and requires **no Qwen weights**—a convenient starting point before exploring the TPU notebook.

## Run locally

From the repository root with Python 3.12 and Node.js installed:

```bash
python -m pip install -r requirements-cpu.txt
QWEN38_BACKEND=mock python -m uvicorn api.app:app --host 127.0.0.1 --port 8090
```

Open **http://127.0.0.1:8090** to try text messages and image uploads. The UI labels the backend **MOCK BACKEND · NO MODEL LOADED**. A mock response illustrates an API contract; **it is not a Qwen-generated response**.

## Routes

| Method | Endpoint | Example use |
|---|---|---|
| GET | `/health` | See health state |
| GET | `/metrics` | Inspect basic runtime counters |
| POST | `/v1/chat/completions` | Submit text in the supported request schema |
| POST | `/v1/vision/analyze` | Submit multipart image plus prompt |

Chat and Vision completion responses from the mock path are marked `mock: true`. For precise payload examples and validation behavior, inspect [API tests](../tests/test_api.py) and the [browser tests](../tests/browser/showcase.spec.cjs).

## Check expected behavior

```bash
python -m pytest -q tests
npm ci
npx playwright install chromium
npm run test:e2e
```

The Python and Playwright suites verify validation, errors, UI behavior and CPU mock contracts. They do not measure TPU inference or model quality.

## Moving from a demo to real inference

Real 27B serving needs a compatible BF16 checkpoint, eight TPU devices, a checksum-verified runtime, one loaded engine, authentication and bounded concurrency. The experimental adapter provided here is **not yet qualified for live TPU HTTP traffic**; keep it disabled and bound to loopback.

Read [TPU API architecture and future validation](TPU-INTEGRATION.md) and [the actual notebook pipeline](../docs/ARCHITECTURE.md).
