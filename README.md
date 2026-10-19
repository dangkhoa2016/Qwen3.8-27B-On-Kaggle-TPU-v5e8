# Qwen3.8-27B on Kaggle TPU v5e-8 — Text, Vision & API Showcase

[![CPU CI](https://github.com/dangkhoa2016/Qwen3.8-27B-On-Kaggle-TPU-v5e8/actions/workflows/cpu-acceptance.yml/badge.svg)](https://github.com/dangkhoa2016/Qwen3.8-27B-On-Kaggle-TPU-v5e8/actions/workflows/cpu-acceptance.yml)
[![MIT](https://img.shields.io/badge/Source-MIT-green.svg)](LICENSE)
[![Qwen Apache 2.0](https://img.shields.io/badge/Model-Apache--2.0-blue.svg)](MODEL_LICENSE.md)
![Python](https://img.shields.io/badge/Python-3.12-3776AB)
![TPU](https://img.shields.io/badge/TPU-v5e--8-4285F4)
![JAX](https://img.shields.io/badge/Inference-JAX%20%2B%20MaxText-purple)
![TPU HTTP](https://img.shields.io/badge/TPU%20HTTP-Pending-orange)

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](README.vi.md)

Independent engineering showcase for original Qwen3.8-27B BF16 Text + Vision on Kaggle TPU v5e-8 using JAX/MaxText. No weight quantization, no model weights committed. Not affiliated with Qwen.

## Verification boundary

- **TPU notebook:** fresh Text/Vision execution evidence exists in [evidence/](evidence/); the newer Git-clone notebook needs its own fresh Run All.
- **CPU REST / UI:** authenticated contracts, fail-closed guards, and Playwright CI have passed.
- **Real TPU HTTP server:** **not yet accepted**. Experimental mode is loopback-only and opt-in. Do not imply mock replies are model inference.
- Original showcase thumbnails are 256×192; three 1024×768 visual assets are inspection-only and are **not** independently inference-qualified.
- Rich-scene OCR was **22/55** in prior evidence, and should not be described as broadly accurate.

## Quick start (CPU mock)

```bash
python -m pip install -r requirements-cpu.txt
python -m pytest -q tests/test_api.py tests/test_runtime.py tests/test_http_runtime.py tests/test_tpu_factory_auth.py tests/test_serving_config.py tests/test_tpu_asgi.py
QWEN38_BACKEND=mock python -m uvicorn api.app:app --host 127.0.0.1 --port 8090
```

The development frontend is served at `http://127.0.0.1:8090`. No real model is loaded.

## Components

- `notebooks/`: Kaggle showcase, Git-pinned examples and image checksum validation
- `api/`: mock ASGI, guarded experimental TPU entrypoint, MaxEngine adapter, lifecycle and security gates
- `web/`: text/vision frontend
- `evidence/`: previously executed notebook, JSON and SHA256 manifest
- `tests/`: CPU contract and browser tests

## Documentation

[Architecture](docs/ARCHITECTURE.md) · [Reproducibility](docs/REPRODUCIBILITY.md) · [Acceptance](docs/ACCEPTANCE.md) · [Changelog](CHANGELOG.md) · [Contributing](.github/CONTRIBUTING.md) · [Security](.github/SECURITY.md) · [Support](.github/SUPPORT.md).

## Licensing

Original project code/docs: **MIT © 2026 Đăng Khoa <i.am@dangkhoa.dev>**. Upstream Qwen3.8-27B model/weights: **Apache-2.0**. See [MODEL_LICENSE.md](MODEL_LICENSE.md) and [third-party notices](THIRD_PARTY_NOTICES.md). Neither the repository nor this MIT license grants ownership of upstream weights.
