# Qwen3.8-27B on Kaggle TPU v5e-8

## Original BF16 Text + Vision inference · FastAPI · Vue playground

[![CPU CI](https://github.com/dangkhoa2016/Qwen3.8-27B-On-Kaggle-TPU-v5e8/actions/workflows/cpu-acceptance.yml/badge.svg)](https://github.com/dangkhoa2016/Qwen3.8-27B-On-Kaggle-TPU-v5e8/actions/workflows/cpu-acceptance.yml)
[![Source MIT](https://img.shields.io/badge/Project%20code-MIT-green)](LICENSE)
[![Model Apache-2.0](https://img.shields.io/badge/Model-Apache--2.0-blue)](THIRD_PARTY_LICENSES/QWEN3.8-27B-APACHE-2.0)
![TPU v5e-8](https://img.shields.io/badge/Kaggle%20TPU-v5e--8-blue)
![BF16](https://img.shields.io/badge/Original%20weights-BF16-purple)
![HTTP pending](https://img.shields.io/badge/Real%20TPU%20HTTP-Pending-orange)

**Language:** English | [Tiếng Việt](README.vi.md)

An independent, auditable engineering showcase for the original [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) image/text-to-text model, deployed on Kaggle TPU v5e-8 with JAX and patched MaxText. The project includes a reproducible notebook, guarded experimental inference runtime, CPU REST mock, no-build Vue CDN interface, Python/browser acceptance and reproducibility evidence.

**Principles:** original BF16 weights, eight TPU devices, no GGUF, no quantization, no pretrained weights committed to Git. Not affiliated with Qwen.

## At a glance

| Part | Implementation | What is verified |
|---|---|---|
| Notebook | Kaggle TPU JAX + MaxText, Text and Vision | Fresh Run All with new image assets is pending; no old outputs retained |
| Model provenance | 18 shards, index, patch and source archive checks | CPU preflight, not new model execution |
| Text examples | Deterministic prompts and semantic checks | Fresh model evaluation pending; not current HTTP acceptance |
| Vision examples | Numeric regression cards and six scenes | Updated 1280×960 input scenes; OCR scores require a new TPU run |
| REST service | FastAPI mock + guarded TPU ASGI | CPU contract tests; live TPU HTTP not yet qualified |
| Interface | Vue 3 CDN, Text Chat + Vision upload | Browser E2E via Playwright |
| CI | Python 3.12, Node 22, Chromium | CPU tests and mock API only |

Full evidence and caveats: [acceptance](docs/ACCEPTANCE.md), [reproducibility](docs/REPRODUCIBILITY.md), and [evidence files](evidence/).

## System architecture

1. **Attached assets:** a Kaggle model mirror of original Qwen BF16 weights plus separately attached offline MaxText support artifacts. Check source and checkpoint provenance before TPU load.
2. **Inference:** one MaxEngine instance, a BF16 parameter tree, JAX backend detection, deterministic Text/Vision scenarios, and measured initialization/prefill/decode timings.
3. **Service boundary:** bounded request sizes, explicit opt-in TPU backend, loopback binding, authentication requirements, single active model request and fail-closed lifecycle behavior. Live HTTP inference remains unverified.
4. **Review UI:** a lightweight Vue playground served by the CPU mock FastAPI backend. Mock results are labeled; they are not real Qwen inference.
5. **Verification:** Python contract tests, Playwright browser E2E, checkpoint/source checks and new TPU notebook and results to be captured with SHA-256.

See [architecture](docs/ARCHITECTURE.md) and [TPU integration](api/TPU-INTEGRATION.md).

## Kaggle TPU notebook setup

| Requirement | Value |
|---|---|
| Accelerator | Kaggle TPU v5e-8, eight TPU devices |
| Model | dangkhoa2016/qwen-qwen3-8-27b (Transformers/default/version 1) |
| Model mount | /kaggle/input/models/dangkhoa2016/qwen-qwen3-8-27b/transformers/default/1 |
| Offline support | dangkhoa2016/qwen3-8-27b-tpu-v5e8-offline-support |
| Support mount | /kaggle/input/datasets/dangkhoa2016/qwen3-8-27b-tpu-v5e8-offline-support |
| Notebook | [Kaggle TPU showcase notebook](notebooks/kaggle-tpu-v5e8-production-showcase.ipynb) |
| Software baseline | JAX/JAXLIB 0.11.0, Transformers 5.12.1 |

**Before Run All:** attach both Kaggle resources, choose TPU v5e-8, keep the notebook bootstrap instructions, and enable **Internet ON** only for anonymous commit-pinned public GitHub examples. Offline model/support files should not be downloaded again unnecessarily. Never load two 27B instances concurrently.

All 11 images are high-resolution (8 at 1280×960 and 3 at 1024×768), documented in [examples/images](examples/images/) with [SHA-256 asset manifest](examples/assets_manifest.json). Some are synthetic diagrams or illustrations, and three are photographic inputs; none should be presented as model-generated outputs. The notebook pins an immutable example commit for reproducibility.

## Local CPU test and mock demo

Use Python 3.12 and Node/npm.

~~~bash
git clone https://github.com/dangkhoa2016/Qwen3.8-27B-On-Kaggle-TPU-v5e8.git
cd Qwen3.8-27B-On-Kaggle-TPU-v5e8
python -m pip install -r requirements-cpu.txt
python -m pytest -q tests
npm ci
npx playwright install chromium
npm run test:e2e
~~~

Start the **CPU mock only**:

~~~bash
QWEN38_BACKEND=mock python -m uvicorn api.app:app --host 127.0.0.1 --port 8090
~~~

Open http://127.0.0.1:8090. This does not load the 27B model. Health, readiness, metrics, Text Chat and Vision upload contracts can be reviewed without claiming real inference.

## Performance and evidence boundaries

- [Current acceptance criteria](docs/ACCEPTANCE.md)
- [Reproducibility instructions](docs/REPRODUCIBILITY.md)
- New TPU executed notebook and measured evidence: **pending qualification**
- [Acceptance criteria](docs/ACCEPTANCE.md)

The current public notebook and replacement Vision inputs have not yet been accepted through a new end-to-end TPU run. Do not reuse cold/warm timings or OCR results from earlier iterations. Real TPU HTTP Text/Vision, concurrency, auth, readiness and shutdown still require live tests. A green CPU CI badge does not qualify those endpoints.

## Repository map and documentation

| Path | Purpose |
|---|---|
| notebooks/ | Kaggle BF16 Text and Vision showcase |
| api/ | FastAPI, MaxEngine adapter, lifecycle, gated TPU startup |
| web/ | Vue CDN interface |
| examples/ | Input images, asset manifest, source notes |
| docs/ACCEPTANCE.md | Current acceptance criteria and pending TPU evidence |
| tests/ | Python and browser acceptance |
| docs/ | Architecture, reproducibility, acceptance |
| .github/ | CI and governance |

[Reproducibility](docs/REPRODUCIBILITY.md) · [Changelog](CHANGELOG.md) · [Contributing](.github/CONTRIBUTING.md) · [Security](.github/SECURITY.md) · [Support](.github/SUPPORT.md).

## Licensing and attribution

**Two separate licenses apply.** Original project code and documentation: [MIT © 2026 Đăng Khoa](LICENSE). Upstream **Qwen/Qwen3.8-27B pretrained model and weights: Apache License 2.0**, not MIT.

**[Read the full model license stored in this Git repository](THIRD_PARTY_LICENSES/QWEN3.8-27B-APACHE-2.0)**, [model attribution policy](MODEL_LICENSE.md), [official upstream model card](https://huggingface.co/Qwen/Qwen3.8-27B), and [third-party notices](THIRD_PARTY_NOTICES.md). No model weights are committed here. Separately attached Kaggle model mirrors and patched runtime assets must independently satisfy applicable upstream licensing and attribution obligations.

**Project status:** implemented components and CPU-tested contracts, TPU notebook and real HTTP acceptance pending. Do not interpret this project as an official Qwen release.
