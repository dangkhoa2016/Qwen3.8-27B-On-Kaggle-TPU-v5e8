# Qwen3.8-27B on Kaggle TPU v5e-8

**Original BF16 · Text + Vision · JAX / MaxText · Reproducible engineering examples**

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](README.vi.md)

[![CPU checks](https://github.com/dangkhoa2016/Qwen3.8-27B-On-Kaggle-TPU-v5e8/actions/workflows/cpu-acceptance.yml/badge.svg)](https://github.com/dangkhoa2016/Qwen3.8-27B-On-Kaggle-TPU-v5e8/actions/workflows/cpu-acceptance.yml)
[![Project code: MIT](https://img.shields.io/badge/Project_code-MIT-green)](LICENSE)
[![Model: Apache 2.0](https://img.shields.io/badge/Upstream_model-Apache--2.0-blue)](MODEL_LICENSE.md)
![Kaggle TPU v5e-8](https://img.shields.io/badge/Accelerator-TPU_v5e--8-blue)
![Original BF16](https://img.shields.io/badge/Checkpoint-Original_BF16-purple)

This project explores how to run **Qwen3.8-27B BF16 from the [Kaggle model mirror](https://www.kaggle.com/models/dangkhoa2016/qwen-qwen3-8-27b)**, on **eight Kaggle TPU v5e devices** using JAX and patched MaxText. In the tested environment the notebook loads the attached Kaggle checkpoint—not directly from a Hugging Face URL. The repository combines measured TPU inference results, a 19-case Text + Vision notebook, a FastAPI/Vue CPU playground and documentation for input integrity and evaluation.

The goal is to share **practical lessons in large-model TPU execution**—especially multimodal fusion, AOT prefill, input integrity, performance measurement and honest interpretation of model answers.

## What you can explore

| Learning path | What you get | Where to begin |
|---|---|---|
| Understand TPU multimodal inference | Visual-to-token pipeline, MRoPE and JAX/XLA checks | [Architecture](docs/ARCHITECTURE.md) |
| Try the API without TPU | Local FastAPI + Vue CDN mock and test suite | [CPU quickstart](#try-the-cpu-playground) |
| Study the actual input examples | 16 hashed images and 19 case definitions | [Example gallery](examples/README.md) |
| Run the TPU notebook | Original BF16 model, offline support and notebook setup | [Step-by-step guide](docs/REPRODUCIBILITY.md) |
| Evaluate model quality | OCR, image reasoning, exact outputs and observed errors | [Evaluation guide](docs/ACCEPTANCE.md) |
| Interpret speed fairly | One measured run, cold/warm methodology and gaps | [Performance guide](docs/PERFORMANCE-ANALYSIS.md) |

## Explore the actual input images

These are **model inputs**, not generated output images from Qwen. The photography collection includes AI-created illustrations that are clearly labeled as such.

| Street crossing | Illustrated 3D scene | Vietnamese meal |
|---|---|---|
| ![Green-signal pedestrian crossing](examples/images/photo_ai_city_crossing_1280x960.png) | ![Robotics station example](examples/images/vision_3d_robotics_station.jpg) | ![Vietnamese meal example](examples/images/photo_ai_vietnamese_meal_1280x960.png) |
| Street-signal and spatial reasoning | Objects and scene relationships | Food and object recognition |

Find the exact case prompts, dimensions and SHA256 in the [16-image manifest](examples/assets_manifest.json) and [examples guide](examples/README.md).

## Measured inference on eight TPU devices

A real Kaggle TPU v5e-8 session on **27 October 2026** recorded:

| Measurement | Result |
|---|---|
| Checkpoint | Qwen3.8-27B original BF16, **18 safetensors shards** |
| Accelerator | Eight v5e devices; one MaxEngine and checkpoint load |
| Workload | **3 Text + 2 OCR + 6 Rich Vision + 8 Photography = 19 cases** |
| Generation completion | **19/19 cases reached EOS** |
| Selected checks | Text 3/3, numeric OCR **731 / 482**, Rich Vision **55/55** keyword checks |
| Total wall time | **1,160.844 seconds (19 min 20.844 s)** |

**Interpretation:** these are observed results for the input hashes and runtime of that measured session, not an accuracy claim for all tasks or an execution record of each subsequent notebook revision. Some responses contained numeric errors (invoice tax and a dashboard comparison), which are documented without alteration.

See [Full19 evidence](docs/FULL19-EVIDENCE.md), [model behavior](docs/FAILURE-ANALYSIS.md) and [performance analysis](docs/PERFORMANCE-ANALYSIS.md).

## Try the CPU playground

You can explore the request/response behavior without a TPU or model checkpoint. From the repository root:

```bash
python -m pip install -r requirements-cpu.txt
QWEN38_BACKEND=mock python -m uvicorn api.app:app --host 127.0.0.1 --port 8090
```

Open **http://127.0.0.1:8090**. The interface clearly says **MOCK BACKEND · NO MODEL LOADED**; its answers are not inference from Qwen3.8-27B.

To run the checks:

```bash
python -m pytest -q tests
npm ci
npx playwright install chromium
npm run test:e2e
```

The latest verified source passed **38 Python tests (one skipped)** and **two Playwright browser tests**. See [API documentation](api/README.md) for routes and validation behavior.

## Run the BF16 Text + Vision notebook on TPU

The [version-controlled Full19 notebook](notebooks/kaggle-tpu-v5e8-production-showcase.ipynb) uses **one MaxEngine**, a separate Vision encoder, JAX/XLA prefill, **512 × 384** input tensors, a **320-token prefill** and a **768-token target**.

In Kaggle, select **TPU v5e-8** and attach both the [BF16 model mirror](https://www.kaggle.com/models/dangkhoa2016/qwen-qwen3-8-27b) and the [Full19 support Dataset](https://www.kaggle.com/datasets/dangkhoa2016/qwen38-27b-tpu-v5e8-full19-runtime-support). The notebook verifies a 48-file support manifest with SHA256 `cf5343f614eb32ff9234c870e4a26093f9f07e62199ee90396d516a804eb8a28` before checkpoint loading.

Follow the [execution guide](docs/REPRODUCIBILITY.md) and [offline support documentation](docs/OFFLINE-SUPPORT.md). Attachment availability and third-party license requirements may restrict access. The Git notebook contains source cells and generates new inference outputs when run; the separately recorded TPU metrics above are not prefilled outputs in that notebook.

## Architecture and repository map

```text
notebooks/   Full19 BF16 / Split-Vision TPU notebook source
examples/    Input images, manifest SHA256 and prompts
docs/        Architecture, reproduction, evaluation and performance guides
api/         FastAPI CPU playground and separate TPU adapter source
web/         Vue CDN browser playground
tests/       Python and browser regression tests
support/     Pinned runtime-support manifest
```

A central design choice is to **encode images separately, fuse image embeddings with text using MRoPE positions, then run a validated JAX/XLA compiled prefill**. This is explained with a pipeline diagram in [Architecture](docs/ARCHITECTURE.md). The optional HTTP TPU adapter is a separate path and has **not** been qualified by live TPU HTTP requests.

## Reuse, attribution and community

The repository contains **no pretrained Qwen weights**. Project-authored code/docs use [MIT](LICENSE), while the upstream model uses the separately documented [Apache-2.0 license](MODEL_LICENSE.md). Third-party runtime components and input-image distribution require their own review; see [notices](THIRD_PARTY_NOTICES.md).

Ideas, documentation corrections and reproducible regression tests are welcome. Read [Contributing](CONTRIBUTING.md), [Support](SUPPORT.md) and [Security](SECURITY.md). This is an independent engineering demonstration, **not an official Qwen project or endorsement**.

[Project changes](CHANGELOG.md) · [Architecture](docs/ARCHITECTURE.md) · [Reproduction](docs/REPRODUCIBILITY.md) · [Evaluation](docs/ACCEPTANCE.md)
