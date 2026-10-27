# Reproduce the Qwen3.8-27B TPU experiment

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](REPRODUCIBILITY.vi.md)

This guide helps you understand the exact resources needed to reproduce original-BF16 Text+Vision inference on Kaggle TPU v5e-8. It separates **repeatable preparation steps** from **results that still require an actual TPU run**.

## Choose an execution route

| Route | Intended use | Access / result |
|---|---|---|
| CPU tests and mock UI | Review API contracts and browser behavior | Works without model weights or TPU |
| Original Full19 support | Study the 27 October experimental setup | Requires recorded support files; 19/19 experimental output was archived separately |
| Replacement-image Full19 | Evaluate the new green-signal crossing scene | Needs a notebook/support pairing that uses that exact image SHA256 and fresh TPU output |

**Kaggle model and support attachments may require authorized access.** The identifiers and SHA256 below document recorded inputs, not a promise that historical Datasets remain publicly downloadable.

## Prerequisites

- A Kaggle TPU v5e-8 runtime showing **exactly eight TPU devices**; one MaxEngine, one BF16 checkpoint load.
- Kaggle Model attached for actual inference: [Kaggle BF16 mirror](https://www.kaggle.com/models/dangkhoa2016/qwen-qwen3-8-27b) (`dangkhoa2016/qwen-qwen3-8-27b`, Transformers/default/v1). The BF16 index declares **18 shards**, **1,199 tensor entries** and **55,562,855,904 indexed bytes**.
- A support Dataset matching the notebook-pinned manifest. **The Git notebook source is the maintained reference**; any alternative input snapshot requires a matching manifest and independently generated results.
- A runtime compatible with the pinned support and model. The recorded experimental software baseline includes **JAX/JAXLIB 0.11.0** and **Transformers 5.12.1**.

| Dataset | Purpose | Portable manifest SHA256 |
|---|---|---|
| `Original experimental support snapshot` | Preserved 27 October input | `1447ee114dc2d2abce6f60f5125927f55311bf530b8cf53ab86a2576ac637ebc` |
| `Replacement-image support snapshot` | New crossing-image candidate | `87d172b3b3e2fb4b478888665da164e6abe7cdff3f566b4699d396c919a2ec08` |

The replacement-image input snapshot was checksum-verified (**48/48 SHA256**), but that check does **not** confirm inference. Use the [maintained Full19 notebook source](../notebooks/kaggle-tpu-v5e8-production-showcase.ipynb) with its own pinned support for reproducibility. For any updated input snapshot, first align the manifest in the notebook and run the model again.

## Step 1 — Explore without a TPU

From a checked-out repository:

```bash
python -m pip install -r requirements-cpu.txt
python -m pytest -q tests
npm ci
npx playwright install chromium
npm run test:e2e
```

These exercises validate the API, input guards and mock UI. The latest checked source passed **36 Python tests, one skipped, and two Playwright tests**. They do **not** run Qwen inference.

## Step 2 — Prepare the Kaggle notebook

1. Enable **TPU v5e-8** and disable Internet if using the offline bundle.
2. Attach the **original BF16 Kaggle Model**, not a GGUF/quantized replacement.
3. Attach support files matching the chosen notebook and its SHA256 manifest. If using the replacement crossing input, update its manifest and rerun the full experiment; do not combine an image from one snapshot with the checksum from another.
4. Confirm that Kaggle has expanded the support archive. The notebook finds `portable-bundle-manifest.json` at a flat mount or within its named extracted subdirectory. `QWEN38_SUPPORT_DIR` can point to a custom location.
5. Keep the notebook's **preflight cell first**. It verifies all 48 file sizes/hashes, required patches, model index and dataset paths **before importing JAX**.

## Step 3 — Execute and observe

The source uses **320-token prefill**, **768-token target** and **512 × 384 inference image resolution**. Execute its cells sequentially in one TPU session and preserve the generated timestamped evidence directory under `/kaggle/working`.

For each of the 19 cases, record the input SHA256, original prompt, generated token IDs and full text, EOS, durations and exceptions. Preserve the device list, model index digest, patch and manifest digests, actual runtime versions and overall elapsed time. A valid cached executable does not exempt native JAX/XLA sharding and dtype checks.

## Step 4 — Compare fairly

The older experimental Full19 completed **19/19 automated cases with EOS in 1,160.844 seconds**. It includes known semantic errors: invoice tax misreading and a dashboard ratio comparison. **Do not compare a new image against answers generated for an older image hash.**

A new run can be called *complete* only after it produces all required cases. Assess factual accuracy separately; keep incorrect responses and failed attempts in the output. See [evaluation guide](ACCEPTANCE.md) and [case evidence](FULL19-EVIDENCE.md).

## Troubleshooting

| Symptom | First check |
|---|---|
| Model not found | Model attachment path, index file and all 18 shards |
| Support SHA256 mismatch | Notebook/Dataset pair, expanded ZIP subdirectory and version |
| Fewer than 8 devices | Accelerator selection and JAX TPU backend |
| AOT shape or sharding failure | Image-token offsets, tensor shapes, dtype, native layout and pinned patch |
| EOS not reached | Decode budget, last token ID and exact raw output |
| Model answer incorrect | Original image and prompt, semantic rubric; do not alter generated text |

[Architecture](ARCHITECTURE.md) · [Support bundle](OFFLINE-SUPPORT.md) · [Acceptance](ACCEPTANCE.md) · [Image examples](../examples/README.md)
