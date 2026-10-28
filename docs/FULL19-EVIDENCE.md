# Full19: TPU inference results and evidence

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](FULL19-EVIDENCE.vi.md)

Full19 is a 19-case Text + Vision exercise for Qwen3.8-27B BF16 on Kaggle TPU v5e-8. It measures whether inference completes and provides examples for visual reasoning and numeric OCR. It is not a statistically representative model-accuracy benchmark.

## Recorded TPU measurements — 27 October 2026

| Property | Observation |
|---|---|
| Checkpoint | Qwen3.8-27B original BF16, 18 safetensors shards |
| Hardware | 8 TPU v5e devices, one MaxEngine |
| Runtime | JAX / patched MaxText, Split-Vision |
| Cases | 3 Text, 2 numeric OCR, 6 Rich Vision, 8 Photography |
| Preprocessing | 512 × 384 image tensors |
| Token configuration | Prefill 320, target 768, up to 448 Rich/Photo decode steps |
| Automated completion | **19/19 cases reached EOS** |
| Selected checks | 3/3 Text, OCR 731 and 482, Rich Vision 55/55 keywords |
| Total wall time | **1,160.844 seconds** |

These figures refer to the input hashes and runner of the **measured session**. The repository notebook and example assets can be used to generate a new evidence set, but benchmark metrics must not be attributed to a different input hash or notebook revision without running it.

## What is captured during inference

A complete record should include the Git revision, model revision, TPU device topology, package versions, support manifest SHA256, image SHA256, prompt, generated token IDs, full text, EOS, per-case timing and any exceptions. When evaluating Vision, retain the processed image dimensions and multimodal token positions.

## Quality observations

- **Numeric OCR cards:** the simple printed-number tasks returned **731** and **482**, respectively.
- **Rich Vision:** 55 keyword checks matched across six scenes; this does not establish perfect descriptions.
- **Invoice:** the image contains **8% / $69.28** tax, but the model reported **5% / $66.28**.
- **Dashboard:** a value of **114** compared with **31** was described as “nearly double”; the actual ratio is approximately **3.68**.

The original generated errors are useful evaluation findings. Automated completion and semantic correctness must be reported as different measures.

[Notebook source](../notebooks/kaggle-tpu-v5e8-production-showcase.ipynb) · [How to reproduce](REPRODUCIBILITY.md) · [Evaluation](ACCEPTANCE.md)
