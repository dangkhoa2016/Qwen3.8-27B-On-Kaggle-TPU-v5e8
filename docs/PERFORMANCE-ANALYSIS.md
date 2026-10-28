# Qwen3.8-27B BF16 inference performance on Kaggle TPU v5e-8

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](PERFORMANCE-ANALYSIS.vi.md)

This report presents measured execution of Qwen3.8-27B BF16 Text + Vision inference on Kaggle TPU v5e-8 using JAX, MaxText and Split-Vision. It distinguishes end-to-end execution time from per-request latency and generation completion from answer accuracy.

## Measured Full19 session

| Metric | Recorded observation |
|---|---|
| Accelerator | Kaggle TPU v5e-8, eight TPU devices |
| Checkpoint | Qwen3.8-27B original BF16, 18 safetensors shards |
| Runtime | JAX / MaxText, separate Vision encoder and compiled multimodal prefill |
| Workload | 19 cases: 3 Text, 2 numeric OCR, 6 Rich Vision, 8 Photography |
| Execution time | **1,160.844 seconds (19 min 20.844 s)** |
| Generation completion | **19/19 cases reached EOS** |
| Selected checks | Text 3/3; OCR digits 731 and 482; Rich Vision 55/55 keyword checks |

The measurements were recorded on **27 October 2026** during one TPU session. They describe the exact benchmark inputs and software used in that session. They are not an independent run of every subsequent repository revision, and should not be interpreted as proof of 19/19 factual correctness.

## Configuration and stages

| Component | Configuration |
|---|---|
| Checkpoint source for TPU execution | [Kaggle BF16 model mirror](https://www.kaggle.com/models/dangkhoa2016/qwen-qwen3-8-27b) |
| Engine | One MaxEngine; checkpoint loaded once |
| Processed image resolution | 512 × 384 pixels |
| Prefill length | 320 tokens |
| Target length | 768 tokens |
| Rich Vision / Photography decode | Up to 448 steps |
| Input provenance | Case prompts, image SHA256 and runtime manifest |

The wall-clock result includes setup, model loading, compilation, Vision encoding, prefill, decoding and result capture. It is **not** the average latency of an HTTP request or the token generation throughput of a serving system.

Multimodal inference adds image preprocessing and embedding computation before text decoding. Cached JAX executables can reduce compilation costs only when device topology, compiler, shapes, sharding and dtypes are compatible; matching a cached file's SHA256 alone does not validate execution.

## How to benchmark other configurations

1. Record the model revision, source Git SHA, support manifest SHA256, prompts, images, runtime versions and device topology.
2. Separate cold start, checkpoint loading, Vision encoding, prefill, decoding and end-to-end timing.
3. Record generated token counts, EOS and raw text for each case.
4. Keep unsuccessful runs and semantic errors alongside timing records.
5. Use repeated comparable observations before publishing median, p95 or concurrency claims.

## Interpretation and limitations

A completed generation is not necessarily a correct answer. The measured outputs included an invoice tax error (8% / $69.28 read as 5% / $66.28) and an inaccurate comparison of 114 with 31. Keyword checks likewise do not certify complete visual understanding.

**Results are attributable only to the exact input hashes and execution record of the measured run.** The [version-controlled notebook](../notebooks/kaggle-tpu-v5e8-production-showcase.ipynb) supplies source code for new runs; its generated evidence must be collected separately. CPU mock tests do not measure Qwen model speed or TPU HTTP serving.

[Architecture](ARCHITECTURE.md) · [Full19 evidence](FULL19-EVIDENCE.md) · [Reproduction](REPRODUCIBILITY.md)
