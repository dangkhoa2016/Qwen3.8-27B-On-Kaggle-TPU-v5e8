# Qwen3.8-27B BF16 performance on Kaggle TPU v5e-8

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](PERFORMANCE-ANALYSIS.vi.md)

**Qwen3.8-27B BF16 can execute Text + Vision tasks across eight TPU v5e devices through JAX and MaxText.** This report presents a measured Full19 TPU run, explains multimodal inference costs, and shares a reusable benchmark methodology for other large-model projects.

## Performance at a glance

| Metric | Recorded result |
|---|---|
| Accelerator | Kaggle TPU v5e-8 — 8 TPU devices |
| Checkpoint | Original **Qwen3.8-27B BF16**, 18 safetensors shards |
| Runtime | JAX / MaxText with Split-Vision integration |
| Workload | **19 cases**: 3 Text, 2 numeric OCR, 6 Rich Vision, 8 Photography |
| Total Full19 wall time | **1,160.844 seconds (19 min 20.844 s)** |
| Generation completion | **19/19 cases reached EOS** |
| Selected response checks | Text 3/3; numeric OCR **731** and **482**; Rich Vision **55/55** keyword checks |

These observations come from **one actual Full19 TPU run on 27 October 2026** using the input images associated with that run. The 19/19 result measures generation-path completion; factual accuracy is assessed separately. The total time represents the whole execution session rather than one API request's latency.

## Execution configuration

| Component | Configuration |
|---|---|
| Model | [Kaggle mirror — Qwen3.8-27B BF16](https://www.kaggle.com/models/dangkhoa2016/qwen-qwen3-8-27b) |
| Engine | One `MaxEngine` on 8 TPU devices; one checkpoint load |
| Processed image size | **512 × 384 pixels** |
| Prefill | **320 tokens** |
| Target budget | **768 tokens** |
| Rich Vision / Photography decode | Up to **448 steps** |
| Input identification | Prompts, manifest and image SHA256 |

To verify the configuration, consult the [Full19 notebook source](../notebooks/kaggle-tpu-v5e8-production-showcase.ipynb), [TPU architecture](ARCHITECTURE.md) and [Full19 results](FULL19-EVIDENCE.md). The model is loaded from the attached Kaggle mirror, not directly from Hugging Face.

## Stages that contribute to elapsed time

| Stage | Main operation | Performance relevance |
|---|---|---|
| Preparation | Dependency, input and SHA256 checks | Avoids wasting TPU runtime on invalid inputs |
| Engine startup | JAX/XLA setup, topology and compilation | Significant cold-start overhead |
| Checkpoint loading | Read and distribute BF16 parameters | I/O and device placement |
| Vision encoder | Turn images into embeddings | Additional cost for image requests |
| Multimodal prefill | Fuse text/image tokens and MRoPE | Sensitive to sequence length and shape |
| Decode | Generate tokens until EOS or budget | Sensitive to answer length |
| Result capture | Save output, timings and evidence | Part of end-to-end measurement |

A separate earlier engineering measurement recorded approximately **49.8 seconds for engine construction** and **147.0 seconds for model loading**. They illustrate startup costs but were collected in a different measurement context; they are not a verified breakdown of the Full19 total above.

Similarly, `1,160.844 / 19` is not the mean per-request latency: the 19 tasks have different lengths and costs, and the session total includes preparation.

## Four engineering lessons

### 1. Cold start differs from warm inference

The first invocation may load weights, construct the engine and compile. Once ready, subsequent requests may reuse that state. A useful benchmark distinguishes **startup**, **first inference**, and **warm inference**.

### 2. Vision encoding deserves its own timing

Text + Vision adds preprocessing, image embeddings and fusion before decode. Decoder tokens per second alone cannot describe the full latency of image-based questions.

### 3. Generated length affects speed

Two prompts on the same image may generate very different token counts. Fair comparisons document processed image size, prompt tokens, token budget, generated count and EOS stopping conditions.

### 4. Quality belongs next to performance

Full19 completed 19 generation paths and passed the recorded keyword checks, yet included an invoice tax reading error and a dashboard ratio reasoning error. Useful measurements preserve **timings, unmodified responses and semantic evaluation** together.

## A repeatable benchmarking method

1. **Pin the environment:** Git SHA, Kaggle Model revision, manifest SHA256, JAX/MaxText versions, TPU topology and compilation-cache policy.
2. **Keep the workload stable:** record prompts, image SHA256, processed dimensions, prefill length and decode budget for every case.
3. **Time each stage:** engine startup, checkpoint loading, vision encoding, prefill, decode and end-to-end.
4. **Separate cold and warm:** collect repeated independent runs with sample counts; retain initialization costs for cold-start reports.
5. **Report appropriate statistics:** median/p95 require multiple comparable observations and an explicit sample size.
6. **Preserve full evidence:** token IDs, generated text, EOS, timings, exceptions and quality rubric, including unsuccessful cases.

With adequate logs, the following metrics make configurations easier to compare:

| Metric | Definition |
|---|---|
| End-to-end latency | Completion time minus request start time, using an explicit request boundary |
| Decode throughput | Generated tokens / corresponding decode duration |
| System throughput | Completed requests or tokens / observation time, with concurrency stated |
| p50/p95 latency | Percentiles from repeated comparable runs |

## Applying the results

The numbers above describe **one Full19 run on real TPU hardware**. Evaluating a user-facing serving system additionally calls for multi-request HTTP tests, concurrency, latency distributions and TPU utilization. The repository's CPU Playground is a mock API, not a BF16 performance benchmark.

A new green-signal pedestrian crossing image replaced an earlier red-signal source in the Git examples. Their SHA256 values differ, so the older response and timing remain associated with the older image; the new image requires its matching manifest and results.

**Explore further:** [Architecture](ARCHITECTURE.md) · [Full19 results](FULL19-EVIDENCE.md) · [Evaluation](ACCEPTANCE.md) · [Reproduction](REPRODUCIBILITY.md)
