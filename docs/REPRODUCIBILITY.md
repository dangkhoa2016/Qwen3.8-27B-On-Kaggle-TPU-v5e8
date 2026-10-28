# Reproduce Qwen3.8-27B BF16 Text + Vision inference on Kaggle TPU v5e-8

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](REPRODUCIBILITY.vi.md)

This guide uses the [version-controlled Full19 notebook](../notebooks/kaggle-tpu-v5e8-production-showcase.ipynb) and its matching Kaggle attachments. It covers 19 inference cases on eight TPU v5e devices. CPU tests and the mock web UI are available separately; they do not run the 27B model.

## Requirements

- Kaggle Notebook with **TPU v5e-8**, eight TPU devices.
- [Qwen3.8-27B original BF16 Kaggle Model](https://www.kaggle.com/models/dangkhoa2016/qwen-qwen3-8-27b) — the attached execution checkpoint, Transformers/default/v1.
- [Full19 support Dataset](https://www.kaggle.com/datasets/dangkhoa2016/qwen38-27b-tpu-v5e8-full19-runtime-support) — 48-file manifest, input images and offline runtime files.
- Compatible pinned runtime including JAX/JAXLIB 0.11.0 and Transformers 5.12.1.

Manifest SHA256: `cf5343f614eb32ff9234c870e4a26093f9f07e62199ee90396d516a804eb8a28`.

## Run the TPU notebook

1. Import [the notebook from Git](../notebooks/kaggle-tpu-v5e8-production-showcase.ipynb) into Kaggle and select TPU v5e-8.
2. Attach the Kaggle BF16 Model and Full19 support Dataset above. Access to either attachment is subject to Kaggle availability and permissions.
3. Run the cells in order without restarting the TPU kernel. The first preflight checks model presence and the support manifest plus **48 individual file checksums**.
4. Let the notebook create its own UTC-stamped output directory beneath `/kaggle/working`.
5. Inspect full responses, token IDs, EOS markers, error records and per-case timings before claiming the run completed.

The reference uses one MaxEngine, a single checkpoint load, 512 × 384 processed images, 320-token prefill, 768-token target and up to 448 decode steps for Rich Vision and Photography. A valid cache file does not bypass JAX/XLA native layout checks.

**Execution evidence:** The measured 27 October 2026 Full19 session completed 19/19 generation cases with EOS in 1,160.844 seconds, but its input hashes and software belong to that recorded session. A new run of this notebook must produce its own outputs; do not copy benchmark figures into the current run. Generation completion is not the same as factual correctness.

## Validate the CPU-only components

From the repository root:

```bash
python -m pip install -r requirements-cpu.txt
python -m pytest -q tests
npm ci
npx playwright install chromium
npm run test:e2e
```

These checks cover API contracts, integrity guards and the CPU mock frontend, **not** TPU model execution.

## Troubleshooting

| Symptom | Check |
|---|---|
| Model unavailable | Kaggle model attachment, checkpoint index, 18 shards |
| Support manifest mismatch | Dataset mount, extracted folder, SHA256 |
| Wrong TPU topology | Accelerator selection, JAX device count |
| AOT compilation issue | Runtime versions, tensor shapes, sharding, dtype and cache compatibility |
| No EOS or wrong output | Token budget, prompt, exact image digest and raw response |

[Support Dataset](OFFLINE-SUPPORT.md) · [Evaluation](ACCEPTANCE.md) · [Performance](PERFORMANCE-ANALYSIS.md)
