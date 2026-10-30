# Reproduce Qwen3.8-27B BF16 Text + Vision inference on Kaggle TPU v5e-8

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](REPRODUCIBILITY.vi.md)

This guide uses the [version-controlled Full19 notebook](../notebooks/kaggle-tpu-v5e8-production-showcase.ipynb) and its matching Kaggle attachments. It covers 19 inference cases on eight TPU v5e devices. CPU tests and the mock web UI are available separately; they do not run the 27B model.

## Requirements

- Kaggle Notebook with **TPU v5e-8**, eight TPU devices.
- [Qwen3.8-27B original BF16 Kaggle Model](https://www.kaggle.com/models/dangkhoa2016/qwen-qwen3-8-27b) — the attached execution checkpoint, Transformers/default/v1.
- [Full19 support Dataset](https://www.kaggle.com/datasets/dangkhoa2016/qwen38-27b-tpu-v5e8-full19-runtime-support) — 48-file data manifest, images and offline runtime files. Attach **version 1** with `notebook-source-manifest.json` and either `notebook_source.zip` or the Kaggle-extracted `notebook_source/` directory (23 Python files), matching the notebook's pinned SHA256. **Access:** verify availability, permissions and files in your Kaggle account before starting TPU.
- Pinned runtime requirement: **Python 3.12, JAX/JAXLIB 0.11.0, Flax 0.12.8 and Transformers 5.12.1**, eight TPU v5e devices and the exact 48-file manifest SHA256. Other runtime versions are not claimed compatible.

Manifest SHA256: `fb2ee48dfe423733499112668d61e965b5d8d7466d52361b403c68dbc5768817`.

## Run the TPU notebook

1. Import [the notebook from Git](../notebooks/kaggle-tpu-v5e8-production-showcase.ipynb) into Kaggle, select TPU v5e-8, and enable Internet for optional Gallery sharing in cell 17.
2. Attach the Kaggle BF16 Model and Full19 support Dataset above. Access to either attachment is subject to Kaggle availability and permissions.
3. Run the cells in order without restarting the TPU kernel. The first preflight checks the model attachment, central manifest and **48 individual SHA256 file hashes**. It supports Kaggle-extracted category folders and verified ZIP packages.
4. Let the notebook create its own UTC-stamped output directory beneath `/kaggle/working`.
5. Review raw responses, token IDs, EOS, quality flags, errors and timings; distinguish collection of 19 cases from EOS completion.

The reference uses one MaxEngine, a single checkpoint load, 512 × 384 processed images, 320-token prefill, 768-token target and up to 448 decode steps for Rich Vision and Photography. A valid cache file does not bypass JAX/XLA native layout checks.

## Optional Vietnamese inference and static HTML gallery

The 19 English cases are saved first. The notebook then runs 17 Vietnamese inferences by default (3 Text, 6 Rich Vision, 8 Photography); each numeric OCR card reuses its own digit result for VI display (no extra OCR inference). Set QWEN38_ENABLE_VI=0 before Run All to skip Vietnamese inference. Evaluate completion and Vietnamese-answer quality from the final notebook output. Questions stay in English with a Vietnamese-only response instruction appended; this is not a native-Vietnamese-prompt benchmark. English keyword checks do not validate Vietnamese replies.

HTML export is on by default (QWEN38_RENDER_HTML=1), runs on CPU after inference, and creates inference-gallery.zip. **Cell 17** can open a temporary Cloudflare HTTPS link directly from Kaggle (Internet On). Only an Interactive Kaggle editor session launches Tunnel and can hold it for 900 seconds by default. API/CLI and Save & Run All batch runs skip the Tunnel automatically, without extra TPU quota for link sharing; download the saved Gallery ZIP after execution. **Alternatively**, download the ZIP and host it on a separate CPU computer after stopping TPU. Adjust Tunnel defaults in the first code cell; see the [Gallery guide](../gallery/README.md).

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

## If Kaggle changes its runtime

Run the initial manifest and dependency checks before TPU-heavy cells. On `BASELINE_RUNTIME_MISMATCH`, stop and record installed versions. Do not pip-replace JAX/JAXLIB in an active TPU session or bypass SHA256. Rebuild and separately qualify a compatible support bundle before claiming reproducibility.

## Troubleshooting

| Symptom | Check |
|---|---|
| Model unavailable | Kaggle model attachment, checkpoint index, 18 shards |
| Support manifest mismatch | Dataset mount, extracted folder, SHA256 |
| Wrong TPU topology | Accelerator selection, JAX device count |
| AOT compilation issue | Runtime versions, tensor shapes, sharding, dtype and cache compatibility |
| No EOS or wrong output | Token budget, prompt, exact image digest and raw response |

[Support Dataset](OFFLINE-SUPPORT.md) · [Evaluation](ACCEPTANCE.md) · [Performance](PERFORMANCE-ANALYSIS.md)


## CPU-only Gallery replay and source parity

The canonical static renderer is gallery/render.py. The notebook calls
the verified offline module via workflow.run('gallery') in cell 16; use support/sync_gallery_notebook.py --check
in CI and --write to refresh the stage call and source pin.

HTML, JavaScript, CSS and optional SHA256-bound annotations can be rebuilt
from a completed run's EN/VI JSON plus the 16 original source images, with
no TPU/GPU inference. CPU Gallery rendering does not execute the TPU model. Verify the EN/VI SHA256, exact 16 image hashes, resulting
evidence.json and browser tests before sharing any Gallery.

Human-review annotations must match the SHA256 of the corresponding
inference outputs. Cases without manual review must remain unreviewed.
