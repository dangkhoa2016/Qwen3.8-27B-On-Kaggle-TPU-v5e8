# TPU showcase notebook architecture

🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](NOTEBOOK-ARCHITECTURE.vi.md)

## What a reader sees

The notebook `notebooks/kaggle-tpu-v5e8-production-showcase.ipynb` is an **execution walkthrough**, not a copy of the implementation. Its 17 code cells comprise 16 verified offline inference/Gallery stages plus cell 17 for the optional Cloudflare Quick Tunnel. The adjacent Markdown explains the purpose of each stage in English and Vietnamese.

The user still attaches only two Kaggle inputs: the original BF16 model and the **version 1** Runtime Support Dataset. No Git clone, source-code paste, or third input is required. **Internet is needed only for the optional Tunnel in cell 17.**

## Source layout

| Location | Responsibility |
|---|---|
| `qwen38_runtime/workflow.py` | Explicit stage order and named state transfer between cells |
| `qwen38_runtime/stages/preflight.py` | Verify model mount, data manifest, 48 data payloads, and output directory |
| `qwen38_runtime/stages/dependencies.py` | Offline wheel and Kaggle TPU runtime compatibility checks |
| `qwen38_runtime/stages/maxtext.py` | Restore SHA256-pinned MaxText and apply the verified Vision patch |
| `qwen38_runtime/stages/inputs.py`, `checkpoint.py` | Verify images, prompts, and BF16 shard metadata |
| `qwen38_runtime/stages/tpu.py`, `engine.py` | Initialize the eight TPU devices and load the single BF16 engine |
| `qwen38_runtime/stages/inference_helpers.py` | Tokenization, image preparation, and deterministic generation |
| `qwen38_runtime/stages/ocr_probe.py`, `ocr_evidence.py` | First real OCR inference, AOT diagnostics and durable trace |
| `qwen38_runtime/stages/vision_reuse.py`, `image_cases.py` | Reuse the verified executable for the remaining 15 image cases |
| `qwen38_runtime/stages/text_cases.py` | Run 3 text cases, finalize original 19-case JSON and review flags |
| `qwen38_runtime/stages/vietnamese.py` + `gallery/vi_state.py` | Optional 17 Vietnamese inferences and resume state |
| `qwen38_runtime/stages/gallery.py` + `gallery/render.py` | Export verifiable offline HTML Gallery from actual recorded output |

Cell 17 imports `gallery/quick_tunnel.py` from the verified source bundle in Dataset v1; no long embedded Tunnel code remains. The first code cell exposes both Tunnel defaults. It serves only Gallery **in an Interactive Kaggle editor session** (`KAGGLE_KERNEL_RUN_TYPE=Interactive`), with a default 900-second hold. Batch/API/CLI and unknown environments skip Tunnel and have no additional TPU hold. Configure `QWEN38_TUNNEL_HOLD_SECONDS` or disable interactive sharing with `QWEN38_TUNNEL_ENABLED=0`.

The short `text_schedule` stage records the planned text-case sequence; it does not run inference early.

## Kaggle's offline source guarantee

Runtime Support Dataset **version 1** contains seven verified data ZIPs (48 SHA256-pinned payloads) and the eighth source ZIP `notebook_source.zip` (23 Python modules), plus `notebook-source-manifest.json`. Kaggle may expose the source ZIP as the extracted `notebook_source/` directory. No weights, TPU outputs or fabricated results are included.

Before importing project modules, the first notebook cell checks the pinned SHA256 of the source manifest, validates either the source ZIP or Kaggle-extracted directory member-by-member, and rejects unsafe ZIP paths and file types. The first loaded stage then checks the original Runtime Support manifest and 48 data files before importing JAX.

**Important deployment dependency:** Attach Runtime Support Dataset **version 1**, with `notebook-source-manifest.json` and either `notebook_source.zip` or `notebook_source/` containing 23 verified Python modules. Its SHA256 must match the notebook pin. Verify attached files before starting TPU; the 48 verified data payloads remain separate.

## Build and validation

With the **original, verified flat 48-file distribution** available locally, run:

```bash
python3 support/package_dataset.py \
  --source /path/to/verified-flat-support \
  --destination /path/to/new-empty-kaggle-dataset
```

This command creates files locally and does not upload or overwrite any Kaggle Dataset. Inspect both manifests, eight ZIP files and all 48 original payloads before distributing a compatible support bundle.

Before changing the notebook or its modules:

```bash
python3 support/sync_notebook_bootstrap.py --check
python3 support/sync_vi_state_notebook.py --check
python3 support/sync_gallery_notebook.py --check
python3 -m pytest -q tests
```

After changing any bundled module, review the diff and update the source pin with `python3 support/sync_notebook_bootstrap.py --write`, then rerun the checks. Source synchronization checks do not modify the deployed Kaggle Dataset or model weights.
