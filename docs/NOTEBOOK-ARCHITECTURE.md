# TPU showcase notebook architecture

🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](NOTEBOOK-ARCHITECTURE.vi.md)

## What a reader sees

The notebook `notebooks/kaggle-tpu-v5e8-production-showcase.ipynb` is an **execution walkthrough**, not a copy of the implementation. Its 17 code cells comprise 16 verified offline inference/Gallery stages plus cell 17 for the optional Cloudflare Quick Tunnel. The adjacent Markdown explains the purpose of each stage in English and Vietnamese.

The user attaches only two Kaggle inputs: the original BF16 model and **Runtime Assets Dataset version 1** (48 files). Project source is cloned anonymously from a public GitHub repository. **Kaggle Internet is required for Git bootstrap**; cell 17 optionally opens Tunnel.

## Source layout

| Location | Responsibility |
|---|---|
| `qwen38_runtime/workflow.py` | Explicit stage order and named state transfer between cells |
| `qwen38_runtime/stages/preflight.py` | Verify model mount, data manifest, 48 data payloads, and output directory |
| `qwen38_runtime/stages/dependencies.py` | Offline wheel and Kaggle TPU runtime compatibility checks |
| `qwen38_runtime/stages/maxtext.py` | Restore SHA256-pinned MaxText and apply the verified Vision patch |
| `qwen38_runtime/stages/inputs.py`, `checkpoint.py` | Verify images and prompts, then check BF16 index metadata and the shard inventory; this is not a per-shard content hash check |
| `qwen38_runtime/stages/tpu.py`, `engine.py` | Initialize the eight TPU devices and load the single BF16 engine |
| `qwen38_runtime/stages/inference_helpers.py` | Tokenization, image preparation, and deterministic generation |
| `qwen38_runtime/stages/ocr_probe.py`, `ocr_evidence.py` | First real OCR inference, AOT diagnostics and durable trace |
| `qwen38_runtime/stages/vision_reuse.py`, `image_cases.py` | Reuse the verified executable for the remaining 15 image cases |
| `qwen38_runtime/stages/text_cases.py` | Run 3 text cases, finalize original 19-case JSON and review flags |
| `qwen38_runtime/stages/vietnamese.py` + `gallery/vi_state.py` | Optional 17 Vietnamese inferences and resume state |
| `qwen38_runtime/stages/gallery.py` + `gallery/render.py` | Export verifiable offline HTML Gallery from actual recorded output |

Cell 17 imports `gallery/quick_tunnel.py` from Git-verified project source; no long embedded Tunnel code remains. The first code cell exposes both Tunnel defaults. It serves only Gallery **in an Interactive Kaggle editor session** (`KAGGLE_KERNEL_RUN_TYPE=Interactive`), with a default 900-second hold. Batch/API/CLI and unknown environments skip Tunnel and have no additional TPU hold. Configure `QWEN38_TUNNEL_HOLD_SECONDS` or disable interactive sharing with `QWEN38_TUNNEL_ENABLED=0`.

The short `text_schedule` stage records the planned text-case sequence; it does not run inference early.

## Git source and offline runtime guarantee

Runtime Assets Dataset **version 1** contains seven verified runtime ZIPs (48 SHA256-pinned payloads), without project Python source. The historic Runtime Support Dataset v1 is retained as evidence and is not the notebook's required source. No weights or TPU outputs are included.

Before importing project modules, the first notebook cell clones Git source and verifies its exact commit SHA. A Git-first preflight checks the repository and invokes the existing integrity gate on the unchanged 48-file manifest before importing JAX.

**Important deployment dependency:** GitHub must be public for anonymous access; attach Runtime Assets Dataset **version 1** plus the BF16 model. No project source archive or GitHub token is required. Verify the exact Git commit and SHA256 hashes before TPU initialization.

## Build and validation

With the **original, verified 48-file legacy Dataset** available locally, run:

```bash
python3 support/package_runtime_assets.py \
  --source /path/to/verified-legacy-dataset-v1 \
  --destination /path/to/new-empty-runtime-assets
```

This command builds seven runtime-only ZIPs locally, without uploading or overwriting Kaggle. Independently check the manifest and all 48 source files before publishing runtime assets.

Before changing the notebook or its modules:

```bash
python3 support/sync_notebook_bootstrap.py --check
python3 support/sync_vi_state_notebook.py --check
python3 support/sync_gallery_notebook.py --check
python3 -m pytest -q tests
```

After changing a Git module, review the diff and rerun sync and tests. Keep the notebook default at `main`; each run records the actual Git commit SHA in source provenance. Pin a historical SHA only when deliberately reproducing that run. Source changes do not require a new Runtime Assets Dataset or model weights.
