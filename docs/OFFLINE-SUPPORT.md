# Organized offline support and integrity validation

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](OFFLINE-SUPPORT.vi.md)

The [Kaggle support Dataset](https://www.kaggle.com/datasets/dangkhoa2016/qwen38-27b-tpu-v5e8-full19-runtime-support) provides the offline dependencies and images needed by the [TPU notebook](../notebooks/kaggle-tpu-v5e8-production-showcase.ipynb). Attach the [original BF16 Kaggle model](https://www.kaggle.com/models/dangkhoa2016/qwen-qwen3-8-27b) separately; the Dataset does not include model weights.

## Dataset layout

Seven ZIP packages contain seven named directories:

| Archive | Extracted folder | Contents |
|---|---|---|
| wheels.zip | wheels/ | Version-pinned Python dependencies |
| source.zip | source/ | Patched MaxText source and patch |
| helpers.zip | helpers/ | JAX layout/sharding and Split-Vision helpers |
| images.zip | images/ | 16 source images |
| cases.zip | cases/ | Input inventory, prompts and case definitions |
| cache.zip | cache/ | Optional JAX compilation cache and manifest |
| runtime.zip | runtime/ | Engine arguments and wheel metadata |

The Dataset root also contains a SHA256 manifest and separate English/Vietnamese guides. Dataset **version 1** additionally has the **8th ZIP** `notebook_source.zip` with **23 Python files** and `notebook-source-manifest.json`. Kaggle may expose those members as `notebook_source/`; the notebook checks both layouts. The source bundle remains separate from seven data ZIPs and 48 SHA256-verified payloads. Inspect the mounted files and manifest before TPU inference.

## Integrity preflight

The central manifest SHA256 is **fb2ee48dfe423733499112668d61e965b5d8d7466d52361b403c68dbc5768817**. The [Git manifest](../support/portable-bundle-manifest.json) records seven ZIP hashes and **48 individual file hashes**.

Before loading weights, the notebook verifies the manifest and all 48 payload files. Kaggle automatically expands ZIP packages into the seven directories listed above; in that mode the notebook verifies file SHA256 in the read-only mount. When supplied with intact ZIP packages instead, it also checks the seven archive sizes, SHA256 values and member lists before secure extraction into a fresh directory under /kaggle/working.

Optional JAX cache files are checked by hash, but reusable compilation still requires compatible TPU topology, tensor shapes, sharding and dtypes.

## Build and distribution

The [version-controlled packager](../support/package_dataset.py) accepts a locally verified flat support directory and writes the organized archives and guides. For example:

    python support/package_dataset.py --source /path/to/flat-support --destination /path/to/organized-dataset

It refuses missing or altered files and does not overwrite an existing nonempty output directory. Third-party wheel/source/image licensing and Kaggle Dataset permissions may limit redistribution. The Git repository does not contain checkpoint weights or the ZIP payloads.

[Execution guide](REPRODUCIBILITY.md) · [Architecture](ARCHITECTURE.md)
