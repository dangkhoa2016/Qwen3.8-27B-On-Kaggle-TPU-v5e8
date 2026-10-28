# Offline support Dataset and integrity verification

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](OFFLINE-SUPPORT.vi.md)

The TPU notebook uses separate Kaggle attachments for the **BF16 model weights** and the **Full19 runtime support Dataset**. The support Dataset contains dependencies, MaxText integration files, image inputs, case definitions and optional compilation-cache files, but **does not include model weights**.

## Required resources

| Resource | Purpose |
|---|---|
| [Kaggle Qwen3.8-27B BF16 Model](https://www.kaggle.com/models/dangkhoa2016/qwen-qwen3-8-27b) | Checkpoint loaded during TPU inference |
| [Kaggle Full19 support Dataset](https://www.kaggle.com/datasets/dangkhoa2016/qwen38-27b-tpu-v5e8-full19-runtime-support) | Offline runtime files and 16 input images |
| [Version-controlled notebook](../notebooks/kaggle-tpu-v5e8-production-showcase.ipynb) | Inference implementation and evidence capture |

The verified support manifest lists **48 files**. The checksum of `portable-bundle-manifest.json` is:

```text
cf5343f614eb32ff9234c870e4a26093f9f07e62199ee90396d516a804eb8a28
```

The notebook checks this checksum and then validates the size and SHA256 of every listed file **before** loading model weights. The Git copy is available at [support/portable-bundle-manifest.json](../support/portable-bundle-manifest.json).

## Using the Dataset

1. In Kaggle, select **TPU v5e-8** and attach both resources above.
2. Run the Git notebook. It locates the Dataset in Kaggle's mount directory and checks the included `portable-bundle-manifest.json`.
3. For a different mount layout, set `QWEN38_SUPPORT_DIR` to the directory containing the manifest.
4. Stop on a checksum mismatch; do not suppress the integrity checks.
5. Collect new output files and timings under `/kaggle/working`.

Compatibility also depends on the installed JAX/MaxText stack, TPU topology and executable cache metadata. A matching manifest is a prerequisite, **not** a guarantee that compilation or inference will succeed on a different runtime.

## Availability and licensing

Model weights, third-party wheels, patched upstream code, images and cached binaries may have separate license or access conditions. A Git checkout alone does not include the attached Kaggle resources. Verify that your Kaggle account can access both attachments and review [third-party notices](../THIRD_PARTY_NOTICES.md) before redistribution.

[Execution guide](REPRODUCIBILITY.md) · [Architecture](ARCHITECTURE.md) · [Image manifest](../examples/assets_manifest.json)
