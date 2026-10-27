# Visual dataset quality: one crossing scene, two different inputs

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](RELEASE-IMAGE-REVIEW.vi.md)

A benchmark example is only reproducible when its **exact image bytes** are identified. This crossing case illustrates how improving an image can also change which model results are valid for it.

## Compare the two sources

| Input | Observed content | SHA256 |
|---|---|---|
| Earlier experimental source | Red pedestrian signal while people cross | `bba573de952942ec97ea6a32c352bdbd3770082b570cfe74520b1cd03b087763` |
| Current Git source | Green pedestrian signal with pedestrians crossing | `ed532c2ba4e7eace4a81e4adfd6867c68f244fe4d64293f2bbfd1097ad7d0356` |

The new 1280 × 960 Git image was resized with **Pillow Lanczos** from a 1448 × 1086 supplied PNG (SHA256 `63a61fd4ed5788ca156313b0ee7b2335e898b2e7ed7f683a725bb1d836c5395d`). The original generator model, seed and prompt **cannot be verified from the PNG alone**.

## Why the distinction matters

The archived Qwen response to the **red-signal** picture correctly warned about the pedestrian/vehicle conflict. It was not a confirmed model reasoning failure. The **green-signal** picture is a different input: the earlier response and its measured case duration must **not** be reused as evidence for the replacement.

The [image manifest](assets_manifest.json) identifies the new Git input. The historical support snapshot remains associated with its own output; a separate replacement-image snapshot passed **48/48 SHA256** after independent download.

Checksum verification covers the image bytes, **not** Qwen inference. Use the [version-controlled notebook source](../notebooks/kaggle-tpu-v5e8-production-showcase.ipynb) and aligned manifests as a starting point; never reuse the old response for the new image.

## A portable image-review checklist

1. Confirm dimensions and content at original resolution, not only in a thumbnail.
2. Check traffic signals, anatomy, geometry and object relations for internal consistency.
3. Preserve the exact image hash and record the transformation from a source file.
4. Associate every prompt, raw model response and token sequence with its input hash.
5. When an input changes, run inference again rather than borrowing earlier results.
6. Retain the old evidence to make before/after comparisons auditable.

This process is applicable to any synthetic-image, OCR or multimodal benchmark—not only traffic scenes.

[Photo case definitions](photo_showcase_cases.json) · [Reproducibility](../docs/REPRODUCIBILITY.md) · [Evaluation](../docs/ACCEPTANCE.md)
