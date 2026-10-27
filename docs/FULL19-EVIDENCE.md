# Full19 results: what the TPU experiment actually showed

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](FULL19-EVIDENCE.vi.md)

Full19 is a **19-prompt Text + Vision exercise** designed to make a large multimodal TPU run observable from input through generation. It is a selected-case demonstration, **not a representative accuracy benchmark**.

## Experimental run recorded on 27 October 2026

| Attribute | Recorded observation |
|---|---|
| Checkpoint | Qwen3.8-27B BF16; 18 safetensors shards, 1,199 tensor entries |
| Compute | TPU v5e-8; eight devices, one MaxEngine |
| Runtime | Split-Vision JAX/MaxText; JAX/JAXLIB 0.11.0 |
| Image preprocessing | Source images converted to 512 × 384 |
| Generation budget | 320 prefill / 768 max target / Rich+Photo up to 448 decode |
| Test cases | 3 Text, 2 numeric OCR, 6 Rich Vision, 8 Photography |
| Program completion | **19/19 distinct cases with EOS** |
| Targeted checks | 3/3 Text, OCR **731** and **482**, 55/55 Rich keywords |
| Measured wall time | **1,160.844 s (19 min 20.844 s)** |

Those results come from an executed **experimental runner** with the original input hashes; they are **not** results from a fresh Run All of the Git notebook with the later replacement crossing image. Some answers contained factual mistakes despite automated completion.

## How to read the evidence

For each case a useful record contains the unmodified prompt and source image SHA256, full generated text, generated token IDs, EOS status, code revision, device topology and time. For Vision, record the image preprocessing size and multimodal offset as well.

The original evidence archives were saved on the Kaggle host with these checksums:

| Evidence archive | SHA256 |
|---|---|
| Full19 executed notebook, case results and review materials | `3449356a47a6c72f43d5279649e3d35af3585f8c10044588563b44be1ed2c06a` |
| Invoice targeted reruns | `781d9f3060380da8d5bd2fc1514389018cbd81bdf10ef4387ded240d241d15d4` |

These are **archived artifacts, not public download links**. The hashes allow verification when the corresponding archives are made available with appropriate rights. The Git repository itself contains the input manifests, notebook source, evaluation definitions and documented results—not the executed experimental ZIP archives.

## Interesting observations

**Numeric OCR:** the two simple digit cards returned **731** and **482**. This is a constrained OCR check, not proof of generalized handwritten or invoice OCR.

**Rich Vision:** six annotated scenes satisfied 55 keyword checks, but keyword presence alone cannot verify every claim in a response.

**Invoice:** the source states **8% tax / $69.28**, while the model said **5% / $66.28**. Two additional deterministic greedy calls reproduced it.

**Dashboard:** the model called 114 versus 31 “nearly double” rather than roughly **3.68×**.

**Pedestrian crossing:** experimental output corresponds to the earlier red-signal input; the green-signal image currently in Git and Private Release Candidate Dataset has a different SHA256. The older answer must not be associated with the replacement.

## Reproducing and extending this exercise

The [version-controlled Full19 notebook](../notebooks/kaggle-tpu-v5e8-production-showcase.ipynb) has **19 cells / 14 code cells** and defines the Split-Vision AOT pipeline. For reproducible inference, attach the Kaggle BF16 mirror and the support files matching the notebook-pinned manifest. No new Full19 result for the replacement crossing image is established by the earlier experimental evidence.

Use the [reproduction guide](REPRODUCIBILITY.md) to run the notebook, [evaluation guide](ACCEPTANCE.md) to interpret its results and [failure lessons](FAILURE-ANALYSIS.md) to audit semantic correctness. Do not report old measurements as if produced by a new Dataset.
