# Offline runtime resources and dataset integrity

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](OFFLINE-SUPPORT.vi.md)

Large TPU experiments are easier to reproduce when model weights, runtime dependencies and image inputs are **separate, versioned resources**. This project uses a Kaggle Model attachment for BF16 weights and a checksum-pinned Dataset for offline support.

## What the support bundle contains

| Resource | Why it is included |
|---|---|
| Patched MaxText archive and patch source | Reproduce the specific Split-Vision integration |
| Versioned Python wheels | Install in restricted/offline Kaggle environments |
| JAX helper modules | Prepare and verify the AOT/prefill path |
| Six Rich Vision images + eight Photography images + two OCR cards | Supply the 16 image test inputs |
| Input/case manifests | Associate each prompt with the correct image and SHA256 |
| Optional compiled cache artifacts | Candidate acceleration, subject to native compatibility checks |

There are **48 manifest-indexed content files**, plus the manifest. BF16 model weights are **not inside the Dataset**.

## Two documented input versions

| Input snapshot | Image set | Manifest SHA256 | Verified status |
|---|---|---|---|
| Original Full19 input snapshot | Red-signal crossing source | `1447ee114dc2d2abce6f60f5125927f55311bf530b8cf53ab86a2576ac637ebc` | 48/48 verified in the recorded support collection |
| Replacement-image input snapshot | Green-signal crossing source | `87d172b3b3e2fb4b478888665da164e6abe7cdff3f566b4699d396c919a2ec08` | 48/48 verified after independent download; separate TPU evidence required |

The updated crossing PNG has SHA256 `ed532c2ba4e7eace4a81e4adfd6867c68f244fe4d64293f2bbfd1097ad7d0356`. Do not substitute it into a notebook that expects the old bundle manifest; the checksum guard is intended to catch this mismatch.

## How Kaggle attaches the files

Kaggle may automatically expand the uploaded ZIP into a named child directory. The notebook searches for `portable-bundle-manifest.json` at supported mount roots and within the expected extracted folder. For custom layouts set `QWEN38_SUPPORT_DIR` to the actual support directory.

A zipped support bundle may unpack into its own named directory. Attach the files expected by the **maintained Git notebook** and check their manifest SHA256 before Run All. Do not treat any transient dataset name or ZIP folder name as a permanent public API.

The first code cell performs **fail-closed verification** of file size, digest, allowed relative paths, manifest format and required file count. It runs **before loading model weights**. If a file is absent or modified, stop rather than bypass the check.

## Why the bundle remains Private

The archive contains redistributable and potentially differently licensed assets: third-party wheels, modified upstream MaxText, images and optional cache binaries. Private availability does **not** mean readers can access or legally redistribute all contents. A public package needs component-by-component license and provenance review.

The documented input snapshots are **not training data** and **not production benchmark outputs**. Historic Full19 results remain associated with original image hashes. Checksum verification for replacement images is not a substitute for a new TPU run.

## Making your own verified bundle

1. Inventory required files and remove unnecessary secrets, user data and transient artifacts.
2. Preserve model weights separately and record the exact model revision.
3. Hash every distributed file; include dimensions for image files.
4. Build an immutable ZIP and verify it after extraction and after downloading from the hosting provider.
5. Pin the manifest hash in the notebook and fail closed on any mismatch.
6. Run the exact notebook on target TPU hardware before reporting runtime metrics.
7. Publish a clear dependency/license inventory alongside the artifacts if redistribution is permitted.

[Run guide](REPRODUCIBILITY.md) · [Model licenses](../MODEL_LICENSE.md) · [Evaluation](ACCEPTANCE.md)
