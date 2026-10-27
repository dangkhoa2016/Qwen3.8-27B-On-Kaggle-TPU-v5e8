# Full19 TPU evidence inventory and quality notes — 27 October 2026

> **English** | [Tiếng Việt](FULL19-EVIDENCE.vi.md)

## Exact provenance: private experimental acceptance

| Item | Verifiable fact |
|---|---|
| Model | Qwen3.8-27B original BF16, 18 safetensors shards; 1,199 index tensors |
| Platform | Kaggle TPU v5e-8 × 8; JAX/JAXLIB 0.11.0 |
| Execution | One MaxEngine, native XLA sharding/layout verification retained |
| Source | Private Split-Vision AOT candidate; not the older Git-tracked notebook |
| Input | 16 checked image assets; 3 text cases; 512×384 inference resize |
| Prefill / target / output limit | 320 / 768 / 448 for Rich Vision and Photography |
| Automated run | COMPLETE, 14/14 executed code cells, 19/19 distinct generated responses with EOS |
| Groups | 3 Text, 2 numeric OCR, 6 Rich Vision, 8 Photography |
| Diagnostics | 5/5 Text/OCR semantic checks; Rich Vision 55/55 keyword checks |
| Runtime | **1,160.844 seconds** (19 min 20.844 s) |
| User review | Other examples judged acceptable; known defects disclosed below |
| Real HTTP TPU serving | Separate feature, not validated |

The SHA256-verified archives are stored in the current Kaggle working directory or separately archived storage; **these filesystem paths are not public download URLs**.

### Full19 executed evidence archive

`/kaggle/working/Qwen38-Full19-SplitVision-Extended448-TPU-AutomatedPass-QA-Pending-Evidence-20261027.zip`

SHA256 `3449356a47a6c72f43d5279649e3d35af3585f8c10044588563b44be1ed2c06a`

Contains executed and source notebooks, per-case raw text and token IDs, 16 original input images, checksums, timing, test statuses, logs and local HTML visual review. A visual review is not a statistically representative model-accuracy benchmark.

### Two repeated invoice-only runs

`/kaggle/working/Qwen38-TPU-Invoice-Tax-Two-Reruns-Evidence-20261027.zip`

SHA256 `781d9f3060380da8d5bd2fc1514389018cbd81bdf10ef4387ded240d241d15d4`

Single BF16 engine/checkpoint load; two separate Vision/AOT prefill and greedy decode calls with the **same original image and prompt**. Each yielded 171 generated tokens and EOS; token IDs were identical. Both reported `Tax (5%): $66.28` instead of the source invoice's `Tax (8%): $69.28`; subtotal `$866.00` and total `$935.28` were correctly preserved. The independent validator v2 fixes a regex issue in an earlier evaluator that had matched “Total” within “Subtotal”; the original model outputs were never edited. These are **deterministic repeats**, not independent random trials.

### Other confirmed limitation

Delivery Dashboard model output compares 114 Saturday deliveries against 31 Monday deliveries as “nearly double.” The actual ratio is 114/31 ≈ 3.68. Preserve this sentence in raw evidence; do not silently correct generated text.

## Publication boundary

Automated engineering acceptance **PASS** is not equivalent to perfect factual/visual correctness or a completed fresh Run All of the public notebook. A public release still requires portable 38-file support packaging, a reachable examples source, the updated 320/768 Split-Vision notebook, an actual fresh execution, seed/provenance qualification and separate HTTP TPU acceptance if that service is claimed. Avoid stating the current evidence proves any of these pending items.
