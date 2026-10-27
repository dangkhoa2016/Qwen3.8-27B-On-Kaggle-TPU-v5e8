# Acceptance criteria and verified results

> **English** | [Tiếng Việt](ACCEPTANCE.vi.md)

## Verified experimental TPU Full19 result — 27 October 2026

The original **Qwen3.8-27B BF16** model (18 safetensors shards, no quantization) completed one measured Full19 inference on **8 Kaggle TPU v5e devices**, using one MaxEngine and a Split-Vision JAX/MaxText AOT path. The executed candidate and evidence were preserved separately from the Git-tracked notebook.

| Gate | Observed result |
|---|---|
| Runner and executed notebook | COMPLETE, 14/14 code cells |
| Total wall time | **1,160.844 seconds (19 min 20.844 sec)** |
| Text | **3/3**, semantic checks and EOS |
| Numeric OCR cards | **2/2**, correct 731 and 482 with EOS |
| Rich Vision | **6/6**, EOS; 55/55 pre-defined keyword checks |
| Photography | **8/8**, EOS |
| All generated cases | **19/19**, distinct and EOS |
| Hardware / checkpoint | TPU v5e-8, original BF16 27B, 18 shards, one engine |
| Fresh public Git-tracked notebook Run All | **NOT YET VERIFIED** |
| Real TPU HTTP API end-to-end | **NOT QUALIFIED** |
| Unqualified general VLM accuracy claim | **NOT SUPPORTED** |

These results establish real execution, not perfect OCR, arithmetic or visual understanding. The user reviewed the remaining examples as satisfactory overall, but quality issues must be preserved. The Delivery Dashboard response describes 114 versus 31 deliveries as “nearly double,” whereas the ratio is about 3.68. The office invoice should show **Tax (8%): $69.28** on $866.00 subtotal and $935.28 total; the model instead reported **Tax (5%): $66.28**. Two targeted reruns using the identical invoice and greedy prompt reproduced that same tax error, both with EOS and identical generated token IDs. Greedy repetitions are deterministic; they are not independent random samples.

Detailed manifest, archive SHA256 values and limitations: [Full19 evidence inventory](FULL19-EVIDENCE.md). Keep the raw model text and images unchanged; translations for the review website are illustrative, not original model-generated Vietnamese responses.

## Required gates before the first official production notebook Run All

1. Replace the Git-tracked outdated 256/384 direct-prefill notebook with one coherent **320/768 Split-Vision AOT** notebook derived from the verified candidate, including 448-token limits for Rich/Photography and correctly bounded OCR/Text paths.
2. Make all model, 38-file support and 16 image paths usable from **fresh Kaggle-attached resources**. The current evidence used private files in /kaggle/working; their existence in the next clean session is not guaranteed. Avoid anonymous cloning of a Private GitHub repository.
3. Independently verify file SHA256, JAX/JAXLIB/Transformers versions, one BF16 load and one engine, native sharding/layout correctness checks and cache compatibility. Reject unverified cache entries.
4. Execute a **fresh notebook** with real token IDs, output text, EOS, per-case and overall timings, concrete device count and status. Preserve failures and partial output; do not claim PASS before measurement.
5. Verify Python CPU contracts and browser E2E in a separate isolated test environment. A CPU PASS does not qualify the experimental HTTP TPU backend.
6. Review outputs and residual quality findings before publication. The private API must not be exposed through a public tunnel before a separate end-to-end HTTP/security acceptance.

See [reproducibility](REPRODUCIBILITY.md). A fresh public production release is still pending.
