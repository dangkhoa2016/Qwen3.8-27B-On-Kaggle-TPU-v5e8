# Reproducibility and clean Kaggle execution

> **English** | [Tiếng Việt](REPRODUCIBILITY.vi.md)

## Scope of the verified experiment

The **19/19 automated Full19 result** from 27 October 2026 was recorded on a real Kaggle TPU v5e-8 in 1,160.844 seconds using a **private experimental candidate**. It is **not** a newly verified Run All of the Git-tracked notebook. Keep these two milestones separate. See [acceptance](ACCEPTANCE.md) and [evidence](FULL19-EVIDENCE.md).

## Pre-run checklist — no new production run until all PASS

1. **Checkpoint:** attach the model mirror `dangkhoa2016/qwen-qwen3-8-27b`, the correct Transformers/default/v1 snapshot; verify index metadata, 1,199 tensor keys, 18 BF16 safetensors shards and expected 55,562,855,904 declared bytes before construction.
2. **Support:** attach the compatible offline dataset as an actual Kaggle input resource. The approved test used 38 SHA256-verified files from `/kaggle/working/qwen38-production-rc-20261025/private-support-v3`. That path is **session-local**; it must not be assumed available in a fresh notebook. Verify every support digest and model/runtime configuration, not only filenames.
3. **Inference assets:** use the 16 images in [examples](../examples/README.md), validate [SHA256](../examples/assets_manifest.json), and use 6 rich / 8 photo case manifests. During inference, resize originals to 512×384 as in the accepted path. These input images are **not Qwen output images**.
4. **Source and deps:** pin a reachable immutable Git reference and verify the patched MaxText source/patch; use JAX/JAXLIB 0.11.0 and Transformers 5.12.1 as qualified. A private repository's anonymous `git clone` cannot be assumed to work. Verify authentication or package inputs without network dependency.
5. **Memory and topology:** require exactly 8 TPU devices, original BF16 (not GGUF/INT8), one MaxEngine and one 18-shard checkpoint load. Do not restart or stop a long-queued TPU session without user approval.
6. **Compilation and inference:** preserve all native JAX sharding and layout getters/checks. The accepted candidate uses Split-Vision prefill 320, maximum target 768, a correctly verified AOT executable and greedy decoding. Rich/Photography generate up to 448 tokens. Historical persistent cache entries are optional and **not guaranteed portable** to another host/JAX shape.
7. **Evidence:** record source revision, model/support/input SHA256, token IDs, original generated text, true EOS, initial compilation and first-generation timings, all 19 per-case timings, RNG/seed provenance and exactly one total wall-clock measurement. Do not assert an independently attested seed when only the MaxEngine RNG split is recorded.
8. **Review:** retain all failures, compare against original images, disclose known dashboard arithmetic and invoice tax errors, and separately qualify any real HTTP TPU service.

## Current blockers

- The Git-tracked notebook has been replaced with a 320/768 Split-Vision source; the current candidate has not passed a fresh TPU Run All.
- The 48-entry portable offline-support bundle has been prepared and SHA256-verified (including the original 38 support files), but a **new Kaggle Dataset attachment and mount verification** are still pending.
- GitHub is Private. The **new notebook no longer clones this repository**; it consumes verified attached model and support resources.
- The **single production source notebook** is prepared and has passed CPU bootstrap; a fresh 8-device TPU Run All has not yet happened.
- Real TPU HTTP serving is out of scope until separate end-to-end acceptance.

The previously completed experimental evidence is not erased or relabeled as fresh execution.
