# Full19 image inputs — 16 SHA256-verified assets

This directory contains **16 source images** for the Qwen3.8-27B Text+Vision showcase, not images produced by Qwen itself. The collection consists of two digit cards (731 and 482), six rich illustrated scenes and eight additional photo/photorealistic scenes. Five of the eight photography cases use AI-generated inputs. Avoid presenting synthetic images as real-world photography.

The authoritative 16 filenames, original dimensions, byte sizes and checksums are in [assets_manifest.json](assets_manifest.json). The six annotated scene prompts are in [vision_showcase_cases.json](vision_showcase_cases.json); eight photography prompts are in [photo_showcase_cases.json](photo_showcase_cases.json). The three text-only Full19 prompts are built in the notebook.

Full19 was run on Kaggle TPU v5e-8 on 27 October 2026 with **all 19 automated cases and EOS passing**, but that run used a private Split-Vision candidate and isolated staged runtime assets; the Git-tracked notebook has **not yet been qualified as a fresh, portable Run All**. The two additional greedy invoice-only repeats both reproduced a tax error. See [acceptance status](../docs/ACCEPTANCE.md) and [evidence inventory](../docs/FULL19-EVIDENCE.md).

The inference preprocessing resized these original source images to 512×384. Do not confuse source dimensions with the effective inference resolution.
