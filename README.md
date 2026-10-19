# Qwen3.8-27B on Kaggle TPU v5e-8 — Production Showcase

Repository: https://github.com/dangkhoa2016/Qwen3.8-27B-On-Kaggle-TPU-v5e8

Reproducible model inference examples, test images, deterministic prompts, and pinned asset digests.

## Source of examples

All inference prompts and images come from `examples/` in this repository. The Kaggle notebook clones and pins a Git commit, and verifies the asset SHA-256 checksums. Model weights and pinned MaxText runtime remain in the separate Kaggle model and offline-support dataset.

## Image fidelity

The original six benchmark images and numeric regression cards are only 256×192 pixels. Notebook rendering must not upscale these and imply additional visual detail. Three additional native 1024×768 images are provided for high-resolution inspection; they are not included in previously qualified inference acceptance. Higher-resolution inference needs a new test run and acceptance.

## Governance

Pin a commit in the notebook rather than cloning an unpinned branch. Do not mark an untested fresh notebook as acceptance PASS.
