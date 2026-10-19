# Reproducibility guide

> **English** | [Tiếng Việt](REPRODUCIBILITY.vi.md)

1. Check out this project's reviewed source commit. The showcase notebook separately pins the commit containing the 11 example images and their SHA-256 manifest.
2. On Kaggle attach the original Qwen3.8-27B BF16 model and offline-support dataset; select TPU v5e-8 and enable Internet only for anonymous commit-pinned GitHub example checkout.
3. Verify the checkpoint index, 18 model shards, patched MaxText archive/patch SHA-256, JAX/JAXLIB 0.11.0 and Transformers 5.12.1. The model weights are not committed to Git.
4. Run the entire notebook and record both successes and errors. The replaced 1280×960 Vision illustrations require fresh validation and cannot inherit scores from the earlier 256×192 inputs.
5. Separately run Python CPU contracts and Playwright browser E2E. These tests are not substitutes for real TPU HTTP qualification.
6. Publish current executed evidence, SHA-256 manifests and hardware timings only when newly produced and independently checked.

The three additional 1024×768 images are input-only visual examples; do not label them model-generated or inference-qualified.
