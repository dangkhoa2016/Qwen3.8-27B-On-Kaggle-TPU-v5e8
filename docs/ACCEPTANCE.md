# Text and Vision acceptance criteria

> **English** | [Tiếng Việt](ACCEPTANCE.vi.md)

## Current qualification status

The CPU REST contract and browser test suite are separately verified by GitHub Actions. The current public Kaggle notebook and high-resolution Vision input set **have not yet completed a fresh end-to-end TPU acceptance**. The experimental TPU HTTP endpoint is also **not qualified**. No historical notebook or prior-run metrics are presented as evidence for the current build.

## Reproducible TPU acceptance

1. Record exact Git commit, model and support mount, source patch SHA-256, dependency versions, 8 detected TPU devices, and run date.
2. Verify exactly one BF16 MaxEngine / parameter load, and record loading and initialization durations.
3. Run deterministic text tasks, the 731 and 482 image-conditioning cards, and all six high-resolution scene inputs; save individual prompts, outputs, numeric checks and latencies.
4. For OCR, report both matched fields and errors; do not infer broad capability from short cards or curated inputs.
5. Independently test real TPU HTTP readiness, authentication, Text/Vision, input bounds, busy rejection (429), failure modes (503), lifecycle and shutdown.
6. Save new executed notebook, structured evidence, checksums, and explicitly identify partial or failed checks. Only then mark the relevant feature PASS.

A green CPU CI badge does not establish live TPU inference.
