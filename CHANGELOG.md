# Project changelog

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](CHANGELOG.vi.md)

This changelog summarizes **user-visible capabilities and documented experiments**, not an internal development diary. Dates indicate when the listed result or resource was recorded.

## 28 October 2026 — Reproducible release-candidate inputs

- Added a new **green-signal pedestrian crossing** source image to the Git example set, with updated original-image SHA256 in the 16-image manifest.
- Prepared a separately named **Private Full19 Release Support Dataset v1**, including the updated image. Independent Kaggle download verified **48/48 file checksums**.
- Documented the difference between checksum-verified replacement input and the previously executed Full19 results. New image-specific TPU outputs must be verified separately; no temporary notebook link is part of the maintained documentation.
- Improved image provenance guidance and API playground labeling to distinguish CPU mock from actual TPU inference.

## 27 October 2026 — Full19 experimental evidence

- Recorded a real **19/19 automated case completion with EOS** on eight TPU v5e devices, taking **1,160.844 seconds**.
- Documented Text, OCR, Rich Vision and Photography case groups, input SHA256 and unedited generation evidence.
- Disclosed invoice-tax and dashboard-comparison errors so that completion is not mistaken for universal factual accuracy.
- Prepared a 48-entry SHA256-verified offline support bundle to simplify repeat experiments.

## Learn more

[Start here](README.md) · [Architecture](docs/ARCHITECTURE.md) · [Reproduce](docs/REPRODUCIBILITY.md) · [Evaluate](docs/ACCEPTANCE.md)
