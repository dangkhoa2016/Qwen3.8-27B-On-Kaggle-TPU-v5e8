# Help, troubleshooting and discussion

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](SUPPORT.vi.md)

If you are exploring the repository, start with the [README](README.md), [step-by-step TPU guide](docs/REPRODUCIBILITY.md) and [CPU playground](api/README.md). You can study the API and evaluation logic even without Kaggle TPU access.

## Choose the right troubleshooting path

| Question | Useful resource |
|---|---|
| The CPU API does not start | [FastAPI playground](api/README.md), Python dependency installation |
| Image upload or UI behavior fails | [Browser tests](tests/browser/showcase.spec.cjs) |
| A Kaggle Dataset is not found | [Offline support and mount layout](docs/OFFLINE-SUPPORT.md) |
| JAX/XLA reports sharding or shape errors | [Architecture](docs/ARCHITECTURE.md) and pinned runtime versions |
| A model answer is incorrect | [Evaluation](docs/ACCEPTANCE.md) and [failure lessons](docs/FAILURE-ANALYSIS.md) |
| You want to contribute | [Contribution guide](CONTRIBUTING.md) |

## Writing a useful issue

Include the commit SHA, exact command, Python/JAX versions, environment and GPU/TPU device list if relevant. Provide a small reproducible example, expected versus actual behavior, and **redacted** errors. For model-output reports, include the input image SHA256 and full prompt; do not assume a similarly named image has identical bytes.

**Access note:** some referenced Kaggle resources may require authorization while their license/release evaluation is ongoing. Avoid pasting access credentials into issues.

General discussion can use GitHub issues; suspected vulnerabilities belong in the [private security channel](SECURITY.md). This community engineering sample does not promise a production support SLA.
