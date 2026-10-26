# Open-source dependencies and acknowledgements

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](THIRD_PARTY_NOTICES.vi.md)

The TPU showcase builds on several open-source projects. Their software and model licenses remain separate from this repository's own [MIT license](LICENSE).

| Project | Role | Upstream information |
|---|---|---|
| [Kaggle BF16 mirror](https://www.kaggle.com/models/dangkhoa2016/qwen-qwen3-8-27b) (upstream Qwen) | BF16 checkpoint actually attached on Kaggle TPU | [Apache-2.0 upstream license copy](THIRD_PARTY_LICENSES/QWEN3.8-27B-APACHE-2.0) |
| [MaxText](https://github.com/AI-Hypercomputer/maxtext) | TPU-focused transformer runtime | Inspect exact pinned upstream revision and license |
| [JAX](https://github.com/jax-ml/jax) | Compilation and TPU array execution | Apache-2.0 upstream project |
| [FastAPI](https://github.com/fastapi/fastapi) | API demonstration | MIT upstream project |
| [Vue](https://github.com/vuejs/core) | Browser playground | MIT upstream project |
| [Playwright](https://github.com/microsoft/playwright) | End-to-end browser tests | Upstream project license |

The support Dataset additionally contains Python wheels, helpers, patched sources and optional compilation cache artifacts. Those packages are **not all authored by this project**; verify their exact versions, licenses and redistribution requirements before repackaging.

Use [model attribution](MODEL_LICENSE.md) for the separation between project source and pretrained weights. An upstream URL or hash is provenance information, **not automatically evidence of permission to redistribute every artifact**.
