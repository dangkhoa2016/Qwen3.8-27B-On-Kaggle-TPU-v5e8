# Model licensing, provenance and attribution

> **English** | [Tiếng Việt](MODEL_LICENSE.vi.md)

## Separate licensing scopes

| Component | Copyright holder / origin | License |
|---|---|---|
| Original project source and documentation | Đăng Khoa | [MIT](LICENSE) |
| Qwen/Qwen3.8-27B pretrained model and weights | Qwen upstream | [Apache License 2.0](THIRD_PARTY_LICENSES/QWEN3.8-27B-APACHE-2.0) |
| MaxText, JAX, FastAPI, Vue and other dependencies | Their respective publishers | Their own applicable terms ([notices](THIRD_PARTY_NOTICES.md)) |

**The complete model license is included in this repository**, at [THIRD_PARTY_LICENSES/QWEN3.8-27B-APACHE-2.0](THIRD_PARTY_LICENSES/QWEN3.8-27B-APACHE-2.0). This is an unmodified copy of the upstream LICENSE, not a new grant issued by this project.

Authoritative references: [official model card](https://huggingface.co/Qwen/Qwen3.8-27B), [upstream LICENSE](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/LICENSE), and [Apache-2.0 license](https://www.apache.org/licenses/LICENSE-2.0). The model card states apache-2.0 at the time of review. Recheck upstream and additional notices before redistributing assets.

## Distribution and boundaries

The 27B model is supplied as a separately attached Kaggle Model; **the Git repository does not contain pretrained model weights**. This repository's MIT license does not transfer ownership or relicense upstream weights. When distributing models or modified derivatives, preserve applicable license copies, copyright/attribution, notices and modification notices where required by the upstream license. Independently audit the Kaggle mirror and patched MaxText offline-support dataset. This document is not a guarantee of compliance for every downstream distribution.

This is an independent engineering showcase, not an official Qwen release or endorsement.
