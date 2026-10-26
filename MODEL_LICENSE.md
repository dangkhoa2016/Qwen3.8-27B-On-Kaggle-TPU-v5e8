# Model attribution and license guide

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](MODEL_LICENSE.vi.md)

This project demonstrates **Qwen3.8-27B on Kaggle TPU v5e-8**. Different parts of the solution come from different upstream communities and retain their original terms.

## Know what each license covers

| Resource | Source | Licensing scope |
|---|---|---|
| Repository-authored code and documentation | This project | [MIT](LICENSE) |
| Qwen3.8-27B pretrained model | Qwen upstream | [Apache 2.0 model license copy](THIRD_PARTY_LICENSES/QWEN3.8-27B-APACHE-2.0) |
| MaxText, JAX, Python wheels, Vue and FastAPI | Their respective maintainers | Each upstream component's terms and notices |
| Example images and runtime-support bundle | Individually sourced/generated assets | Verify image rights and component redistribution terms separately |

The included Apache-2.0 text is attribution for the upstream Qwen model, **not** a separate license granted by this project. The checkpoint actually attached for TPU execution is the Kaggle mirror identified below; upstream authorship is credited without suggesting that the notebook loaded weights from Hugging Face.

## How the model is supplied

TPU inference attaches the [Kaggle BF16 mirror](https://www.kaggle.com/models/dangkhoa2016/qwen-qwen3-8-27b) (`dangkhoa2016/qwen-qwen3-8-27b`, Transformers/default/v1). Its runtime mount is `/kaggle/input/models/dangkhoa2016/qwen-qwen3-8-27b/transformers/default/1`. **No pretrained weights are committed to Git.** The model is a mirror of the upstream Qwen checkpoint, not a new training output or a locally loaded Hugging Face endpoint.

When distributing a modified runtime or downstream model, inspect the exact versions, preserve applicable copyright/license/NOTICE files and mark modifications where the relevant license requires it. See [third-party notices](THIRD_PARTY_NOTICES.md).

## Responsible reuse

You can learn from the architecture, API contracts, tests and evaluation procedures without implying that this is an official Qwen deployment. For redistribution of support Datasets, confirm the rights for **all** third-party wheels, patched MaxText source, generated images and cache artifacts first.

This is an engineering attribution guide, not a legal opinion or a guarantee that all possible redistributions are permitted.
