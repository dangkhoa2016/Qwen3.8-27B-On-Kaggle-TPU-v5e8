# Open-source dependencies and acknowledgements

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](THIRD_PARTY_NOTICES.vi.md)

The TPU showcase builds on several open-source projects. Their software and model licenses remain separate from this repository's own [MIT license](LICENSE).

| Project | Role | Upstream information |
|---|---|---|
| [Kaggle BF16 mirror](https://www.kaggle.com/models/dangkhoa2016/qwen-qwen3-8-27b) (upstream Qwen) | BF16 checkpoint for Kaggle TPU inference | [Apache-2.0 upstream license copy](THIRD_PARTY_LICENSES/QWEN3.8-27B-APACHE-2.0) |
| [MaxText](https://github.com/AI-Hypercomputer/maxtext) | TPU-focused transformer runtime | Inspect exact pinned upstream revision and license |
| [JAX](https://github.com/jax-ml/jax) | Compilation and TPU array execution | Apache-2.0 upstream project |
| [FastAPI](https://github.com/fastapi/fastapi) | API demonstration | MIT upstream project |
| [Vue](https://github.com/vuejs/core) | Browser playground | MIT upstream project |
| [Playwright](https://github.com/microsoft/playwright) | End-to-end browser tests | Upstream project license |

The support Dataset additionally contains Python wheels, helpers, patched sources and optional compilation cache artifacts. Those packages are **not all authored by this project**; verify their exact versions, licenses and redistribution requirements before repackaging.

Use [model attribution](MODEL_LICENSE.md) for the separation between project source and pretrained weights. An upstream URL or hash is provenance information, **not automatically evidence of permission to redistribute every artifact**.

## Runtime support distribution notes

The patched MaxText archive contains prominent change notices on integration files and a source modification record; its upstream Apache-2.0 LICENSE is retained. The xprof 2.23.1 wheel declares Apache-2.0 but did not contain a separately identified license file in this audit. The accompanying Runtime Support Dataset supplies a standalone Apache-2.0 license copy. The other wheels retain their individual upstream license conditions.


### xprof 2.23.1 upstream license (audited)

The exact upstream license for [xprof-v2.23.1](https://github.com/openxla/xprof/blob/xprof-v2.23.1/LICENSE) is preserved at [XPROF-2.23.1-LICENSE](THIRD_PARTY_LICENSES/XPROF-2.23.1-LICENSE). It contains Apache-2.0, TensorFlow Authors copyright and Polymer/Web Components BSD attribution text. The pinned wheel has no standalone LICENSE or NOTICE; the source tag has no root NOTICE. Bundled JS additionally has Angular MIT and Chromium notices, which remain in the unchanged wheel. These additional asset notices must not be described as Apache-only.

### Bundled frontend licensing and attribution (audited)
Angular UI bundles embed MIT headers (Angular 15.2.9; zone.js also has an Angular v15.1.0-next.0 header). The full MIT license from angular/angular tag 15.2.9 is retained at [ANGULAR-15.2.9-MIT-LICENSE](THIRD_PARTY_LICENSES/ANGULAR-15.2.9-MIT-LICENSE).

Lit Web Components carry BSD-3-Clause SPDX headers in the JavaScript bundle; the official upstream Lit license snapshot is at [LIT-BSD-3-CLAUSE-LICENSE](THIRD_PARTY_LICENSES/LIT-BSD-3-CLAUSE-LICENSE). The exact Lit package version was not established from this bundled asset.

Chromium copyright/BSD text occurs in the tracing viewer; retain the full [CHROMIUM-BSD-3-CLAUSE-LICENSE](THIRD_PARTY_LICENSES/CHROMIUM-BSD-3-CLAUSE-LICENSE). Microsoft tslib's 0BSD license text remains embedded in the JS bundle and is separately retained at [TSLIB-0BSD-LICENSE](THIRD_PARTY_LICENSES/TSLIB-0BSD-LICENSE).

The wheel's Material Icons WOFF2 matches the exact asset in the XProf 2.23.1 source tag; Google documents Material Icons as Apache-2.0. The existing Apache-2.0 sidecar covers those terms. The proprietary-looking font binary is **not redistributed separately** by this repo: it remains unchanged inside its original wheel.

Source records: https://github.com/angular/angular/blob/15.2.9/LICENSE ; https://github.com/lit/lit/blob/main/LICENSE ; https://github.com/chromium/chromium/blob/main/LICENSE ; https://github.com/microsoft/tslib/blob/main/LICENSE.txt ; https://developers.google.com/fonts/docs/material_icons .

The unchanged wheel's frontend JavaScript contains further upstream headers; this inventory is based on those detected licenses, **not a complete independent supply-chain audit**. The precise library provenance and notice obligations for all native/WASM-linked code are not fully established.

### WASM frontend build-input licenses (scope-limited)

The [XProf 2.23.1 frontend BUILD](https://github.com/openxla/xprof/blob/xprof-v2.23.1/frontend/app/components/trace_viewer_v2/BUILD) and [WORKSPACE](https://github.com/openxla/xprof/blob/xprof-v2.23.1/WORKSPACE) pin dependencies associated with the WebAssembly frontend: **Dear ImGui 1.88** (MIT), **Perfetto 53.0** (Apache-2.0 plus included extra license terms), **Dawn WebGPU v20250713.025201** (BSD-style), and **Emscripten 4.0.6** (MIT / University of Illinois-NCSA). Exact upstream license copies accompany this distribution at [IMGUI-1.88-MIT-LICENSE](THIRD_PARTY_LICENSES/IMGUI-1.88-MIT-LICENSE), [PERFETTO-53.0-LICENSE](THIRD_PARTY_LICENSES/PERFETTO-53.0-LICENSE), [DAWN-20250713-BSD-LICENSE](THIRD_PARTY_LICENSES/DAWN-20250713-BSD-LICENSE), and [EMSCRIPTEN-4.0.6-LICENSE](THIRD_PARTY_LICENSES/EMSCRIPTEN-4.0.6-LICENSE). **Abseil C++** is also in the build dependencies and is Apache-2.0.

These are identified upstream build inputs; the exact static-link closure of the wheel's WASM/native binary was not independently reconstructed. Keeping the notices is prudent, **not a claim of exhaustive audit or endorsement**.
