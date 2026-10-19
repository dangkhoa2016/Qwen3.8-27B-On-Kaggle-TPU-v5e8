# TPU inference integration

> **English** | [Tiếng Việt](TPU-INTEGRATION.vi.md)

## Architecture and current support

The optional Qwen3.8-27B BF16 TPU inference backend uses a guarded MaxText bootstrap and a single MaxEngine runner. The default REST service remains CPU mock only; it does not produce model inference. Live TPU HTTP serving is experimental and **not yet qualified** by end-to-end acceptance.

| Module | Responsibility |
|---|---|
| maxengine_adapter.py | Text and Vision prefill/decode adapter |
| runtime.py | Single-runner startup/shutdown; READY/FAILED/CLOSED; reject concurrent requests |
| tpu_factory.py | Verify model's 18 shards, MaxText archive/patch digests, eight devices and BF16 load |
| app.py | Default mock-only FastAPI contract and test runner injection |
| serving_config.py | Validate experimental mode, request bounds and credentials |
| tpu_asgi.py | Separate opt-in TPU ASGI entrypoint with model loading inside lifespan |

The implementation is governed by pinned source, model and patch integrity checks. The checked-in showcase notebook documents Text/Vision preprocessing and inference workflows. A fresh executed notebook and independent HTTP evidence for this source revision are still required; prior experimental outputs are not bundled as current acceptance.

## Execution and security contract

The experimental entrypoint requires QWEN38_SERVING_MODE=tpu-experimental, QWEN38_EXPERIMENTAL_TPU_ENABLE=YES_I_ACCEPT_UNQUALIFIED_API, QWEN38_TPU_HTTP_ACCEPTANCE=I_ACCEPT_NOT_YET_QUALIFIED, and QWEN38_API_KEY with at least 24 characters. Bind to 127.0.0.1 and exactly one worker; do not use Uvicorn auto-reload.

Request bounds: maximum 160 output tokens, 4,000 prompt characters, and 8 MiB per image. Busy requests are rejected rather than queued indefinitely; initialization failures fail closed. Configuration validation is not a substitute for an ingress security review: **do not expose this unqualified endpoint via public tunnels**.

## Acceptance requirements

Execute real TPU Text/Vision requests on the current 1280×960 inputs; verify one engine/one BF16 load, readiness, authentication, concurrent request rejection (429), failure handling (503), shutdown and measured latency. Capture reproducible SHA-256 verified evidence before claiming PASS. See [acceptance criteria](../docs/ACCEPTANCE.md).

CPU unit and browser tests qualify only the mock/contract behavior.
