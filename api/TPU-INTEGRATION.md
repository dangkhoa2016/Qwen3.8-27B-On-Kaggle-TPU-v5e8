# TPU adapter integration status

**Status:** CPU lifecycle unit tests PASS; actual TPU HTTP inference not yet tested.

The accepted notebook at `evidence/qwen38-production-fresh-executed.ipynb`
is the source for MaxEngine text and vision preprocessing.

The new `api/maxengine_adapter.py` contains the extraction of the
single engine's `prefill` and `generate` code paths. The new
`api/runtime.py` manages startup, READY / FAILED / CLOSED states,
at-most-one active inference and shutdown. It does not import JAX on CPU.

## Next TPU acceptance stage

1. Bootstrap pinned MaxText, patch and model artifacts, and create exactly
   one `MaxEngine` and one BF16 parameter tree, using notebook settings.
2. Instantiate `MaxEngineRunner(engine, params, config, tokenizer)` inside
   the server's lifespan startup; pass a factory to `InferenceRuntime`.
3. Wire API endpoints to `InferenceRuntime.invoke` only once model readiness
   is verified; preserve the `mock: true` flag for CPU mock responses.
4. Verify text, strict vision cards, 6 rich scenes, serialization, readiness,
   error paths, file-size limits and shutdown on live TPU.
5. Only after that qualify external HTTP access, authentication and
   Cloudflare Tunnel. Don't publicly expose unprotected inference routes.

The adapter hasn't yet been fully connected to the FastAPI application;
the current CPU HTTP API deliberately remains mock-only. Input limits in
the port match the previous notebook (fixed prefill 256 / target 384).

## CPU HTTP lifecycle validation

FastAPI now provides `create_app(runner_factory=...)` for injected test runners, while the default executable `app` remains strictly mock-only. `lifespan` starts a fake runner once and closes it on shutdown. HTTP readiness, refusal on failed startup (503), and busy rejection (429) are tested in `tests/test_http_runtime.py`. This does not enable a real MaxEngine HTTP service.

## Startup factory and API key

`api/tpu_factory.py` now contains a fail-closed TPU-only factory with pinned MaxText
source, BF16 model archive checks, device verification and one MaxEngine load.
The factory requires `QWEN38_EXPERIMENTAL_TPU_ENABLE=YES_I_ACCEPT_UNQUALIFIED_API`
before model initialization and has **not** been run on TPU. The default ASGI
`app` remains mock-only; do not expose a real runner through Cloudflare yet.

Set `QWEN38_API_KEY` to require a Bearer token for text and vision endpoints.
This is an initial CPU authentication gate, not a complete public-facing
security review (rate limits, headers, ingress policies and secrets management
still pending).
