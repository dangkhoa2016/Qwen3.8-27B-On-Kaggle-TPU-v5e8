"""CPU-safe API entrypoint with injectable, fail-closed inference lifecycle.

The executable ASGI app is mock-only. A fake runner can be injected by
tests via create_app(...), but is never enabled by an environment variable.
"""
from __future__ import annotations

import os
import secrets
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable, Literal

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from api.runtime import BackendBusy, BackendUnavailable, InferenceRuntime, State
from api.serving_config import ServingConfig

SERVING_CONFIG = ServingConfig.from_environment()
if SERVING_CONFIG.mode != "mock":
    raise RuntimeError("Default ASGI application is mock-only; TPU staging adapter is not qualified")
BACKEND = os.getenv("QWEN38_BACKEND", "mock")
if BACKEND != "mock":
    raise RuntimeError("ASGI entrypoint is mock-only until live TPU HTTP acceptance passes")

class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=16000)

class CompletionRequest(BaseModel):
    model: str = "qwen3.8-27b"
    messages: list[Message] = Field(min_length=1, max_length=32)
    max_tokens: int = Field(default=128, ge=1, le=160)
    temperature: float = Field(default=0.0, ge=0, le=2)

def mock_reply(content: str) -> str:
    return "[MOCK — NOT MODEL INFERENCE] Received text: " + content[:300]

def _infer_error(error: Exception) -> HTTPException:
    if isinstance(error, BackendBusy):
        return HTTPException(status_code=429, detail="Inference runner busy", headers={"Retry-After": "1"})
    if isinstance(error, BackendUnavailable):
        return HTTPException(status_code=503, detail="Inference runner unavailable")
    if isinstance(error, ValueError):
        return HTTPException(status_code=422, detail="Inference input rejected")
    return HTTPException(status_code=500, detail="Inference runner failed")

def _image_signature(header: bytes, content_type: str) -> bool:
    return (
        (content_type == "image/png" and header.startswith(b"\x89PNG\r\n\x1a\n"))
        or (content_type == "image/jpeg" and header.startswith(b"\xff\xd8\xff"))
        or (content_type == "image/webp" and header[:4] == b"RIFF" and header[8:12] == b"WEBP")
    )

def require_api_key(authorization: str | None = Header(default=None)) -> None:
    expected = os.environ.get("QWEN38_API_KEY", "")
    if not expected:
        # Local mock mode is allowed without a key. Test runner injection is not
        # a public server; real TPU serving must require a key before exposure.
        return
    supplied = (authorization or "")
    prefix = "Bearer "
    if not supplied.startswith(prefix) or not secrets.compare_digest(supplied[len(prefix):], expected):
        raise HTTPException(status_code=401, detail="Unauthorized", headers={"WWW-Authenticate": "Bearer"})

def create_app(*, runner_factory: Callable | None = None) -> FastAPI:
    """For tests only, supply runner_factory. Default ASGI app never loads a model."""
    test_double = runner_factory is not None
    runtime = InferenceRuntime(runner_factory) if test_double else None
    backend = "fake-test-runner" if test_double else "mock"

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if runtime is not None:
            # Failed initialization is recorded as FAILED and HTTP routes fail closed.
            try:
                await runtime.startup()
            except BackendUnavailable:
                pass
        try:
            yield
        finally:
            if runtime is not None:
                await runtime.shutdown()

    application = FastAPI(
        title="Qwen3.8-27B TPU Showcase API", version="0.2.0", lifespan=lifespan,
        description="CPU-only mock and injectable test-double API; no genuine model responses.")
    application.state.runtime = runtime

    @application.get("/", include_in_schema=False)
    def playground():
        return FileResponse(Path(__file__).resolve().parent.parent / "web" / "index.html")

    @application.get("/health")
    def health():
        state = runtime.health() if runtime else None
        return {
            "status": "ok" if state is None or state["state"] == "ready" else "degraded",
            "backend": backend, "model_loaded": False, "inference_available": False,
            "runner_ready": bool(state and state["state"] == "ready"),
            "runner_state": state["state"] if state else "not_configured",
        }

    @application.get("/readyz")
    def readyz():
        if runtime is None:
            return {"ready": True, "mode": "mock", "real_model_ready": False}
        if runtime.state is not State.READY:
            raise HTTPException(status_code=503, detail="Runner not ready")
        return {"ready": True, "mode": backend, "real_model_ready": False}

    @application.get("/metrics")
    def metrics():
        state = runtime.health() if runtime else {}
        return {
            "backend": backend, "model_load_count": 0, "is_tpu": False,
            "real_inference_requests": 0, "note": "CPU mock/test-double only",
            "test_runner_load_count": state.get("model_load_count", 0),
            "test_runner_completed_requests": state.get("completed_requests", 0),
            "test_runner_busy": state.get("busy", False),
        }

    @application.post("/v1/chat/completions", dependencies=[Depends(require_api_key)])
    async def chat(req: CompletionRequest):
        prompt = next((m.content for m in reversed(req.messages) if m.role == "user"), None)
        if not prompt:
            raise HTTPException(status_code=422, detail="A user message is required")
        if len(prompt) > SERVING_CONFIG.max_prompt_chars:
            raise HTTPException(status_code=413, detail="Prompt exceeds configured budget")
        if runtime is not None:
            try:
                result = await runtime.invoke("text", prompt, max_decode_steps=req.max_tokens)
            except Exception as exc:
                raise _infer_error(exc) from exc
            text = "[FAKE RUNNER — NOT QWEN INFERENCE] " + str(result.get("generated_text", ""))
        else:
            text = mock_reply(prompt)
        return {
            "id": "mock-" + uuid.uuid4().hex, "object": "chat.completion",
            "created": int(time.time()), "model": req.model, "backend": backend,
            "mock": True,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
        }

    @application.post("/v1/vision/analyze", dependencies=[Depends(require_api_key)])
    async def vision(image: UploadFile = File(...), prompt: str = Form(...)):
        if not prompt.strip():
            raise HTTPException(status_code=422, detail="Prompt must not be blank")
        if len(prompt) > SERVING_CONFIG.max_prompt_chars:
            raise HTTPException(status_code=422, detail="Prompt too long")
        content_type = image.content_type or ""
        if content_type not in ("image/png", "image/jpeg", "image/webp"):
            raise HTTPException(status_code=415, detail="Only PNG, JPEG and WebP are supported")
        size = 0
        header = bytearray()
        file_path = None
        try:
            if runtime is not None:
                temp_file = tempfile.NamedTemporaryFile(prefix="qwen38-test-", suffix={
                    "image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"
                }[content_type], delete=False)
                file_path = temp_file.name
                temp_file.close()
            with open(file_path, "wb") if file_path else _NullWriter() as sink:
                while chunk := await image.read(1024 * 1024):
                    size += len(chunk)
                    if len(header) < 16:
                        header.extend(chunk[:16 - len(header)])
                    if size > SERVING_CONFIG.max_image_bytes:
                        raise HTTPException(status_code=413, detail="Image exceeds 8 MiB")
                    sink.write(chunk)
            if not size:
                raise HTTPException(status_code=422, detail="Image is empty")
            if not _image_signature(bytes(header), content_type):
                raise HTTPException(status_code=422, detail="Image signature does not match MIME type")
            if runtime is not None:
                try:
                    result = await runtime.invoke("vision", prompt, image_path=file_path, max_decode_steps=160)
                except Exception as exc:
                    raise _infer_error(exc) from exc
                answer = "[FAKE RUNNER — NOT QWEN VISION INFERENCE] " + str(result.get("generated_text", ""))
            else:
                answer = "[MOCK — NOT VISION INFERENCE] Image uploaded successfully. No model was run."
            return {
                "id": "mock-" + uuid.uuid4().hex, "backend": backend, "mock": True,
                "filename": image.filename, "bytes_received": size, "content_type": content_type,
                "prompt": prompt, "answer": answer,
            }
        finally:
            if file_path:
                Path(file_path).unlink(missing_ok=True)

    return application

class _NullWriter:
    """Discard upload bytes in mock mode without materializing image files."""
    def __enter__(self): return self
    def __exit__(self, *_): return False
    def write(self, _: bytes): pass

app = create_app()
