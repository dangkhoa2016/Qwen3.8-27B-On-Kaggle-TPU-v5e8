"""Fail-closed inference lifecycle. No TPU dependencies imported on CPU."""
from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Protocol


class State(str, Enum):
    NOT_STARTED = "not_started"
    LOADING = "loading"
    READY = "ready"
    FAILED = "failed"
    CLOSED = "closed"


class InferenceRunner(Protocol):
    def generate_text(self, prompt: str, *, max_decode_steps: int) -> Any: ...
    def generate_vision(self, image_path: str, prompt: str, *, max_decode_steps: int) -> Any: ...
    def close(self) -> Any: ...


class BackendUnavailable(RuntimeError):
    pass


class BackendBusy(RuntimeError):
    pass


class InferenceRuntime:
    """One runner per process, readiness gate and at-most-one active inference.

    A blocking MaxEngine call runs on a worker thread, not FastAPI's event loop.
    The lock is shared by text and vision because decode state isn't concurrent-safe.
    """

    def __init__(self, factory: Callable[[], InferenceRunner], *, timeout_seconds: float = 120):
        self.factory = factory
        self.timeout_seconds = timeout_seconds
        self.state = State.NOT_STARTED
        self.runner: InferenceRunner | None = None
        self.last_error: str | None = None
        self.load_count = 0
        self.request_count = 0
        self._init_lock = asyncio.Lock()
        self._call_lock = asyncio.Lock()

    async def startup(self) -> None:
        async with self._init_lock:
            if self.state is State.READY:
                return
            if self.state in (State.FAILED, State.CLOSED):
                raise BackendUnavailable(f"Runtime state: {self.state.value}")
            self.state = State.LOADING
            try:
                runner = await asyncio.to_thread(self.factory)
                if not hasattr(runner, 'generate_text') or not hasattr(runner, 'generate_vision'):
                    raise TypeError("Runner does not implement both inference modes")
                self.runner = runner
                self.load_count += 1
                self.state = State.READY
                self.last_error = None
            except Exception as exc:
                self.state = State.FAILED
                self.last_error = type(exc).__name__ + ": " + str(exc)[:240]
                raise BackendUnavailable("Model initialization failed") from exc

    async def invoke(self, kind: str, prompt: str, *, max_decode_steps: int, image_path: str | None = None) -> Any:
        if self.state is not State.READY or self.runner is None:
            raise BackendUnavailable("Model not ready")
        # Do not pile up TPU requests in an implicit unbounded queue.
        if self._call_lock.locked():
            raise BackendBusy("Inference engine is busy")
        async with self._call_lock:
            if self.state is not State.READY or self.runner is None:
                raise BackendUnavailable("Model shutting down")
            if kind == "text":
                fn = lambda: self.runner.generate_text(prompt, max_decode_steps=max_decode_steps)
            elif kind == "vision" and image_path:
                fn = lambda: self.runner.generate_vision(image_path, prompt, max_decode_steps=max_decode_steps)
            else:
                raise ValueError("Unsupported inference request")
            # Timeouts must NOT release the engine to receive a second request while
            # the worker thread continues. Wait until completion; HTTP ingress can time out.
            try:
                result = await asyncio.to_thread(fn)
                self.request_count += 1
                return result
            except Exception as exc:
                self.last_error = type(exc).__name__ + ": " + str(exc)[:240]
                raise

    async def shutdown(self) -> None:
        async with self._init_lock:
            async with self._call_lock:
                if self.state is State.CLOSED:
                    return
                if self.runner is not None:
                    close = getattr(self.runner, "close", None)
                    if close is not None:
                        if inspect.iscoroutinefunction(close):
                            await close()
                        else:
                            await asyncio.to_thread(close)
                self.runner = None
                self.state = State.CLOSED

    def health(self) -> dict:
        return {"state": self.state.value, "model_loaded": self.state is State.READY,
                "model_load_count": self.load_count, "completed_requests": self.request_count,
                "busy": self._call_lock.locked(), "last_error": self.last_error}
