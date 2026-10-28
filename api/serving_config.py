"""Fail-closed CPU-validated configuration for the unqualified TPU staging API."""
from __future__ import annotations

import os
from dataclasses import dataclass

class ConfigurationError(ValueError):
    pass

@dataclass(frozen=True)
class ServingConfig:
    mode: str
    host: str
    workers: int
    api_key: str
    max_tokens: int
    max_image_bytes: int
    max_prompt_chars: int

    @classmethod
    def from_environment(cls, env=None):
        env = os.environ if env is None else env
        mode = env.get("QWEN38_SERVING_MODE", "mock")
        if mode not in ("mock", "tpu-experimental"):
            raise ConfigurationError("Unknown serving mode")
        host = env.get("QWEN38_HOST", "127.0.0.1")
        workers = _integer(env, "QWEN38_WORKERS", 1, 1, 1)
        max_tokens = _integer(env, "QWEN38_MAX_TOKENS", 160, 1, 160)
        max_image_bytes = _integer(env, "QWEN38_MAX_IMAGE_BYTES", 8 * 1024 * 1024, 1024, 8 * 1024 * 1024)
        max_prompt_chars = _integer(env, "QWEN38_MAX_PROMPT_CHARS", 4000, 1, 4000)
        api_key = env.get("QWEN38_API_KEY", "")
        if mode == "tpu-experimental":
            if env.get("QWEN38_EXPERIMENTAL_TPU_ENABLE") != "YES_I_ACCEPT_UNQUALIFIED_API":
                raise ConfigurationError("Experimental TPU gate is disabled")
            if env.get("QWEN38_TPU_HTTP_ACCEPTANCE") != "I_ACCEPT_NOT_YET_QUALIFIED":
                raise ConfigurationError("Unqualified TPU HTTP acceptance gate is disabled")
            if host != "127.0.0.1":
                raise ConfigurationError("Experimental server must bind loopback")
            if len(api_key) < 24 or api_key.strip() != api_key:
                raise ConfigurationError("Experimental mode requires an API key of >=24 characters")
        return cls(mode, host, workers, api_key, max_tokens, max_image_bytes, max_prompt_chars)

def _integer(env, key, default, minimum, maximum):
    try:
        value = int(env.get(key, str(default)))
    except (ValueError, TypeError) as exc:
        raise ConfigurationError(f"Invalid integer: {key}") from exc
    if not minimum <= value <= maximum:
        raise ConfigurationError(f"{key} out of range [{minimum}, {maximum}]")
    return value
