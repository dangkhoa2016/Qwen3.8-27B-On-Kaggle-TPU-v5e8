"""Deliberately separate, unqualified TPU ASGI application.

Importing this module requires explicit TPU staging configuration.
The model is loaded only when the ASGI lifespan starts; never on import.
DO NOT expose to the Internet before live TPU acceptance.
"""
from __future__ import annotations

from api.serving_config import ServingConfig, ConfigurationError

config = ServingConfig.from_environment()
if config.mode != "tpu-experimental":
    raise ConfigurationError("Use api.app:app for CPU mock; TPU staging disabled")
if config.host != "127.0.0.1" or config.workers != 1:
    raise ConfigurationError("TPU staging is loopback-only, one worker")

from api.app import create_app
from api.tpu_factory import build_tpu_runner

app = create_app(runner_factory=build_tpu_runner, _experimental_real=True)
