"""CPU contract checks for the separate TPU ASGI entrypoint.

No TPU hardware is used; tests verify imports and startup gating only.
"""
import os
import subprocess
import sys

REPO_MODULE = "api.tpu_asgi"

def probe(extra):
    env = os.environ.copy()
    for key in (
        "QWEN38_SERVING_MODE", "QWEN38_EXPERIMENTAL_TPU_ENABLE",
        "QWEN38_TPU_HTTP_ACCEPTANCE", "QWEN38_API_KEY", "QWEN38_HOST",
        "QWEN38_WORKERS",
    ):
        env.pop(key, None)
    env.update(extra)
    return subprocess.run(
        [sys.executable, "-c", "import api.tpu_asgi; print('STAGING_IMPORT_PASS')"],
        env=env, capture_output=True, text=True, timeout=15,
    )

def test_reject_staging_import_without_opt_in():
    assert probe({}).returncode != 0

def test_reject_staging_with_missing_auth():
    cfg = {
        "QWEN38_SERVING_MODE": "tpu-experimental",
        "QWEN38_EXPERIMENTAL_TPU_ENABLE": "YES_I_ACCEPT_UNQUALIFIED_API",
        "QWEN38_TPU_HTTP_ACCEPTANCE": "I_ACCEPT_NOT_YET_QUALIFIED",
    }
    assert probe(cfg).returncode != 0

def test_reject_staging_non_loopback():
    cfg = {
        "QWEN38_SERVING_MODE": "tpu-experimental",
        "QWEN38_EXPERIMENTAL_TPU_ENABLE": "YES_I_ACCEPT_UNQUALIFIED_API",
        "QWEN38_TPU_HTTP_ACCEPTANCE": "I_ACCEPT_NOT_YET_QUALIFIED",
        "QWEN38_API_KEY": "test-key-only-never-for-real-access",
        "QWEN38_HOST": "0.0.0.0",
    }
    assert probe(cfg).returncode != 0

def test_explicit_staging_import_never_loads_model():
    cfg = {
        "QWEN38_SERVING_MODE": "tpu-experimental",
        "QWEN38_EXPERIMENTAL_TPU_ENABLE": "YES_I_ACCEPT_UNQUALIFIED_API",
        "QWEN38_TPU_HTTP_ACCEPTANCE": "I_ACCEPT_NOT_YET_QUALIFIED",
        "QWEN38_API_KEY": "test-key-only-never-for-real-access",
        "QWEN38_HOST": "127.0.0.1",
    }
    result = probe(cfg)
    assert result.returncode == 0, result.stderr
    assert "STAGING_IMPORT_PASS" in result.stdout
