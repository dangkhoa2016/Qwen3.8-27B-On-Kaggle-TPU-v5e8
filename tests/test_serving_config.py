import pytest
from api.serving_config import ConfigurationError, ServingConfig

BASE = {
    "QWEN38_SERVING_MODE": "tpu-experimental",
    "QWEN38_EXPERIMENTAL_TPU_ENABLE": "YES_I_ACCEPT_UNQUALIFIED_API",
    "QWEN38_TPU_HTTP_ACCEPTANCE": "I_ACCEPT_NOT_YET_QUALIFIED",
    "QWEN38_API_KEY": "not-a-real-secret-for-unit-testing-only",
}

def test_mock_default_is_cpu_only():
    c = ServingConfig.from_environment({})
    assert c.mode == "mock" and c.workers == 1
    assert c.host == "127.0.0.1"

@pytest.mark.parametrize("remove", [
    "QWEN38_EXPERIMENTAL_TPU_ENABLE", "QWEN38_TPU_HTTP_ACCEPTANCE",
    "QWEN38_API_KEY",
])
def test_staging_missing_guard_rejected(remove):
    env = BASE.copy()
    env.pop(remove)
    with pytest.raises(ConfigurationError):
        ServingConfig.from_environment(env)

def test_tpu_staging_requires_loopback_and_single_worker():
    for key, value in [
        ("QWEN38_HOST", "0.0.0.0"),
        ("QWEN38_WORKERS", "2"),
        ("QWEN38_MAX_TOKENS", "161"),
        ("QWEN38_MAX_IMAGE_BYTES", "9000000"),
        ("QWEN38_MAX_PROMPT_CHARS", "5000"),
    ]:
        env = dict(BASE, **{key: value})
        with pytest.raises(ConfigurationError):
            ServingConfig.from_environment(env)

def test_valid_explicit_staging_only_is_configuration_not_model_load():
    c = ServingConfig.from_environment(BASE)
    assert c.mode == "tpu-experimental"
    assert c.host == "127.0.0.1"
    assert c.workers == 1 and c.max_tokens == 160

def test_invalid_mode_is_fail_closed():
    with pytest.raises(ConfigurationError):
        ServingConfig.from_environment({"QWEN38_SERVING_MODE": "unknown"})
