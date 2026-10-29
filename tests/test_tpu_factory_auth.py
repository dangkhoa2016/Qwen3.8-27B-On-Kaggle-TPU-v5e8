import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from api.tpu_factory import build_tpu_runner

def test_tpu_factory_is_disabled_on_cpu(monkeypatch):
    monkeypatch.delenv("QWEN38_EXPERIMENTAL_TPU_ENABLE", raising=False)
    with pytest.raises(RuntimeError, match="disabled"):
        build_tpu_runner()

def test_api_key_required_for_chat_and_vision(monkeypatch):
    monkeypatch.setenv("QWEN38_API_KEY", "unit-test-secret-123")
    with TestClient(create_app()) as client:
        payload = {"messages": [{"role": "user", "content": "Hello"}]}
        assert client.post("/v1/chat/completions", json=payload).status_code == 401
        assert client.post("/v1/chat/completions", json=payload,
                           headers={"Authorization": "Bearer wrong"}).status_code == 401
        accepted = client.post("/v1/chat/completions", json=payload,
                               headers={"Authorization": "Bearer unit-test-secret-123"})
        assert accepted.status_code == 200 and accepted.json()["mock"] is True
        image = b"\x89PNG\r\n\x1a\n" + b"dummy image bytes"
        upload = {"image": ("test.png", image, "image/png")}
        assert client.post("/v1/vision/analyze", data={"prompt": "describe"},
                           files=upload).status_code == 401
        response = client.post("/v1/vision/analyze", data={"prompt": "describe"},
                               files=upload, headers={"Authorization": "Bearer unit-test-secret-123"})
        assert response.status_code == 200 and response.json()["mock"] is True

def test_api_key_does_not_break_public_health(monkeypatch):
    monkeypatch.setenv("QWEN38_API_KEY", "unit-test-secret-123")
    with TestClient(create_app()) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/readyz").status_code == 200
