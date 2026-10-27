import io
import threading
import time
from fastapi.testclient import TestClient
from api.app import create_app

class Fake:
    def __init__(self):
        self.calls = 0
        self.closed = False
    def generate_text(self, prompt, *, max_decode_steps):
        self.calls += 1
        return {"generated_text": prompt + " (fake)"}
    def generate_vision(self, image_path, prompt, *, max_decode_steps):
        self.calls += 1
        assert open(image_path, "rb").read(8) == b"\x89PNG\r\n\x1a\n"
        return {"generated_text": "fake vision"}
    def close(self):
        self.closed = True

def test_http_lifespan_single_load_chat_vision_shutdown():
    fake = Fake()
    loads = []
    def factory():
        loads.append(1)
        return fake
    with TestClient(create_app(runner_factory=factory)) as client:
        assert client.get("/health").json()["runner_ready"] is True
        assert client.get("/health").json()["model_loaded"] is False
        assert client.get("/readyz").json()["real_model_ready"] is False
        r = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "ping"}]})
        assert r.status_code == 200 and r.json()["mock"] is True
        assert "FAKE RUNNER" in r.json()["choices"][0]["message"]["content"]
        png = b"\x89PNG\r\n\x1a\n" + b"abcdefghijkl"
        r = client.post("/v1/vision/analyze", data={"prompt": "describe"},
                        files={"image": ("sample.png", io.BytesIO(png), "image/png")})
        assert r.status_code == 200, r.text
        assert "FAKE RUNNER" in r.json()["answer"]
        assert client.get("/metrics").json()["test_runner_completed_requests"] == 2
        assert loads == [1]
    assert fake.closed

def test_http_failed_startup_is_503():
    def factory():
        raise RuntimeError("test failure")
    with TestClient(create_app(runner_factory=factory)) as client:
        assert client.get("/health").json()["runner_state"] == "failed"
        assert client.get("/readyz").status_code == 503
        r = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "ping"}]})
        assert r.status_code == 503
        assert "test failure" not in r.text

def test_http_busy_returns_429():
    started, release = threading.Event(), threading.Event()
    class Slow(Fake):
        def generate_text(self, prompt, *, max_decode_steps):
            started.set()
            assert release.wait(8)
            return {"generated_text": prompt}
    with TestClient(create_app(runner_factory=Slow)) as client:
        outcomes = []
        worker = threading.Thread(target=lambda: outcomes.append(
            client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "slow"}]})))
        worker.start()
        assert started.wait(4)
        r = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "parallel"}]})
        assert r.status_code == 429 and r.headers["retry-after"] == "1"
        release.set()
        worker.join(5)
        assert outcomes and outcomes[0].status_code == 200
