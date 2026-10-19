import io
from fastapi.testclient import TestClient
from api.app import app

client=TestClient(app)

def test_health_reports_no_model():
    x=client.get("/health")
    assert x.status_code==200
    assert x.json()["model_loaded"] is False
    assert x.json()["backend"]=="mock"

def test_metrics_no_false_inference():
    d=client.get("/metrics").json()
    assert d["real_inference_requests"]==0

def test_chat_is_explicitly_mock():
    x=client.post("/v1/chat/completions",json={"messages":[{"role":"user","content":"Hello"}]})
    assert x.status_code==200
    assert x.json()["mock"] is True
    assert "NOT MODEL INFERENCE" in x.json()["choices"][0]["message"]["content"]

def test_chat_requires_user_message():
    x=client.post("/v1/chat/completions",json={"messages":[{"role":"assistant","content":"test"}]})
    assert x.status_code==422

def test_vision_png_mock():
    img=b"\x89PNG\r\n\x1a\n" + b"abcdef012345"
    r=client.post("/v1/vision/analyze",data={"prompt":"What is in image?"},
                  files={"image":("test.png",io.BytesIO(img),"image/png")})
    assert r.status_code==200, r.text
    assert r.json()["mock"] is True and r.json()["bytes_received"]==len(img)

def test_vision_rejects_non_image():
    r=client.post("/v1/vision/analyze",data={"prompt":"Read"},files={"image":("a.txt",b"test","text/plain")})
    assert r.status_code==415

def test_vision_rejects_mime_spoof():
    r=client.post("/v1/vision/analyze",data={"prompt":"Read"},files={"image":("fake.png",b"invalid data","image/png")})
    assert r.status_code==422


def test_ui_served():
    response=client.get("/")
    assert response.status_code==200
    assert "MOCK BACKEND" in response.text
    assert "sendVision" in response.text
