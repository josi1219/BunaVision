"""
Tests for API request validation and error handling according to spec Section 12.5.
"""
import io
from PIL import Image
from fastapi.testclient import TestClient
import app.main as main
from app.main import app
from app.config import settings
from app.schemas import YOLODetectionResult
from app.services import gemini_vision

client = TestClient(app)

def test_invalid_mime_rejected():
    # Sending a text file instead of an image
    file_content = b"this is not an image"
    response = client.post(
        "/api/v1/detect",
        files={"file": ("test.txt", file_content, "text/plain")}
    )
    assert response.status_code == 400
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == "INVALID_IMAGE"

def test_empty_file_rejected():
    response = client.post(
        "/api/v1/detect",
        files={"file": ("empty.jpg", b"", "image/jpeg")}
    )
    assert response.status_code == 400
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == "INVALID_IMAGE"

def test_valid_image_detect_endpoint():
    # Create small valid synthetic image
    img = Image.new("RGB", (100, 100), color="green")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    response = client.post(
        "/api/v1/detect",
        files={"file": ("test.jpg", buf.getvalue(), "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "model_version" in data
    assert "total_beans" in data
    assert "detected_beans" in data


def test_assessment_offline_fallback(monkeypatch):
    img = Image.new("RGB", (100, 100), color="green")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")

    yolo_result = YOLODetectionResult(
        model_version="test-model",
        total_beans=2,
        healthy_beans=-1,
        defect_beans=2,
        defect_counts={"Full Black": 2},
        detected_beans=[],
    )
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(main.yolo_service, "detect", lambda image_bytes: yolo_result)

    def gemini_unavailable(*args, **kwargs):
        raise RuntimeError("429 RESOURCE_EXHAUSTED")

    monkeypatch.setattr(gemini_vision, "_call_gemini_sdk", gemini_unavailable)
    monkeypatch.setattr(gemini_vision, "_call_gemini_rest", gemini_unavailable)

    response = client.post(
        "/api/v1/assessments",
        files={"file": ("test.jpg", buf.getvalue(), "image/jpeg")},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    assert data["grade"]["grade_confidence"] == "partial"
    assert data["summary"]["defect_beans_count"] == 2
    assert "offline" in data["cupping_notes"]["aroma"].lower()
    assert "must not be interpreted as the defect rate" in data["limitations"][-1]
