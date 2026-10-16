"""
Tests for API request validation and error handling according to spec Section 12.5.
"""
import io
from PIL import Image
from fastapi.testclient import TestClient
from app.main import app

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
