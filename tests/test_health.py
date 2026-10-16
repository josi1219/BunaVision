"""
Tests for API root and /health endpoints.
"""
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_api_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "project" in data
    assert "BunaVision" in data["project"]
    assert "endpoints" in data
    assert "assessments" in data["endpoints"]

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "BunaVision-API"
    assert "yolo_model_loaded" in data
    assert "gemini_configured" in data
