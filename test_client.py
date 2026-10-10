"""
Client test script to test BunaVision API
"""
import sys
import requests
import json
from pathlib import Path

# Force UTF-8 on Windows terminal for Amharic characters
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_URL = "http://127.0.0.1:8000"

def test_health():
    print("[1] Testing /health endpoint...")
    try:
        res = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"Status: {res.status_code}")
        print("Response:", json.dumps(res.json(), indent=2))
        return res.status_code == 200
    except Exception as e:
        print(f"Failed to connect to {BASE_URL}: {e}")
        return False

def test_assess(image_path: str):
    print(f"\n[2] Testing /api/v1/assess with image: {image_path}...")
    p = Path(image_path)
    if not p.exists():
        print(f"Error: Image {image_path} not found.")
        return

    with open(p, "rb") as f:
        files = {"file": (p.name, f, "image/jpeg")}
        try:
            res = requests.post(f"{BASE_URL}/api/v1/assess", files=files, timeout=60)
            print(f"Status: {res.status_code}")
            if res.status_code == 200:
                data = res.json()
                print("\n================== ASSESSEMENT REPORT ==================")
                print(f"Total Beans Detected: {data['total_beans_detected']}")
                print(f"Defect Count:         {data['total_defects_count']} ({data['defect_percentage']}%)")
                print(f"Assigned Grade:       {data['grade']}")
                print(f"Export Eligible:      {data['export_eligible']}")
                print(f"\nSummary (EN): {data['quality_summary']}")
                print(f"Summary (AM): {data['amharic_summary']}")
                print("\nRecommendations:")
                for rec in data.get('recommendations', []):
                    print(f" - {rec}")
                print("========================================================\n")
            else:
                print("Error response:", res.text)
        except Exception as e:
            print(f"Request failed: {e}")

def test_detect(image_path: str):
    print(f"\n[3] Testing /api/v1/detect (YOLO) with image: {image_path}...")
    p = Path(image_path)
    if not p.exists():
        return

    with open(p, "rb") as f:
        files = {"file": (p.name, f, "image/jpeg")}
        try:
            res = requests.post(f"{BASE_URL}/api/v1/detect", files=files, timeout=30)
            print(f"Status: {res.status_code}")
            if res.status_code == 200:
                data = res.json()
                print(f"Model:        {data['model_version']}")
                print(f"Total Beans:  {data['total_beans']}")
                print(f"Defect Beans: {data['defect_beans']}")
                print(f"Annotated Image Base64: {'Present' if data.get('annotated_image_base64') else 'None'}")
            else:
                print("Error:", res.text)
        except Exception as e:
            print(f"Detect request failed: {e}")

if __name__ == "__main__":
    if not test_health():
        print("Server is not running or unreachable. Start server with: uvicorn app.main:app --reload")
        sys.exit(1)

    img = "test_sample.jpg" if len(sys.argv) < 2 else sys.argv[1]
    test_assess(img)
    test_detect(img)
