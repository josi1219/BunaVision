# BunaVision API Integration Guide (Spec v1.0 Compliant)

This document is prepared for **Frontend (React / Next.js / Mobile)** and **Backend / Cloud Engineers** on the team to consume the BunaVision AI microservice endpoints.

---

## 1. Base URL & Interactive Documentation

* **Base URL (Local):** `http://localhost:8000`
* **Interactive Swagger UI:** `http://localhost:8000/docs`
* **OpenAPI Schema (JSON):** `http://localhost:8000/openapi.json`
* **Alternative ReDoc:** `http://localhost:8000/redoc`

---

## 2. API Endpoints Overview

| Method | Path | Summary | Content-Type |
|---|---|---|---|
| `GET` | `/health` | Health Check & Model Load Status | `application/json` |
| `POST` | `/api/v1/assessments` | Full Hybrid Assessment (YOLO + ECX + Gemini) | `multipart/form-data` |
| `POST` | `/api/v1/detect` | Pure YOLOv8 Defect Localization | `multipart/form-data` |
| `POST` | `/api/v1/reload-model` | Admin hot-reload of model weights | `application/json` |

---

## 3. Endpoint Specifications

### A. Health Check (`GET /health`)
Used by AWS ECS/Lambda/ALB load balancers and frontend initialization checks.

#### Response:
```json
{
  "status": "healthy",
  "service": "BunaVision-API",
  "version": "0.3.0",
  "gemini_configured": true,
  "yolo_model_loaded": true,
  "yolo_model_path": "D:\\Coding\\Hackathon\\models\\best.onnx"
}
```

---

### B. Full Hybrid Assessment (`POST /api/v1/assessments`)
* **Method:** `POST`
* **Route:** `/api/v1/assessments`
* **Content-Type:** `multipart/form-data`
* **Form Field:** `file` (Image file: JPEG, PNG, or WEBP, max 10MB)
* **Query Parameter:** `include_yolo=true` (Default: `true`)

#### Sample Response (Spec Section 12.1):
```json
{
  "assessment_id": "asmt_7a3d9e1f4b2c",
  "status": "completed",
  "model_version": "YOLOv8n-Coffee-ONNX-v1",
  "created_at": "2026-10-13T17:40:00Z",
  "processing_time_ms": 340,
  "summary": {
    "total_beans_detected": 42,
    "healthy_beans_count": 36,
    "defect_beans_count": 6,
    "primary_defects_count": 0,
    "secondary_defects_count": 6,
    "defect_percentage": 14.29,
    "detected_classes": 1,
    "detection_count": 6
  },
  "predictions": [
    {
      "class_name": "Partial Black",
      "raw_class": "hitam-sebagian",
      "category": "Secondary Defect",
      "confidence": 0.892,
      "count": 6
    }
  ],
  "grade": {
    "estimated_grade": "Grade 1",
    "grade_class": "Specialty",
    "is_export_eligible": true,
    "grade_basis": "0 primary defect points, 2.0 secondary defect points (threshold: 0 primary, < 3 secondary). Calculated via ECX equivalency table.",
    "grade_confidence": "partial"
  },
  "cupping_notes": {
    "aroma": "Sweet floral jasmine with light citrus undertones",
    "flavor": "Clean stone fruit and black tea finish",
    "acidity": "Bright malic acidity",
    "body": "Silky light-medium body",
    "aftertaste": "Crisp and lingering",
    "estimated_sca_score": 86.5
  },
  "summary_text": {
    "english": "Outstanding specialty lot with minimal secondary defects. Suitable for single-origin export.",
    "amharic": "ይህ የቡና ናሙና ከፍተኛ ጥራት ያለው ደረጃ 1 (ስፔሻሊቲ) ነው። በጥቂቱ የተለዩ ጉድለቶችን በእጅ በመለየት ጥራቱን መጠበቅ ይቻላል።"
  },
  "recommendations": [
    "Maintain current cherry selection and drying protocols to preserve Grade 1 consistency.",
    "Calibrate depulper cylinder clearance to prevent chipping."
  ],
  "limitations": [
    "Preliminary AI assessment only. Not an official ECX or SCA grade.",
    "Only 3 of 13 ECX defect classes are detected: Full Black, Partial Black, Coffee Husk / Parchment.",
    "Defects hidden from the camera surface cannot be detected.",
    "Counts require manual verification before any commercial or export decision."
  ],
  "coverage": {
    "classes_scanned": ["Full Black", "Partial Black", "Coffee Husk / Parchment"],
    "classes_not_scanned": ["Full Sour", "Fungus / Mold", "Severe Insect Damage", "Floater / Chalky", "Immature / Quaker", "Broken / Chipped", "Foreign Matter"],
    "coverage_note": "Grade estimated from 3 of 13 ECX defect classes. Unscanned classes may increase defect points."
  },
  "yolo_detections": {
    "model_version": "YOLOv8n-Coffee-ONNX-v1",
    "total_beans": 6,
    "healthy_beans": -1,
    "defect_beans": 6,
    "defect_counts": {
      "Partial Black": 6
    },
    "detected_beans": [
      {
        "box": { "x1": 120.5, "y1": 84.0, "x2": 178.2, "y2": 142.6 },
        "class_name": "Partial Black",
        "confidence": 0.892,
        "category": "Secondary Defect"
      }
    ],
    "annotated_image_base64": "/9j/4AAQSkZJRgABAQ..."
  },
  "annotated_image_url": null
}
```

---

### C. Pure YOLO Detection (`POST /api/v1/detect`)
* **Method:** `POST`
* **Route:** `/api/v1/detect`
* **Content-Type:** `multipart/form-data`
* **Form Field:** `file`
* **Query Parameter:** `confidence=0.25` (Optional threshold, range 0.05 to 1.0)

---

## 4. Code Snippets for Teammates

### Frontend: React / Next.js / TypeScript

```typescript
export async function assessCoffeeImage(file: File) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch("http://localhost:8000/api/v1/assessments", {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(`[${errorData.error.code}] ${errorData.error.message}`);
  }

  const data = await response.json();
  console.log("Assessment ID:", data.assessment_id);
  console.log("Estimated Grade:", data.grade.estimated_grade, data.grade.grade_class);
  console.log("Amharic Summary:", data.summary_text.amharic);

  // How to display the annotated image with colored bounding boxes:
  if (data.yolo_detections?.annotated_image_base64) {
    const imageSrc = `data:image/jpeg;base64,${data.yolo_detections.annotated_image_base64}`;
    document.getElementById("annotatedPreview").src = imageSrc;
  }

  return data;
}
```

### Mobile: Flutter / Dart

```dart
import 'package:http/http.dart' as http;
import 'dart:convert';

Future<Map<String, dynamic>> submitSampleAssessment(String filePath) async {
  var request = http.MultipartRequest(
    'POST',
    Uri.parse('http://YOUR_SERVER_IP:8000/api/v1/assessments'),
  );

  request.files.add(await http.MultipartFile.fromPath('file', filePath));
  var streamedResponse = await request.send();
  var response = await http.Response.fromStream(streamedResponse);

  if (response.statusCode == 200) {
    return jsonDecode(response.body);
  } else {
    throw Exception('Failed assessment: ${response.body}');
  }
}
```

### Backend: Python / Django / Requests

```python
import requests

def assess_sample_file(image_path: str):
    url = "http://localhost:8000/api/v1/assessments"
    with open(image_path, "rb") as img:
        files = {"file": (image_path, img, "image/jpeg")}
        response = requests.post(url, files=files, timeout=45)
        response.raise_for_status()
        return response.json()
```

### cURL

```bash
curl -X POST "http://localhost:8000/api/v1/assessments" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/beans.jpg"
```
