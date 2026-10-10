# ☕ BunaVision AI Engine: Developer Integration Guide

> **For:** Frontend Developers (React, Flutter, Next.js) & Backend Teammates (Django, Node.js)  
> **Service:** BunaVision AI Coffee Quality Pre-Assessment Microservice  
> **Base URL (Local):** `http://localhost:8000`  
> **Swagger Interactive Docs:** `http://localhost:8000/docs`

---

## 1. Quick Overview

BunaVision evaluates green coffee bean samples according to **Ethiopian Commodity Exchange (ECX)** and **Specialty Coffee Association (SCA)** grading standards.

It exposes two primary image analysis endpoints:
1. **`POST /api/v1/assess` (Recommended - Full Hybrid)**: Runs YOLOv8 object detection to count and locate defects + Gemini 2.0 Flash to generate cupping notes, ECX grade classification, and bilingual recommendations (English & Amharic).
2. **`POST /api/v1/detect` (Fast Bounding Boxes Only)**: Pure computer vision inference returning exact bounding box coordinates and an annotated image.

---

## 2. API Endpoints Specification

### A. Service Health Check
* **Method:** `GET`
* **Route:** `/health`
* **Response (`application/json`):**
```json
{
  "status": "healthy",
  "service": "BunaVision-API",
  "version": "0.2.0",
  "gemini_configured": true,
  "yolo_model_loaded": true,
  "yolo_model_path": "D:\\Coding\\Hackathon\\models\\best.pt"
}
```

---

### B. Full Hybrid Assessment (`POST /api/v1/assess`)
* **Method:** `POST`
* **Route:** `/api/v1/assess`
* **Content-Type:** `multipart/form-data`
* **Form Field:** `file` (Image file: JPEG, PNG, or WEBP)
* **Query Parameter:** `include_yolo=true` (Default: `true`)

#### Sample Response:
```json
{
  "total_beans_detected": 35,
  "healthy_beans_count": 31,
  "primary_defects_count": 1,
  "secondary_defects_count": 3,
  "total_defects_count": 4,
  "defect_percentage": 11.4,
  "grade": "Grade 2 (Specialty - Premium)",
  "export_eligible": true,
  "defects": {
    "primary_defects": [
      {
        "name": "Full Black",
        "category": "Primary",
        "count": 1,
        "description": "Fermented/darkened bean caused by pre-harvest issues"
      }
    ],
    "secondary_defects": [
      {
        "name": "Broken / Chipped",
        "category": "Secondary",
        "count": 3,
        "description": "Mechanical damage during pulping or hulling"
      }
    ]
  },
  "yolo_detections": {
    "model_version": "YOLOv8n-Coffee-v1",
    "total_beans": 35,
    "healthy_beans": 31,
    "defect_beans": 4,
    "defect_counts": {
      "healthy": 31,
      "black": 1,
      "broken": 3
    },
    "detected_beans": [
      {
        "box": { "x1": 120.5, "y1": 84.0, "x2": 178.2, "y2": 142.6 },
        "class_name": "black",
        "confidence": 0.89,
        "category": "Primary Defect"
      }
    ],
    "annotated_image_base64": "/9j/4AAQSkZJRgABAQ..."
  },
  "quality_summary": "High potential specialty lot with minor mechanical chipping.",
  "amharic_summary": "ይህ የቡና ናሙና ከፍተኛ ጥራት ያለው ደረጃ 2 (ስፔሻሊቲ) ነው። በጥቂቱ የተሰበሩ ፍሬዎችን በእጅ በመለየት ደረጃ 1 ማድረስ ይቻላል።",
  "recommendations": [
    "Conduct careful hand-sorting to eliminate broken beans.",
    "Calibrate depulper cylinder clearance to prevent chipping."
  ]
}
```

---

### C. Pure YOLO Detection (`POST /api/v1/detect`)
* **Method:** `POST`
* **Route:** `/api/v1/detect`
* **Content-Type:** `multipart/form-data`
* **Form Field:** `file`
* **Query Parameter:** `confidence=0.25` (Optional threshold, range 0.05 to 1.0)

Returns `yolo_detections` containing localized boxes and the base64-annotated image.

---

## 3. Code Snippets for Teammates

### 1. Frontend: React / Next.js / JavaScript

```javascript
async function assessCoffeeImage(fileInput) {
  const formData = new FormData();
  formData.append("file", fileInput.files[0]);

  const response = await fetch("http://localhost:8000/api/v1/assess", {
    method: "POST",
    body: formData,
  });

  const data = await response.json();
  console.log("Estimated Grade:", data.grade);
  console.log("Amharic Summary:", data.amharic_summary);

  // How to display the annotated image with bounding boxes:
  if (data.yolo_detections?.annotated_image_base64) {
    const imageSrc = `data:image/jpeg;base64,${data.yolo_detections.annotated_image_base64}`;
    document.getElementById("defectImage").src = imageSrc;
  }
}
```

---

### 2. Frontend: Flutter / Dart

```dart
import 'dart:convert';
import 'package:http/http.dart' as http;

Future<Map<String, dynamic>> assessCoffeeSample(String filePath) async {
  var request = http.MultipartRequest(
    'POST',
    Uri.parse('http://YOUR_SERVER_IP:8000/api/v1/assess'),
  );
  request.files.add(await http.MultipartFile.fromPath('file', filePath));

  var streamedResponse = await request.send();
  var response = await http.Response.fromStream(streamedResponse);

  if (response.statusCode == 200) {
    var data = jsonDecode(response.body);
    return data;
  } else {
    throw Exception('Failed to assess image: ${response.body}');
  }
}
```

---

### 3. Backend: Python / Django Integration

```python
import requests

def grade_coffee_with_bunavision(image_path_or_bytes):
    url = "http://localhost:8000/api/v1/assess"
    files = {"file": ("sample.jpg", image_path_or_bytes, "image/jpeg")}
    
    response = requests.post(url, files=files, timeout=45)
    response.raise_for_status()
    
    result = response.json()
    return {
        "grade": result["grade"],
        "export_eligible": result["export_eligible"],
        "defect_count": result["total_defects_count"],
        "advisory_en": result["quality_summary"],
        "advisory_am": result["amharic_summary"],
        "annotated_image_b64": result.get("yolo_detections", {}).get("annotated_image_base64")
    }
```

---

### 4. cURL (Terminal / Postman)

```bash
curl -X POST "http://localhost:8000/api/v1/assess" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_sample.jpg;type=image/jpeg"
```

---

## 4. UI Color Code Guide for Defects

When rendering bounding boxes or badges on the frontend, use this standard color scheme:

| Category | Typical Defects | Hex Color | Severity |
| :--- | :--- | :--- | :--- |
| **Primary Defect** | Full Black, Fungus/Mold, Severe Insect, Stinker | `#DC2626` (Red) | Severe (Zero tolerance in Grade 1) |
| **Secondary Defect** | Broken, Chipped, Partial Black, Immature, Sour | `#EAB308` (Amber) | Moderate (Allowed within limits) |
| **Healthy** | Sound, defect-free green bean | `#22C55E` (Green) | High Quality |
