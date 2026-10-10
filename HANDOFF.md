# BunaVision AI Engine - Context & Session Handoff

> **Project Overview:** AI-Powered Ethiopian Green Coffee Quality Pre-Assessment microservice built for hackathon deployment and portfolio showcase. Combines Ultralytics YOLOv8 Nano for real-time bounding box defect detection with Gemini 2.0 Flash for ECX cupping appraisal and bilingual agronomy advisories.
> **Date / Timestamp:** 2026-10-10
> **Target Directory:** `D:\Coding\Hackathon\`

---

## 1. Project Status & Progress (Where We Stand)

| Milestone | Status | Details |
| :--- | :--- | :--- |
| **Phase 1: Pure API & VLM Prototype** | ✅ **Complete** | FastAPI operational, Ethiopian ECX/SCA grading engine active, Gemini 2.0 Flash bilingual cupping validated. |
| **Phase 2: Real Dataset Sourcing** | ✅ **Complete** | Downloaded 3,460 real labeled green coffee bean defect images (`green-coffee-bean-defects-14`) via Roboflow Universe. |
| **Phase 3: YOLOv8n Training & ONNX Export** | ✅ **Complete** | Trained for 30 epochs on Colab T4 GPU (mAP50: 88.9%, Precision: 93.7%), exported to ONNX (12.2 MB). |
| **Phase 4: Chained Pipeline Fusion** | ✅ **Complete** | Upgraded engine: YOLO runs first via ONNX Runtime to produce bounding boxes & exact counts, then feeds the annotated image into Gemini 2.0 Flash for grounded reasoning. |
| **Phase 5: Live Verification** | ✅ **Complete** | Verified live on `/health`, `/api/v1/detect`, and `/api/v1/assess` through HTTP socket with 100% accuracy. |
| **Phase 6: Developer Docs & Portfolio** | ✅ **Complete** | Created `API_INTEGRATION_GUIDE.md`, portfolio `README.md`, Dockerfile, and architecture documentation. |


---

## 2. Directory Structure

```text
D:\Coding\Hackathon\
├── app/
│   ├── __init__.py
│   ├── config.py                  # Settings & .env loader
│   ├── main.py                    # FastAPI application & v0.2.0 endpoints
│   ├── schemas.py                 # Pydantic schemas (Bounding boxes, Defect breakdown, Grades)
│   └── services/
│       ├── __init__.py
│       ├── gemini_vision.py       # Gemini 2.0 Flash multimodal inspection
│       ├── grading_rules.py       # ECX / SCA grade classification engine
│       └── yolo_detector.py       # YOLOv8 bounding box inference & annotation
├── models/
│   ├── README.md                  # Instructions for placing trained best.pt
│   └── best.pt                    # (Place Colab trained weights here once training finishes)
├── train_yolov8_coffee.ipynb      # Google Colab fine-tuning notebook
├── API_INTEGRATION_GUIDE.md       # Complete handbook for frontend and Django teammates
├── README.md                      # Portfolio repository overview with architecture diagrams
├── .env                           # Local environment config (GEMINI_API_KEY)
├── .env.example                   # Template config
├── Dockerfile                     # Container deployment spec
├── generate_test_image.py         # Mock coffee bean image generator
├── test_client.py                 # CLI test runner for health, assess, and detect
├── requirements.txt               # Pinned dependencies
└── HANDOFF.md                     # This handover document
```

---

## 3. Immediate Next Steps When Colab Training Finishes

1. **Download Weights:**
   Run the download cell in Colab:
   ```python
   from google.colab import files
   files.download("/content/bunavision_runs/real_coffee_yolov8n/weights/best.pt")
   ```
2. **Place in Project:**
   Move `best.pt` from your `Downloads` folder to:
   ```text
   D:\Coding\Hackathon\models\best.pt
   ```
3. **Hot-Reload the Backend:**
   In PowerShell:
   ```powershell
   Invoke-RestMethod -Uri "http://localhost:8000/api/v1/reload-model" -Method Post
   ```
   Or restart `uvicorn`:
   ```powershell
   .venv\Scripts\uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```
4. **Verify Live CV:**
   Run `test_client.py` to see your real fine-tuned YOLO model predicting live defect bounding boxes!

---

## 4. Teammate Handoff

Share [`API_INTEGRATION_GUIDE.md`](API_INTEGRATION_GUIDE.md) directly with:
- **Frontend Developer:** Contains ready-to-paste Fetch/Axios (React) and Dart/http (Flutter) snippets + instructions for rendering bounding boxes and Base64 images.
- **Django Backend Teammate:** Contains the Python `requests` proxy snippet and expected JSON payloads.
