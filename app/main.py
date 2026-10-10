from fastapi import FastAPI, File, UploadFile, HTTPException, status, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging

from app.config import settings
from app.schemas import (
    CoffeeAssessmentResponse,
    HealthCheckResponse,
    YOLODetectionResult
)
from app.services.gemini_vision import analyze_coffee_sample
from app.services.yolo_detector import yolo_service, MODEL_PATH

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("bunavision")

app = FastAPI(
    title="BunaVision AI Engine",
    description="AI-Powered Green Coffee Quality Pre-Assessment API (ECX & SCA Standards). Combines YOLOv8 Object Detection with Gemini 2.0 Multimodal Reasoning.",
    version="0.2.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for frontend, mobile, and third-party consumers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", summary="API Root")
def root():
    return {
        "project": "BunaVision AI Engine",
        "tagline": "AI-Powered Ethiopian Coffee Quality Pre-Assessment",
        "version": "0.2.0",
        "documentation": "/docs",
        "endpoints": {
            "health": "/health",
            "detect": "/api/v1/detect (POST multipart/form-data - YOLO Bounding Boxes)",
            "assess": "/api/v1/assess (POST multipart/form-data - Full Hybrid Assessment)",
            "reload_model": "/api/v1/reload-model (POST - Admin)"
        }
    }

@app.get("/health", response_model=HealthCheckResponse, summary="Service Health Check")
def health_check():
    return HealthCheckResponse(
        status="healthy",
        service="BunaVision-API",
        version="0.2.0",
        gemini_configured=bool(settings.GEMINI_API_KEY),
        yolo_model_loaded=yolo_service.is_loaded,
        yolo_model_path=str(MODEL_PATH) if MODEL_PATH.exists() else None
    )

@app.post(
    "/api/v1/detect",
    response_model=YOLODetectionResult,
    summary="YOLOv8 Green Bean & Defect Localization",
    description="Localizes individual coffee beans and defects with bounding boxes. Returns coordinates, class tags, and a base64-annotated image."
)
async def detect_beans(
    file: UploadFile = File(...),
    confidence: float = Query(0.25, ge=0.05, le=1.0, description="YOLO detection confidence threshold")
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type ({file.content_type}). Please upload an image (JPEG, PNG, WEBP)."
        )

    try:
        image_bytes = await file.read()
        if len(image_bytes) == 0:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty image received.")

        result = yolo_service.detect(image_bytes, confidence_threshold=confidence)
        return result
    except Exception as e:
        logger.error(f"YOLO detection error: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Detection failed: {str(e)}")

@app.post(
    "/api/v1/assess",
    response_model=CoffeeAssessmentResponse,
    summary="Comprehensive Coffee Quality Assessment (Hybrid Engine)",
    description="Fuses YOLOv8 bounding box localization with Gemini 2.0 Flash Q-grader reasoning to deliver exact counts, ECX/SCA grades, and bilingual farmer advisories."
)
async def assess_coffee_sample(
    file: UploadFile = File(...),
    include_yolo: bool = Query(True, description="Attach YOLO bounding box detections to response")
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Uploaded file is not an image ({file.content_type}). Please provide JPEG, PNG, or WEBP."
        )

    try:
        image_bytes = await file.read()
        if len(image_bytes) == 0:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty image received.")

        logger.info(f"Analyzing sample: {file.filename} ({len(image_bytes)} bytes)")
        
        # 1. Run local YOLO defect localization & annotation first
        yolo_result = yolo_service.detect(image_bytes)

        # 2. Feed annotated image & verified YOLO counts into Gemini 2.0 Flash
        assessment = analyze_coffee_sample(image_bytes, yolo_result=yolo_result)

        # 3. Attach YOLO detection objects if requested
        if include_yolo:
            assessment.yolo_detections = yolo_result

        return assessment

    except ValueError as ve:
        logger.warning(f"Validation error during assessment: {ve}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        logger.error(f"Error during assessment inference: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Assessment inference failed: {str(e)}"
        )

@app.post("/api/v1/reload-model", summary="Hot-reload YOLO weights")
def reload_model():
    """Reloads the YOLO model weights from models/best.pt without restarting the server."""
    loaded = yolo_service.reload_model()
    return {
        "status": "reloaded" if loaded else "weights_not_found",
        "yolo_model_loaded": loaded,
        "path": str(MODEL_PATH)
    }
