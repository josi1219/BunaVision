import uuid
import time
import logging
from datetime import datetime, timezone

from fastapi import FastAPI, File, UploadFile, HTTPException, status, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.schemas import (
    CoffeeAssessmentResponse,
    HealthCheckResponse,
    YOLODetectionResult,
    ErrorDetail,
    ErrorResponse,
)
from app.services.gemini_vision import analyze_coffee_sample
from app.services.yolo_detector import yolo_service, MODEL_PATH

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("bunavision")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_IMAGE_BYTES = 10 * 1024 * 1024   # 10 MB hard limit
SUPPORTED_MIME_PREFIXES = ("image/jpeg", "image/png", "image/webp")

# Directory to automatically persist annotated images on POST
import base64
from pathlib import Path

YOLO_IMAGES_DIR = Path(__file__).resolve().parent.parent / "yolo_images"
YOLO_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

def _save_annotated_image(b64_str: str | None, filename_prefix: str) -> str | None:
    """Decodes and saves annotated bounding-box image to yolo_images/ folder."""
    if not b64_str:
        return None
    try:
        raw_bytes = base64.b64decode(b64_str)
        target_file = YOLO_IMAGES_DIR / f"{filename_prefix}.jpg"
        with open(target_file, "wb") as f:
            f.write(raw_bytes)
        logger.info(f"Saved annotated image with bounding boxes to: {target_file}")
        return str(target_file)
    except Exception as e:
        logger.warning(f"Could not save annotated image: {e}")
        return None

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="BunaVision AI Engine",
    description=(
        "AI-Powered Green Coffee Quality Pre-Assessment API "
        "(ECX & SCA Standards). Combines YOLOv8 Object Detection with "
        "Gemini 2.0 Flash Multimodal Reasoning."
    ),
    version="0.3.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# CORS
# allow_credentials=True is incompatible with allow_origins=["*"].
# Using wildcard origin with no credentials — correct for a public AI API.
# The backend team should restrict origins once they know their frontend URL.
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _error_response(code: str, message: str, http_status: int) -> JSONResponse:
    """
    Builds a spec Section 12.5 structured error response.

    Supported error codes:
        INVALID_IMAGE      — wrong file type or unreadable content
        IMAGE_TOO_LARGE    — exceeds 10 MB limit
        UNSUPPORTED_FORMAT — MIME type not JPEG / PNG / WEBP
        INFERENCE_FAILED   — YOLO or Gemini pipeline error
        INTERNAL_ERROR     — unexpected server fault
    """
    body = ErrorResponse(error=ErrorDetail(code=code, message=message))
    return JSONResponse(status_code=http_status, content=body.model_dump())


def _validate_upload(file: UploadFile, image_bytes: bytes) -> JSONResponse | None:
    """
    Validates content type and file size.
    Returns an ErrorResponse JSONResponse if invalid, None if OK.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        return _error_response(
            code="INVALID_IMAGE",
            message=f"Uploaded file is not a supported image (received: {file.content_type}). "
                    "Please upload JPEG, PNG, or WEBP.",
            http_status=status.HTTP_400_BAD_REQUEST,
        )

    if file.content_type not in SUPPORTED_MIME_PREFIXES:
        return _error_response(
            code="UNSUPPORTED_FORMAT",
            message=f"Format '{file.content_type}' is not supported. Use JPEG, PNG, or WEBP.",
            http_status=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
        )

    if len(image_bytes) == 0:
        return _error_response(
            code="INVALID_IMAGE",
            message="The uploaded image file is empty.",
            http_status=status.HTTP_400_BAD_REQUEST,
        )

    if len(image_bytes) > MAX_IMAGE_BYTES:
        mb = len(image_bytes) / (1024 * 1024)
        return _error_response(
            code="IMAGE_TOO_LARGE",
            message=f"Image size ({mb:.1f} MB) exceeds the 10 MB limit. "
                    "Please resize the image before uploading.",
            http_status=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
        )

    return None  # all checks passed


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", summary="API Root")
def root():
    return {
        "project": "BunaVision AI Engine",
        "tagline": "AI-Powered Ethiopian Green Coffee Quality Pre-Assessment",
        "version": "0.3.0",
        "documentation": "/docs",
        "endpoints": {
            "health":       "GET  /health",
            "detect":       "POST /api/v1/detect        — YOLO bounding boxes only",
            "assessments":  "POST /api/v1/assessments   — Full hybrid AI assessment",
            "reload_model": "POST /api/v1/reload-model  — Admin: hot-reload YOLO weights",
        },
    }


@app.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="Service Health Check",
    description="Returns service status, Gemini API key presence, and YOLO model load state. "
                "Safe for load-balancer probes — exposes no secrets or internal config.",
)
def health_check():
    return HealthCheckResponse(
        status="healthy",
        service="BunaVision-API",
        version="0.3.0",
        gemini_configured=bool(settings.GEMINI_API_KEY),
        yolo_model_loaded=yolo_service.is_loaded,
        yolo_model_path=str(MODEL_PATH) if MODEL_PATH.exists() else None,
    )


@app.post(
    "/api/v1/detect",
    response_model=YOLODetectionResult,
    summary="YOLOv8 Defect Localization",
    description=(
        "Runs ONNX inference on the uploaded image and returns bounding boxes, "
        "class names, confidence scores, and a base64-annotated JPEG. "
        "Use this endpoint when you need raw detection data without Gemini analysis."
    ),
)
async def detect_beans(
    file: UploadFile = File(...),
    confidence: float = Query(0.25, ge=0.05, le=1.0, description="YOLO confidence threshold"),
):
    image_bytes = await file.read()

    validation_error = _validate_upload(file, image_bytes)
    if validation_error:
        return validation_error

    try:
        result = yolo_service.detect(image_bytes, confidence_threshold=confidence)
        if result.annotated_image_base64:
            _save_annotated_image(result.annotated_image_base64, f"detect_{int(time.time())}")
        return result
    except Exception as e:
        logger.error(f"YOLO detection error: {e}", exc_info=True)
        return _error_response(
            code="INFERENCE_FAILED",
            message="YOLO detection pipeline encountered an error. Please try again.",
            http_status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@app.post(
    "/api/v1/assessments",
    response_model=CoffeeAssessmentResponse,
    summary="Full Coffee Quality Assessment (Spec Section 12.1)",
    description=(
        "Runs the full hybrid AI pipeline: "
        "YOLOv8 ONNX defect localization → ECX rule engine grading → "
        "Gemini 2.0 Flash cupping analysis. "
        "Returns a complete assessment_id-stamped report in the spec Section 12.1 JSON format."
    ),
)
async def create_assessment(
    file: UploadFile = File(...),
    include_yolo: bool = Query(
        True,
        description="Include full YOLO bounding box data in response. "
                    "Backend team: strip annotated_image_base64 before DB write and store S3 key instead.",
    ),
):
    # --- Generate assessment identity up front ---
    assessment_id = f"asmt_{uuid.uuid4().hex[:12]}"
    created_at = _utc_now_iso()
    t_start = time.monotonic()

    # --- Read and validate upload ---
    image_bytes = await file.read()
    validation_error = _validate_upload(file, image_bytes)
    if validation_error:
        return validation_error

    logger.info(
        f"[{assessment_id}] Received sample: {file.filename} "
        f"({len(image_bytes) / 1024:.1f} KB)"
    )

    try:
        # Stage 1 — YOLO: Defect localization & bounding box annotation
        yolo_result = yolo_service.detect(image_bytes)
        logger.info(
            f"[{assessment_id}] YOLO complete — "
            f"{yolo_result.defect_beans} defect instance(s) detected"
        )
        if yolo_result.annotated_image_base64:
            _save_annotated_image(yolo_result.annotated_image_base64, f"{assessment_id}")

        # Stage 2 — Gemini: Grounded multimodal cupping analysis
        assessment = analyze_coffee_sample(
            image_bytes,
            yolo_result=yolo_result,
            assessment_id=assessment_id,
            created_at=created_at,
        )
        logger.info(f"[{assessment_id}] Gemini analysis complete")

        # Stage 3 — Attach timing and YOLO detections
        assessment.processing_time_ms = int((time.monotonic() - t_start) * 1000)

        if include_yolo:
            assessment.yolo_detections = yolo_result

        return assessment

    except ValueError as ve:
        # Raised by gemini_vision for bad API key or unreadable image
        logger.warning(f"[{assessment_id}] Validation error: {ve}")
        return _error_response(
            code="INVALID_IMAGE",
            message=str(ve),
            http_status=status.HTTP_400_BAD_REQUEST,
        )
    except Exception as e:
        logger.error(f"[{assessment_id}] Pipeline error: {e}", exc_info=True)
        return _error_response(
            code="INFERENCE_FAILED",
            message="The AI assessment pipeline encountered an unexpected error. "
                    "Please try again or contact support.",
            http_status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@app.post(
    "/api/v1/reload-model",
    summary="Hot-reload YOLO Weights (Admin)",
    description="Reloads model weights from the models/ directory without restarting the server. "
                "Intended for admin use after deploying updated weights to the server.",
)
def reload_model():
    loaded = yolo_service.reload_model()
    return {
        "status": "reloaded" if loaded else "weights_not_found",
        "yolo_model_loaded": loaded,
        "model_path": str(MODEL_PATH),
    }
