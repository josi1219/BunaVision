from typing import List, Optional, Dict
from pydantic import BaseModel, Field

# -------------------------------------------------------------
# YOLO Computer Vision Schemas (internal — do not modify)
# These are consumed by yolo_detector.py and must stay stable
# -------------------------------------------------------------
class BoundingBox(BaseModel):
    x1: float = Field(..., description="Top-left X coordinate (pixels)")
    y1: float = Field(..., description="Top-left Y coordinate (pixels)")
    x2: float = Field(..., description="Bottom-right X coordinate (pixels)")
    y2: float = Field(..., description="Bottom-right Y coordinate (pixels)")

class DetectedBean(BaseModel):
    box: BoundingBox = Field(..., description="Bounding box coordinates")
    class_name: str = Field(..., description="Canonical English defect class name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence score (0-1)")
    category: str = Field(..., description="'Primary Defect', 'Secondary Defect', or 'Healthy'")

class YOLODetectionResult(BaseModel):
    model_version: str = Field(..., description="Model identifier, e.g. 'YOLOv8n-Coffee-ONNX-v1'")
    total_beans: int = Field(..., ge=0, description="Total defect instances localized by YOLO")
    healthy_beans: int = Field(
        ...,
        ge=-1,
        description=(
            "Healthy bean count. -1 means 'not determinable by YOLO' — "
            "this model was trained only on defect classes and cannot count healthy beans. "
            "Gemini estimates the real value from the full image."
        )
    )
    defect_beans: int = Field(..., ge=0, description="Count of defective bean instances")
    defect_counts: Dict[str, int] = Field(default_factory=dict, description="Raw count per defect class e.g. {'Full Black': 2}")
    detected_beans: List[DetectedBean] = Field(default_factory=list, description="Full list of bounding box detections")
    annotated_image_base64: Optional[str] = Field(None, description="Base64-encoded JPEG with colored bounding boxes drawn")

# -------------------------------------------------------------
# Spec Section 12.1 - Assessment Response Schemas
# These define the public JSON contract consumed by the frontend
# -------------------------------------------------------------

class AssessmentSummary(BaseModel):
    """
    High-level numeric summary block — maps directly to spec Section 12.1 'summary' object.
    Designed to be displayed on the results card in the frontend without parsing nested data.
    """
    total_beans_detected: int = Field(..., ge=0, description="Total visible green coffee beans estimated in the image")
    healthy_beans_count: int = Field(..., ge=0, description="Number of sound, defect-free beans (Gemini estimate)")
    defect_beans_count: int = Field(..., ge=0, description="Total number of beans with any detected defect")
    primary_defects_count: int = Field(..., ge=0, description="Count of Category 1 / severe defect instances")
    secondary_defects_count: int = Field(..., ge=0, description="Count of Category 2 / minor defect instances")
    defect_percentage: float = Field(..., ge=0.0, le=100.0, description="Defective beans as a percentage of total")
    detected_classes: int = Field(..., ge=0, description="Number of distinct defect classes found in this image")
    detection_count: int = Field(..., ge=0, description="Total bounding box detections returned by YOLO")


class DefectPrediction(BaseModel):
    """
    Per-class prediction entry — maps to spec Section 12.1 'predictions' array item.
    One entry per distinct defect class detected in the image.
    """
    class_name: str = Field(..., description="Canonical English defect name e.g. 'Full Black', 'Coffee Husk / Parchment'")
    raw_class: str = Field(..., description="Original dataset class label e.g. 'hitam', 'kulit-kopi-kecil'")
    category: str = Field(..., description="'Primary Defect' or 'Secondary Defect'")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Average model confidence for this class across detections")
    count: int = Field(..., ge=0, description="Number of instances of this defect class detected")


class DetectionCoverage(BaseModel):
    """
    Transparency block — explicitly tells the frontend (and DB) which ECX defect
    classes our model can and cannot detect. Required by spec Section 15.3 AI transparency.
    """
    classes_scanned: List[str] = Field(
        default_factory=list,
        description="ECX defect classes this model version can detect"
    )
    classes_not_scanned: List[str] = Field(
        default_factory=list,
        description="ECX defect classes this model cannot detect — manual inspection needed"
    )
    coverage_note: str = Field(
        default="",
        description="Human-readable explanation of model coverage limitations"
    )


class GradeAssessment(BaseModel):
    """
    ECX grade output block. Separates grade logic from narrative summaries
    so the frontend can display the grade badge independently.
    """
    estimated_grade: str = Field(..., description="e.g. 'Grade 1', 'Grade 2', 'Grade 3'")
    grade_class: str = Field(..., description="e.g. 'Specialty', 'Commercial Specialty', 'Commercial'")
    is_export_eligible: bool = Field(..., description="True if sample meets minimum Ethiopian export standards")
    grade_basis: str = Field(..., description="Short explanation of why this grade was assigned")
    grade_confidence: str = Field(
        default="partial",
        description="'full' if all ECX classes were scanned, 'partial' if coverage is incomplete"
    )


class CuppingNotes(BaseModel):
    """
    Structured cupping evaluation from Gemini 2.5 Flash.
    Modeled on SCA cupping protocol categories.
    """
    aroma: Optional[str] = Field(None, description="Aroma characteristics observed")
    flavor: Optional[str] = Field(None, description="Primary flavor profile")
    acidity: Optional[str] = Field(None, description="Acidity quality and intensity")
    body: Optional[str] = Field(None, description="Mouthfeel and weight")
    aftertaste: Optional[str] = Field(None, description="Finish and lingering notes")
    estimated_sca_score: Optional[float] = Field(None, ge=0.0, le=100.0, description="AI-estimated SCA cup score (not certified)")


class BilingualSummary(BaseModel):
    """
    English and Amharic narrative summaries for farmer-facing reports.
    """
    english: str = Field(..., description="Professional cupper/grader overview in English")
    amharic: str = Field(..., description="ጥራት ማጠቃለያ እና ደረጃ በአማርኛ ለኢትዮጵያ አምራቾች")


# -------------------------------------------------------------
# Top-Level Assessment Response (spec Section 12.1)
# This is what POST /api/v1/assessments returns to the frontend
# -------------------------------------------------------------
class CoffeeAssessmentResponse(BaseModel):
    """
    Complete assessment response contract — spec Section 12.1.

    Field layout mirrors the spec's JSON example:
      assessment_id   -> generated by main.py (UUID)
      status          -> 'completed' or 'failed'
      model_version   -> YOLO model identifier
      created_at      -> ISO 8601 timestamp
      processing_time_ms -> wall-clock ms for full pipeline
      summary         -> AssessmentSummary (numeric overview)
      predictions     -> List[DefectPrediction] (per-class breakdown)
      grade           -> GradeAssessment (ECX grade block)
      cupping_notes   -> CuppingNotes (SCA-style AI evaluation)
      summary_text    -> BilingualSummary (EN + AM narrative)
      recommendations -> actionable producer advisories
      limitations     -> transparency array (required by spec 15.3)
      coverage        -> DetectionCoverage (which classes were scanned)
      yolo_detections -> optional full bounding box data
      annotated_image_url -> null from us; backend fills after S3 upload
    """

    # --- Spec top-level identity fields ---
    assessment_id: str = Field(..., description="UUID generated by the backend at request time")
    status: str = Field(default="completed", description="'completed' or 'failed'")
    model_version: str = Field(default="YOLOv8n-Coffee-ONNX-v1", description="AI model version that produced this result")
    created_at: str = Field(..., description="ISO 8601 UTC timestamp of assessment creation")
    processing_time_ms: int = Field(default=0, description="Total inference wall-clock time in milliseconds")

    # --- Spec summary block ---
    summary: AssessmentSummary = Field(..., description="High-level numeric overview for results card display")

    # --- Spec predictions array ---
    predictions: List[DefectPrediction] = Field(
        default_factory=list,
        description="Per-class defect predictions. One entry per distinct defect class found."
    )

    # --- Grade block ---
    grade: GradeAssessment = Field(..., description="ECX grade assignment with basis and confidence")

    # --- AI cupping analysis ---
    cupping_notes: Optional[CuppingNotes] = Field(None, description="Gemini 2.5 Flash structured cupping evaluation")

    # --- Bilingual narratives ---
    summary_text: BilingualSummary = Field(..., description="English and Amharic narrative quality summaries")

    # --- Actionable recommendations ---
    recommendations: List[str] = Field(
        default_factory=list,
        description="2-4 actionable steps for the producer or washing station"
    )

    # --- Spec Section 15.3 AI transparency ---
    limitations: List[str] = Field(
        default_factory=lambda: [
            "Preliminary AI assessment only. Not an official ECX or SCA grade.",
            "Only 3 of 13 ECX defect classes are detected: Full Black, Partial Black, Coffee Husk / Parchment.",
            "Defects hidden from the camera surface cannot be detected.",
            "Counts require manual verification before any commercial or export decision."
        ],
        description="Mandatory transparency statements about model limitations"
    )

    # --- Detection coverage transparency ---
    coverage: DetectionCoverage = Field(
        default_factory=lambda: DetectionCoverage(
            classes_scanned=["Full Black", "Partial Black", "Coffee Husk / Parchment"],
            classes_not_scanned=[
                "Full Sour", "Fungus / Mold", "Severe Insect Damage",
                "Floater / Chalky", "Immature / Quaker",
                "Broken / Chipped", "Foreign Matter"
            ],
            coverage_note="7 of 10 ECX defect classes are not scanned. Actual defect count may be higher than reported."
        ),
        description="Explicit record of which ECX defect classes this model version covers"
    )

    # --- Optional full YOLO data (backend can strip before DB save) ---
    yolo_detections: Optional[YOLODetectionResult] = Field(
        None,
        description="Full bounding box detection data. Backend strips base64 and stores S3 key separately."
    )

    # --- Filled by backend after S3 upload, null from our service ---
    annotated_image_url: Optional[str] = Field(
        None,
        description="S3 URL of annotated image. Null from AI service — backend fills after upload."
    )


# -------------------------------------------------------------
# Health Check Response
# -------------------------------------------------------------
class HealthCheckResponse(BaseModel):
    status: str
    service: str
    version: str
    gemini_configured: bool
    yolo_model_loaded: bool
    yolo_model_path: Optional[str] = None


# -------------------------------------------------------------
# Error Response (spec Section 12.5)
# -------------------------------------------------------------
class ErrorDetail(BaseModel):
    code: str = Field(..., description="Machine-readable error code e.g. 'INVALID_IMAGE'")
    message: str = Field(..., description="Human-readable error explanation")

class ErrorResponse(BaseModel):
    error: ErrorDetail
