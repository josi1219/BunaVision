from typing import List, Optional, Dict
from pydantic import BaseModel, Field

# -------------------------------------------------------------
# YOLO Computer Vision Schemas
# -------------------------------------------------------------
class BoundingBox(BaseModel):
    x1: float = Field(..., description="Top-left X coordinate (pixels or normalized)")
    y1: float = Field(..., description="Top-left Y coordinate (pixels or normalized)")
    x2: float = Field(..., description="Bottom-right X coordinate (pixels or normalized)")
    y2: float = Field(..., description="Bottom-right Y coordinate (pixels or normalized)")

class DetectedBean(BaseModel):
    box: BoundingBox = Field(..., description="Bounding box coordinates")
    class_name: str = Field(..., description="Defect or bean class: healthy, black, broken, insect_damaged, sour")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence score")
    category: str = Field(..., description="'Healthy', 'Primary Defect', or 'Secondary Defect'")

class YOLODetectionResult(BaseModel):
    model_version: str = Field(..., description="Model identifier, e.g. 'YOLOv8n-Coffee-v1' or 'Mock-Detector'")
    total_beans: int = Field(..., ge=0, description="Total bean instances localized")
    healthy_beans: int = Field(..., ge=0, description="Count of healthy beans")
    defect_beans: int = Field(..., ge=0, description="Count of defective beans")
    defect_counts: Dict[str, int] = Field(default_factory=dict, description="Count per individual defect class")
    detected_beans: List[DetectedBean] = Field(default_factory=list, description="List of localized beans with boxes")
    annotated_image_base64: Optional[str] = Field(None, description="Base64-encoded JPEG image with colored bounding boxes")

# -------------------------------------------------------------
# Defect & Grading Schemas
# -------------------------------------------------------------
class DefectItem(BaseModel):
    name: str = Field(..., description="Name of defect, e.g., 'Full Black', 'Broken / Chipped', 'Insect Damaged'")
    category: str = Field(..., description="'Primary' or 'Secondary'")
    count: int = Field(..., ge=0, description="Number of beans exhibiting this defect")
    description: str = Field(..., description="Short explanation of visible characteristics")

class DefectBreakdown(BaseModel):
    primary_defects: List[DefectItem] = Field(default_factory=list, description="Category 1 / Severe defects")
    secondary_defects: List[DefectItem] = Field(default_factory=list, description="Category 2 / Minor defects")

class CoffeeAssessmentResponse(BaseModel):
    total_beans_detected: int = Field(..., ge=0, description="Total count of visible green coffee beans")
    healthy_beans_count: int = Field(..., ge=0, description="Number of sound, defect-free green beans")
    primary_defects_count: int = Field(..., ge=0, description="Total count of primary defect beans")
    secondary_defects_count: int = Field(..., ge=0, description="Total count of secondary defect beans")
    total_defects_count: int = Field(..., ge=0, description="Sum of primary and secondary defects")
    defect_percentage: float = Field(..., ge=0.0, le=100.0, description="Percentage of defective beans")
    
    # Grading according to Ethiopian ECX / Specialty standards
    grade: str = Field(..., description="e.g. 'Grade 1 (Specialty)', 'Grade 2 (Specialty)', 'Grade 3 (Commercial)', 'Grade 4', 'Grade 5 / Local'")
    export_eligible: bool = Field(..., description="True if coffee meets international specialty or commercial export grade")
    
    # Detailed breakdown
    defects: DefectBreakdown = Field(..., description="Itemized list of detected defects")
    
    # Optional YOLO visual detection data (Bounding Boxes & Annotated Image)
    yolo_detections: Optional[YOLODetectionResult] = Field(None, description="Detailed bounding boxes and annotated image")

    # Natural language explanations & advisory
    quality_summary: str = Field(..., description="Professional cupper/grader overview of the sample")
    amharic_summary: str = Field(..., description="ጥራት ማጠቃለያ እና ደረጃ በአማርኛ ለኢትዮጵያ አምራቾች")
    recommendations: List[str] = Field(default_factory=list, description="Actionable steps for the washing station or producer to improve quality")

class HealthCheckResponse(BaseModel):
    status: str
    service: str
    version: str
    gemini_configured: bool
    yolo_model_loaded: bool
    yolo_model_path: Optional[str] = None
