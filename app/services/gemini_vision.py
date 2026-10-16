"""
gemini_vision.py — Gemini 2.0 Flash Multimodal Analysis Service
================================================================

Role in the pipeline:
    Stage 3 of 3. Receives the YOLO-annotated image + verified defect counts,
    then uses Gemini 2.0 Flash to:
        1. Estimate total_beans and healthy_beans from the full image
           (YOLO only counted defects — it cannot see healthy beans)
        2. Generate structured SCA-style cupping notes (aroma, flavor, etc.)
        3. Write professional English and Amharic quality summaries
        4. Produce actionable producer recommendations

    ECX grade is then OVERRIDDEN by the deterministic grading_rules engine
    to guarantee mathematical compliance regardless of what Gemini guesses.
"""

import io
import json
import base64
import logging
from datetime import datetime, timezone
from typing import Optional

from PIL import Image

from app.config import settings
from app.schemas import (
    CoffeeAssessmentResponse,
    AssessmentSummary,
    DefectPrediction,
    GradeAssessment,
    CuppingNotes,
    BilingualSummary,
    DetectionCoverage,
    YOLODetectionResult,
)
from app.services.grading_rules import calculate_grade, GradingResult

logger = logging.getLogger("bunavision.gemini")

# ---------------------------------------------------------------------------
# Detection coverage constants — what our current model can and cannot see
# ---------------------------------------------------------------------------
CLASSES_SCANNED = ["Full Black", "Partial Black", "Coffee Husk / Parchment"]
CLASSES_NOT_SCANNED = [
    "Full Sour", "Fungus / Mold", "Severe Insect Damage",
    "Floater / Chalky", "Immature / Quaker", "Broken / Chipped", "Foreign Matter",
]

# Category map for defect point calculation
YOLO_CATEGORY_MAP = {
    "Full Black":              "Primary Defect",
    "Partial Black":           "Secondary Defect",
    "Coffee Husk / Parchment": "Secondary Defect",
}

# Limitations shown on every response (spec Section 15.3)
STANDARD_LIMITATIONS = [
    "Preliminary AI assessment only. Not an official ECX or SCA grade.",
    f"Only {len(CLASSES_SCANNED)} of 13 ECX defect classes are detected: "
    f"{', '.join(CLASSES_SCANNED)}.",
    "Defects hidden from the camera surface cannot be detected.",
    "Counts require manual verification before any commercial or export decision.",
]

# ---------------------------------------------------------------------------
# System prompt — instructs Gemini on EXACTLY what JSON to return
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """
You are a certified Q-Grader and Ethiopian green coffee quality specialist with expertise
in ECX (Ethiopian Commodity Exchange) and SCA (Specialty Coffee Association) standards.

You are receiving a green coffee bean sample image with colored bounding boxes already
drawn by a YOLOv8 computer vision model:
  - RED boxes    = Primary Defects (Full Black beans)
  - AMBER boxes  = Secondary Defects (Partial Black, Coffee Husk / Parchment)
  - GREEN boxes  = Healthy beans (if visible)

IMPORTANT — YOUR SPECIFIC TASKS:
1. Estimate the TOTAL number of green coffee beans visible in the image (healthy + defective).
2. Estimate the number of HEALTHY beans (the YOLO model only detected defects,
   so YOU must estimate healthy_beans by looking at the full image).
3. Provide structured SCA cupping notes based on what visible defects suggest about cup quality.
4. Write bilingual summaries for producers.

OUTPUT FORMAT — return ONLY valid JSON with exactly these fields:

{
  "total_beans_estimated": <integer — your visual estimate of ALL beans in the image>,
  "healthy_beans_estimated": <integer — your estimate of defect-free beans>,

  "cupping_notes": {
    "aroma": "<describe expected aroma based on defect presence>",
    "flavor": "<describe expected flavor profile>",
    "acidity": "<describe acidity quality>",
    "body": "<describe body/mouthfeel>",
    "aftertaste": "<describe finish>",
    "estimated_sca_score": <float between 60 and 100>
  },

  "quality_summary_en": "<2-3 sentence professional English summary referencing the boxed defects>",
  "quality_summary_am": "<2-3 sentences in Amharic (አማርኛ) for Ethiopian producers — የቡና ጥራት ማጠቃለያ>",

  "recommendations": [
    "<recommendation 1 — specific and actionable>",
    "<recommendation 2 — specific and actionable>",
    "<recommendation 3 — optional>"
  ]
}

RULES:
- Do NOT invent defect classes not visible in the image.
- Do NOT assign a final grade — the grading engine handles that deterministically.
- Ground healthy_beans_estimated in what you actually see in the image.
- Cupping notes should reflect how detected defects (black beans, partial fermentation,
  husks) would impact the cup — be specific to what the boxes show.
- Both summaries must reference the bounding box colors actually visible.
- Amharic summary must be genuine Amharic, not transliterated English.
"""


def _build_grounding_block(yolo_result: YOLODetectionResult) -> str:
    """
    Builds the YOLO grounding text appended to the prompt.
    Tells Gemini exactly what the deterministic model found so it
    cannot hallucinate different counts.
    """
    defect_lines = "\n".join(
        f"    - {cls}: {count} instance(s)"
        for cls, count in yolo_result.defect_counts.items()
    ) or "    - No defects detected by YOLO above confidence threshold"

    return f"""

=== VERIFIED YOLO DETECTION DATA (ground your response in these numbers) ===

The YOLOv8 model has already localized defects and drawn bounding boxes on the image.
These counts are mathematically verified — do NOT contradict them:

  Total defect instances detected by YOLO : {yolo_result.defect_beans}
  Defect breakdown by class:
{defect_lines}

  YOLO healthy_beans value: -1 (model cannot count healthy beans — YOU estimate this)

Your job:
  - Estimate total_beans_estimated  (all beans visible, healthy + defective)
  - Estimate healthy_beans_estimated (total_beans_estimated minus verified defects)
  - The YOLO defect counts above are the ground truth for defect counts

Bounding box colour scheme in the image:
  RED   boxes = Primary Defects (Full Black)
  AMBER boxes = Secondary Defects (Partial Black, Coffee Husk / Parchment)
  GREEN boxes = Healthy beans
=============================================================================
"""


def _call_gemini_sdk(
    client,
    pil_img: Image.Image,
    active_image_bytes: bytes,
    prompt_text: str,
) -> str:
    """Call Gemini via the official google-genai SDK."""
    from google.genai import types

    mime = Image.MIME.get(pil_img.format, "image/jpeg")
    image_part = types.Part.from_bytes(data=active_image_bytes, mime_type=mime)

    response = client.models.generate_content(
        model=settings.MODEL_NAME,
        contents=[image_part, prompt_text],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.1,
        ),
    )
    return response.text


def _call_gemini_rest(
    active_image_bytes: bytes,
    prompt_text: str,
) -> str:
    """Fallback: call Gemini via direct HTTP REST if the SDK fails."""
    import requests

    img_b64 = base64.b64encode(active_image_bytes).decode("utf-8")
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{settings.MODEL_NAME}:generateContent?key={settings.GEMINI_API_KEY}"
    )
    payload = {
        "contents": [{
            "parts": [
                {"text": prompt_text},
                {"inline_data": {"mime_type": "image/jpeg", "data": img_b64}},
            ]
        }],
        "generationConfig": {
            "response_mime_type": "application/json",
            "temperature": 0.1,
        },
    }
    res = requests.post(url, json=payload, timeout=45)
    res.raise_for_status()
    return res.json()["candidates"][0]["content"]["parts"][0]["text"]


def _build_predictions(yolo_result: YOLODetectionResult) -> list[DefectPrediction]:
    """
    Converts YOLO defect_counts dict into the spec predictions array.
    Computes per-class average confidence from detected_beans list.
    """
    # Build confidence lookup: class_name → [confidence values]
    conf_map: dict[str, list[float]] = {}
    for bean in yolo_result.detected_beans:
        conf_map.setdefault(bean.class_name, []).append(bean.confidence)

    # Raw class name for each display name (reverse lookup)
    raw_class_map = {
        "Full Black":              "hitam",
        "Partial Black":           "hitam-sebagian",
        "Coffee Husk / Parchment": "kulit-kopi-kecil",
    }

    predictions = []
    for class_name, count in yolo_result.defect_counts.items():
        confs = conf_map.get(class_name, [])
        avg_conf = round(sum(confs) / len(confs), 3) if confs else 0.0
        category = YOLO_CATEGORY_MAP.get(class_name, "Secondary Defect")
        raw = raw_class_map.get(class_name, class_name.lower().replace(" ", "_"))

        predictions.append(DefectPrediction(
            class_name=class_name,
            raw_class=raw,
            category=category,
            confidence=avg_conf,
            count=count,
        ))

    return predictions


def analyze_coffee_sample(
    image_bytes: bytes,
    yolo_result: Optional[YOLODetectionResult] = None,
    assessment_id: str = "asmt_unknown",
    created_at: str = "",
) -> CoffeeAssessmentResponse:
    """
    Stage 3 of the BunaVision pipeline.

    Accepts:
        image_bytes    — original uploaded image bytes
        yolo_result    — output from yolo_detector (defect counts + annotated image)
        assessment_id  — UUID from main.py (echoed into response)
        created_at     — ISO timestamp from main.py (echoed into response)

    Returns:
        CoffeeAssessmentResponse — fully populated, matching spec Section 12.1

    Pipeline:
        1. Validate API key & image
        2. Choose image: annotated (with boxes) > original
        3. Build grounded prompt with YOLO counts
        4. Call Gemini (SDK → REST fallback)
        5. Parse Gemini JSON
        6. Apply deterministic ECX grading_rules (overrides Gemini's grade guess)
        7. Assemble and return CoffeeAssessmentResponse
    """
    # --- 1. Validate API key ---
    if not settings.GEMINI_API_KEY:
        raise ValueError(
            "GEMINI_API_KEY is not configured. "
            "Add it to your .env file (see .env.example for the required key name)."
        )

    # --- 2. Validate image ---
    try:
        pil_img = Image.open(io.BytesIO(image_bytes))
        pil_img.verify()
        pil_img = Image.open(io.BytesIO(image_bytes))  # re-open after verify() consumes stream
    except Exception as e:
        raise ValueError(f"Cannot decode image: {str(e)}")

    # --- 3. Prefer annotated image (boxes already drawn by YOLO) ---
    active_image_bytes = image_bytes
    if yolo_result and yolo_result.annotated_image_base64:
        try:
            active_image_bytes = base64.b64decode(yolo_result.annotated_image_base64)
            logger.info(f"[{assessment_id}] Using YOLO-annotated image for Gemini input")
        except Exception as e:
            logger.warning(f"[{assessment_id}] Could not decode annotated image: {e} — using original")

    # --- 4. Build prompt ---
    prompt_text = _SYSTEM_PROMPT
    if yolo_result and yolo_result.defect_beans > 0:
        prompt_text += _build_grounding_block(yolo_result)
    elif yolo_result and yolo_result.defect_beans == 0:
        prompt_text += (
            "\n\n=== YOLO RESULT: No defects detected above confidence threshold ===\n"
            "Your visual inspection may find additional defects the model missed.\n"
            "Estimate total_beans_estimated and healthy_beans_estimated from what you see.\n"
        )

    # --- 5. Call Gemini ---
    raw_text = ""
    try:
        from google import genai
        client = genai.Client(api_key=settings.GEMINI_API_KEY)
        raw_text = _call_gemini_sdk(client, pil_img, active_image_bytes, prompt_text)
        logger.info(f"[{assessment_id}] Gemini SDK call succeeded")
    except Exception as sdk_err:
        if "API_KEY_INVALID" in str(sdk_err) or "PERMISSION_DENIED" in str(sdk_err):
            raise ValueError(f"Gemini API key error: {sdk_err}")
        logger.warning(
            f"[{assessment_id}] Gemini SDK failed ({sdk_err}), trying REST fallback"
        )
        try:
            raw_text = _call_gemini_rest(active_image_bytes, prompt_text)
            logger.info(f"[{assessment_id}] Gemini REST fallback succeeded")
        except Exception as rest_err:
            raise RuntimeError(
                f"Gemini API unavailable. SDK error: {sdk_err}. REST error: {rest_err}"
            )

    # --- 6. Parse Gemini JSON ---
    try:
        gemini_data = json.loads(raw_text)
    except json.JSONDecodeError as e:
        logger.error(f"[{assessment_id}] Gemini returned non-JSON: {raw_text[:500]}")
        raise RuntimeError(f"Gemini returned invalid JSON: {e}")

    # --- 7. Extract Gemini estimates ---
    total_beans = int(gemini_data.get("total_beans_estimated", 0))
    healthy_beans = int(gemini_data.get("healthy_beans_estimated", 0))

    # Defect counts come from YOLO (ground truth), not Gemini
    yolo_defect_counts = yolo_result.defect_counts if yolo_result else {}
    primary_count = sum(
        count for cls, count in yolo_defect_counts.items()
        if YOLO_CATEGORY_MAP.get(cls) == "Primary Defect"
    )
    secondary_count = sum(
        count for cls, count in yolo_defect_counts.items()
        if YOLO_CATEGORY_MAP.get(cls) == "Secondary Defect"
    )
    total_defects = primary_count + secondary_count
    defect_pct = round((total_defects / total_beans * 100) if total_beans > 0 else 0.0, 2)

    # --- 8. Run deterministic ECX grading engine ---
    grading: GradingResult = calculate_grade(
        total_beans=total_beans,
        primary_defects_count=primary_count,
        secondary_defects_count=secondary_count,
        raw_counts=yolo_defect_counts,
        category_map=YOLO_CATEGORY_MAP,
    )

    # --- 9. Build cupping notes from Gemini output ---
    cn_raw = gemini_data.get("cupping_notes", {})
    cupping_notes = CuppingNotes(
        aroma=cn_raw.get("aroma"),
        flavor=cn_raw.get("flavor"),
        acidity=cn_raw.get("acidity"),
        body=cn_raw.get("body"),
        aftertaste=cn_raw.get("aftertaste"),
        estimated_sca_score=cn_raw.get("estimated_sca_score"),
    )

    # --- 10. Build predictions array from YOLO counts ---
    predictions = _build_predictions(yolo_result) if yolo_result else []

    # --- 11. Build final response ---
    return CoffeeAssessmentResponse(
        assessment_id=assessment_id,
        status="completed",
        model_version="YOLOv8n-Coffee-ONNX-v1",
        created_at=created_at or datetime.now(timezone.utc).isoformat(),
        processing_time_ms=0,  # filled in by main.py after this call returns

        summary=AssessmentSummary(
            total_beans_detected=total_beans,
            healthy_beans_count=healthy_beans,
            defect_beans_count=total_defects,
            primary_defects_count=primary_count,
            secondary_defects_count=secondary_count,
            defect_percentage=defect_pct,
            detected_classes=len(yolo_defect_counts),
            detection_count=yolo_result.defect_beans if yolo_result else 0,
        ),

        predictions=predictions,

        grade=GradeAssessment(
            estimated_grade=grading.estimated_grade,
            grade_class=grading.grade_class,
            is_export_eligible=grading.is_export_eligible,
            grade_basis=grading.grade_basis,
            grade_confidence="partial",  # always partial — model covers 3/13 ECX classes
        ),

        cupping_notes=cupping_notes,

        summary_text=BilingualSummary(
            english=gemini_data.get("quality_summary_en", ""),
            amharic=gemini_data.get("quality_summary_am", ""),
        ),

        recommendations=grading.recommendations + gemini_data.get("recommendations", []),

        limitations=STANDARD_LIMITATIONS,

        coverage=DetectionCoverage(
            classes_scanned=CLASSES_SCANNED,
            classes_not_scanned=CLASSES_NOT_SCANNED,
            coverage_note=grading.detection_note,
        ),

        yolo_detections=None,       # attached by main.py if include_yolo=True
        annotated_image_url=None,   # filled by backend after S3 upload
    )
