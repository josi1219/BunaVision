import io
import json
import base64
import logging
from typing import Optional
from PIL import Image
from app.config import settings
from app.schemas import (
    CoffeeAssessmentResponse,
    DefectBreakdown,
    DefectItem,
    YOLODetectionResult
)
from app.services.grading_rules import calculate_grade

logger = logging.getLogger("bunavision.gemini")

COFFEE_SYSTEM_PROMPT = """
You are an expert Q-Grader and green coffee quality assessment AI specializing in Ethiopian Green Coffee (ECX - Ethiopian Commodity Exchange and SCA standards).
Your task is to analyze the provided green coffee bean sample image and generate an objective cupping evaluation and grading report.

Categories according to Ethiopian ECX / SCA Standards:
1. Primary Defects (Category 1 - severe, zero tolerance in Grade 1):
   - Full Black (totally darkened/black bean)
   - Full Sour / Stinker (yellowish-brown, ferment odor appearance)
   - Pod / Dried Cherry (unpulped dry husk)
   - Fungus / Mold (white/grey/green spores)
   - Severe Insect Damage / Berry Borer (multiple puncture holes)
   - Foreign matter (stones, sticks)

2. Secondary Defects (Category 2 - minor):
   - Partial Black (partially darkened bean)
   - Partial Sour (light brown/tan discoloration)
   - Parchment / Hull piece
   - Floater / Chalky White (sponge-like, bleached white)
   - Immature / Quaker (thin, greenish-yellow, boat-shaped)
   - Broken / Chipped / Cut (nicked or fragmented bean)
   - Slight Insect Damage (single small pinhole)

Output strictly structured JSON with:
- total_beans_detected (integer)
- healthy_beans_count (integer)
- primary_defects_count (integer)
- secondary_defects_count (integer)
- total_defects_count (integer)
- defect_percentage (float)
- grade (string, e.g. 'Grade 1 (Specialty)', 'Grade 2 (Specialty)', 'Grade 3 (Commercial Export)', etc.)
- export_eligible (boolean)
- defects: {
    "primary_defects": [{"name": str, "category": "Primary", "count": int, "description": str}],
    "secondary_defects": [{"name": str, "category": "Secondary", "count": int, "description": str}]
  }
- quality_summary: Professional English cupper summary explaining visual findings and cup impact.
- amharic_summary: Professional quality summary in Amharic (የቡና ጥራት ደረጃ ማጠቃለያ በአማርኛ ለኢትዮጵያ አምራቾች).
- recommendations: 2-4 actionable recommendations for washing, drying bed management, or hand sorting.
"""

def analyze_coffee_sample(
    image_bytes: bytes,
    yolo_result: Optional[YOLODetectionResult] = None
) -> CoffeeAssessmentResponse:
    """
    Analyzes an uploaded green coffee image using Gemini 2.0 Flash.
    If yolo_result is provided with detected beans, chains the YOLO detections
    and annotated image into Gemini for maximum accuracy and visual grounding.
    """
    if not settings.GEMINI_API_KEY:
        raise ValueError(
            "GEMINI_API_KEY is not configured in .env. Please add your key to D:\\Coding\\Hackathon\\.env"
        )

    # Validate image format using PIL
    try:
        pil_img = Image.open(io.BytesIO(image_bytes))
        pil_img.verify()
        pil_img = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        raise ValueError(f"Invalid image file: {str(e)}")

    # Prepare image payload: prefer annotated image if YOLO produced one
    active_image_bytes = image_bytes
    if yolo_result and yolo_result.annotated_image_base64:
        try:
            active_image_bytes = base64.b64decode(yolo_result.annotated_image_base64)
            logger.info("Chaining YOLO-annotated image (with bounding boxes) into Gemini 2.0 Flash.")
        except Exception as e:
            logger.warning(f"Could not decode YOLO annotated image: {e}")
            active_image_bytes = image_bytes

    # Build prompt: augment with verified YOLO findings if available
    prompt_text = COFFEE_SYSTEM_PROMPT
    if yolo_result and yolo_result.total_beans > 0:
        prompt_text += f"""

VERIFIED YOLO OBJECT DETECTION DATA:
A local YOLOv8 neural network localized the coffee beans and rendered colored bounding boxes on the image:
- Total Beans: {yolo_result.total_beans}
- Healthy Beans: {yolo_result.healthy_beans}
- Defective Beans: {yolo_result.defect_beans}
- Detected Breakdown: {json.dumps(yolo_result.defect_counts)}

Bounding Box Color Scheme on the image:
- Red boxes: Primary Defects (Severe)
- Amber/Yellow boxes: Secondary Defects (Minor)
- Green boxes: Healthy Beans

CRITICAL: Ground your response in these verified object counts. Align total_beans_detected, healthy_beans_count, and the defects breakdown with these figures, and reference the boxed visual defects in your English and Amharic summaries.
"""

    # Prefer the official google-genai SDK, fallback to direct REST API if needed
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=settings.GEMINI_API_KEY)
        
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
        raw_text = response.text
    except Exception as e:
        if "API_KEY_INVALID" in str(e) or "PERMISSION_DENIED" in str(e):
            raise ValueError(f"Gemini API key error: {str(e)}")
        
        # Fallback to direct HTTP request using requests
        import requests
        img_b64 = base64.b64encode(active_image_bytes).decode("utf-8")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{settings.MODEL_NAME}:generateContent?key={settings.GEMINI_API_KEY}"
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt_text},
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": img_b64
                            }
                        }
                    ]
                }
            ],
            "generationConfig": {
                "response_mime_type": "application/json",
                "temperature": 0.1
            }
        }
        res = requests.post(url, json=payload, timeout=45)
        res.raise_for_status()
        data = res.json()
        raw_text = data["candidates"][0]["content"]["parts"][0]["text"]

    # Parse and validate response
    try:
        data_dict = json.loads(raw_text)
        
        total_beans = data_dict.get("total_beans_detected", 0)
        primary = data_dict.get("primary_defects_count", 0)
        secondary = data_dict.get("secondary_defects_count", 0)
        
        computed_grade, is_export, rule_recs = calculate_grade(total_beans, primary, secondary)
        
        if "grade" not in data_dict or not data_dict["grade"]:
            data_dict["grade"] = computed_grade
        if "export_eligible" not in data_dict:
            data_dict["export_eligible"] = is_export

        return CoffeeAssessmentResponse(**data_dict)
    except Exception as e:
        logger.error(f"Failed to parse Gemini output: {raw_text}")
        raise RuntimeError(f"Error parsing model assessment response: {str(e)}")
