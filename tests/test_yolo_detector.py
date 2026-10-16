"""
Unit tests for YOLO detector service and NMS.
"""
import numpy as np
from app.services.yolo_detector import yolo_service, nms

def test_classify_category_exact_matching():
    # Hitam -> Primary
    assert yolo_service.classify_category("hitam") == "Primary Defect"
    assert yolo_service.classify_category("black") == "Primary Defect"
    assert yolo_service.classify_category("full_black") == "Primary Defect"

    # Hitam-sebagian -> Secondary (CRITICAL BUG FIX)
    assert yolo_service.classify_category("hitam-sebagian") == "Secondary Defect"
    assert yolo_service.classify_category("partial_black") == "Secondary Defect"

    # Kulit-kopi-kecil -> Secondary
    assert yolo_service.classify_category("kulit-kopi-kecil") == "Secondary Defect"

    # Healthy
    assert yolo_service.classify_category("healthy") == "Healthy"

    # Unknown fallback
    assert yolo_service.classify_category("some_unseen_defect") == "Secondary Defect"

def test_nms_suppression():
    # Two identical boxes with different confidence
    boxes = np.array([
        [10.0, 10.0, 50.0, 50.0],
        [11.0, 11.0, 50.0, 50.0],
    ])
    scores = np.array([0.9, 0.7])
    keep = nms(boxes, scores, iou_threshold=0.45)
    # The lower-confidence overlapping box must be suppressed
    assert keep == [0]

def test_nms_different_boxes():
    # Two non-overlapping boxes
    boxes = np.array([
        [10.0, 10.0, 50.0, 50.0],
        [100.0, 100.0, 150.0, 150.0],
    ])
    scores = np.array([0.85, 0.80])
    keep = nms(boxes, scores, iou_threshold=0.45)
    # Both must be kept
    assert len(keep) == 2
