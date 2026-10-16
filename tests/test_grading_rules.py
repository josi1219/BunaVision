"""
Unit tests for ECX defect point equivalency and grading rules.
"""
from app.services.grading_rules import calculate_defect_points, calculate_grade

def test_calculate_defect_points_equivalency():
    raw_counts = {
        "Full Black": 2,
        "Partial Black": 9,
        "Coffee Husk / Parchment": 2
    }
    cat_map = {
        "Full Black": "Primary Defect",
        "Partial Black": "Secondary Defect",
        "Coffee Husk / Parchment": "Secondary Defect"
    }
    primary_pts, secondary_pts = calculate_defect_points(raw_counts, cat_map)
    # 2 Full Black / 1 = 2.0 primary
    # 9 Partial Black / 3 = 3.0 secondary
    # 2 Husk / 5 = 0.4 secondary
    assert primary_pts == 2.0
    assert secondary_pts == 3.4

def test_grade_1_specialty():
    # 0 primary, 6 partial black (6/3 = 2.0 secondary pts < 3) -> Grade 1
    res = calculate_grade(
        total_beans=80,
        primary_defects_count=0,
        secondary_defects_count=6,
        raw_counts={"Partial Black": 6},
        category_map={"Partial Black": "Secondary Defect"}
    )
    assert res.estimated_grade == "Grade 1"
    assert res.grade_class == "Specialty"
    assert res.is_export_eligible is True
    assert res.primary_defect_points == 0.0
    assert res.secondary_defect_points == 2.0

def test_grade_2_specialty():
    # 0 primary, 15 partial black (15/3 = 5.0 secondary pts <= 6) -> Grade 2
    res = calculate_grade(
        total_beans=80,
        primary_defects_count=0,
        secondary_defects_count=15,
        raw_counts={"Partial Black": 15},
        category_map={"Partial Black": "Secondary Defect"}
    )
    assert res.estimated_grade == "Grade 2"
    assert res.grade_class == "Specialty"
    assert res.is_export_eligible is True
    assert res.secondary_defect_points == 5.0

def test_grade_3_commercial_export():
    # 2 full black (2 pts <= 2), 9 partial black (3 pts <= 12) -> Grade 3
    res = calculate_grade(
        total_beans=80,
        primary_defects_count=2,
        secondary_defects_count=9,
        raw_counts={"Full Black": 2, "Partial Black": 9},
        category_map={"Full Black": "Primary Defect", "Partial Black": "Secondary Defect"}
    )
    assert res.estimated_grade == "Grade 3"
    assert res.grade_class == "Commercial Specialty"
    assert res.is_export_eligible is True
    assert res.primary_defect_points == 2.0
    assert res.secondary_defect_points == 3.0

def test_zero_beans_detected():
    res = calculate_grade(total_beans=0, primary_defects_count=0, secondary_defects_count=0)
    assert res.estimated_grade == "Unknown"
    assert res.is_export_eligible is False
