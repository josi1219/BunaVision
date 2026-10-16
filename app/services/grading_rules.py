"""
grading_rules.py — ECX Defect Point Equivalency Engine
=======================================================

Source of truth: Ethiopian Commodity Exchange (ECX) grading standards
as documented by BunaUtopia (bunautopia.com) and the Coffee Liquoring Unit (CLU).

KEY DISTINCTION — Photo-based vs. laboratory 300g sample:
    Official ECX grading uses a calibrated 300g physical sample.
    BunaVision analyses a photo, so we cannot measure exact sample weight.
    This engine applies the official ECX DEFECT POINT EQUIVALENCY RULES
    to the bean counts we detect, producing an honest estimated grade.
    The result is always presented as a preliminary AI assessment.

ECX Defect Point Equivalency Table (per official standard):
┌─────────────────────────┬────────────┬────────────────────────────┐
│ Defect Class            │ Category   │ Beans needed for 1 point   │
├─────────────────────────┼────────────┼────────────────────────────┤
│ Full Black              │ Primary    │ 1 bean = 1 primary point   │
│ Full Sour               │ Primary    │ 1 bean = 1 primary point   │
│ Pod / Dried Cherry      │ Primary    │ 1 pod  = 1 primary point   │
│ Fungus / Mold           │ Primary    │ 1 bean = 1 primary point   │
│ Severe Insect Damage    │ Primary    │ 1 bean = 1 primary point   │
│ Foreign Matter          │ Primary    │ 1 piece= 1 primary point   │
├─────────────────────────┼────────────┼────────────────────────────┤
│ Partial Black           │ Secondary  │ 3 beans = 1 secondary point│
│ Partial Sour            │ Secondary  │ 3 beans = 1 secondary point│
│ Parchment / Hull Piece  │ Secondary  │ 5 pieces= 1 secondary point│
│ Floater / Chalky        │ Secondary  │ 5 beans = 1 secondary point│
│ Immature / Quaker       │ Secondary  │ 5 beans = 1 secondary point│
│ Broken / Chipped / Cut  │ Secondary  │ 5 beans = 1 secondary point│
│ Slight Insect Damage    │ Secondary  │ 10 beans= 1 secondary point│
└─────────────────────────┴────────────┴────────────────────────────┘

ECX Grade Thresholds (per 300g sample — applied to estimated defect points):
    Grade 1 (Specialty — Outstanding):  0 primary points,  < 3 secondary points
    Grade 2 (Specialty — Premium):      0 primary points,  ≤ 6 secondary points
    Grade 3 (Commercial Specialty):     ≤ 2 primary points, ≤ 12 secondary points
    Grade 4 (Commercial Standard):      > 2 primary points OR > 12 secondary points
    Grade 5 / Local (Undergrade):       High defect concentration, not export eligible
    Ungraded:                           Exceeds Grade 5 tolerance — local market only
"""

from typing import Dict, List, NamedTuple


# ---------------------------------------------------------------------------
# Defect point equivalency — how many raw bean instances = 1 ECX defect point
# Source: ECX / Coffee Liquoring Unit official grading standard
# ---------------------------------------------------------------------------
PRIMARY_EQUIVALENCY: Dict[str, int] = {
    "Full Black":             1,   # 1 bean = 1 primary point
    "Full Sour":              1,
    "Pod / Dried Cherry":     1,
    "Fungus / Mold":          1,
    "Severe Insect Damage":   1,
    "Foreign Matter":         1,
}

SECONDARY_EQUIVALENCY: Dict[str, int] = {
    "Partial Black":          3,   # 3 beans = 1 secondary point
    "Partial Sour":           3,
    "Coffee Husk / Parchment": 5,  # 5 pieces = 1 secondary point
    "Floater / Chalky":       5,
    "Immature / Quaker":      5,
    "Broken / Chipped":       5,
    "Slight Insect Damage":   10,  # 10 beans = 1 secondary point
}

# Fallback divisors for classes not individually listed above
# (applies to any defect class our model detects that isn't in the tables)
DEFAULT_PRIMARY_DIVISOR = 1
DEFAULT_SECONDARY_DIVISOR = 3


# ---------------------------------------------------------------------------
# Result type — keeps return values explicit and readable
# ---------------------------------------------------------------------------
class GradingResult(NamedTuple):
    estimated_grade: str        # e.g. "Grade 1"
    grade_class: str            # e.g. "Specialty"
    is_export_eligible: bool
    grade_basis: str            # short explanation string
    primary_defect_points: float
    secondary_defect_points: float
    recommendations: List[str]
    detection_note: str         # coverage warning for the response


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------
def calculate_defect_points(
    raw_counts: Dict[str, int],
    category_map: Dict[str, str],
) -> tuple[float, float]:
    """
    Converts raw detected bean counts into ECX defect points
    using the equivalency table.

    Args:
        raw_counts:   { "Full Black": 2, "Partial Black": 9, ... }
        category_map: { "Full Black": "Primary Defect", "Partial Black": "Secondary Defect", ... }

    Returns:
        (primary_points, secondary_points) as floats (rounded to 2 dp)

    Example:
        2 Full Black beans   → 2.0 primary points   (2 / 1)
        9 Partial Black      → 3.0 secondary points  (9 / 3)
        2 Coffee Husk pieces → 0.4 secondary points  (2 / 5)
        Total secondary      → 3.4 secondary points
    """
    primary_pts = 0.0
    secondary_pts = 0.0

    for class_name, count in raw_counts.items():
        category = category_map.get(class_name, "Secondary Defect")

        if category == "Primary Defect":
            divisor = PRIMARY_EQUIVALENCY.get(class_name, DEFAULT_PRIMARY_DIVISOR)
            primary_pts += count / divisor
        else:
            divisor = SECONDARY_EQUIVALENCY.get(class_name, DEFAULT_SECONDARY_DIVISOR)
            secondary_pts += count / divisor

    return round(primary_pts, 2), round(secondary_pts, 2)


def calculate_grade(
    total_beans: int,
    primary_defects_count: int,
    secondary_defects_count: int,
    raw_counts: Dict[str, int] = None,
    category_map: Dict[str, str] = None,
) -> GradingResult:
    """
    Assigns an ECX grade using official defect point equivalency.

    Preferred call: pass raw_counts + category_map for accurate equivalency math.
    Fallback call:  pass only primary_defects_count + secondary_defects_count for
                    backward-compatibility (treats each detected bean as 1 point —
                    less accurate but still functional).

    ECX Thresholds applied:
        Grade 1: 0 primary points  AND  secondary_points < 3
        Grade 2: 0 primary points  AND  secondary_points ≤ 6
        Grade 3: primary_points ≤ 2  AND  secondary_points ≤ 12
        Grade 4: primary_points ≤ 4  AND  secondary_points ≤ 25
        Grade 5: primary_points ≤ 8  OR   secondary_points > 25
        Ungraded: exceeds Grade 5 tolerance
    """
    if total_beans == 0:
        return GradingResult(
            estimated_grade="Unknown",
            grade_class="Undetermined",
            is_export_eligible=False,
            grade_basis="No beans detected in image.",
            primary_defect_points=0.0,
            secondary_defect_points=0.0,
            recommendations=["Please provide a clear image containing visible green coffee beans."],
            detection_note="No beans detected — cannot apply grading rules.",
        )

    # --- Compute defect points ---
    if raw_counts and category_map:
        # Full equivalency math using the per-class table
        primary_pts, secondary_pts = calculate_defect_points(raw_counts, category_map)
        calc_method = "ECX equivalency table"
    else:
        # Fallback: treat each raw count as 1 point (less accurate)
        primary_pts = float(primary_defects_count)
        secondary_pts = float(secondary_defects_count)
        calc_method = "raw count fallback (no per-class breakdown available)"

    # --- Coverage warning (always shown — our model only covers 3 classes) ---
    detection_note = (
        "Grade estimated from 3 of 13 ECX defect classes "
        "(Full Black, Partial Black, Coffee Husk / Parchment). "
        "Unscanned classes: Full Sour, Fungus/Mold, Insect Damage, "
        "Floater, Immature, Broken, Foreign Matter. "
        "Actual defect count may be higher."
    )

    # --- Grade assignment (ECX thresholds on defect points, not raw %) ---
    recommendations: List[str] = []

    if primary_pts == 0 and secondary_pts < 3:
        grade = "Grade 1"
        grade_class = "Specialty"
        export_eligible = True
        basis = (
            f"0 primary defect points, {secondary_pts} secondary defect points "
            f"(threshold: 0 primary, < 3 secondary). Calculated via {calc_method}."
        )
        recommendations.append(
            "Outstanding specialty quality. Suitable for premium single-origin export."
        )
        recommendations.append(
            "Maintain current cherry selection and drying protocols to preserve Grade 1 consistency."
        )

    elif primary_pts == 0 and secondary_pts <= 6:
        grade = "Grade 2"
        grade_class = "Specialty"
        export_eligible = True
        basis = (
            f"0 primary defect points, {secondary_pts} secondary defect points "
            f"(threshold: 0 primary, ≤ 6 secondary). Calculated via {calc_method}."
        )
        recommendations.append(
            "High specialty grade. Hand-sort secondary defects to achieve Grade 1 for premium markets."
        )
        recommendations.append(
            f"{secondary_pts} secondary defect points detected — "
            "target < 3 points for Grade 1 upgrade."
        )

    elif primary_pts <= 2 and secondary_pts <= 12:
        grade = "Grade 3"
        grade_class = "Commercial Specialty"
        export_eligible = True
        basis = (
            f"{primary_pts} primary defect points, {secondary_pts} secondary defect points "
            f"(threshold: ≤ 2 primary, ≤ 12 secondary). Calculated via {calc_method}."
        )
        recommendations.append(
            "Commercial export grade. Eliminate primary defects via density sorting or hand-picking."
        )
        if primary_pts > 0:
            recommendations.append(
                f"{primary_pts} primary defect point(s) present. "
                "Investigate cherry selection at intake — primary defects often indicate "
                "over-fermentation or diseased cherry."
            )

    elif primary_pts <= 4 and secondary_pts <= 25:
        grade = "Grade 4"
        grade_class = "Commercial"
        export_eligible = True
        basis = (
            f"{primary_pts} primary defect points, {secondary_pts} secondary defect points "
            f"(threshold: ≤ 4 primary, ≤ 25 secondary). Calculated via {calc_method}."
        )
        recommendations.append(
            "Standard commercial grade. Requires extensive sorting before auction."
        )
        recommendations.append(
            "Review pulping and fermentation controls — high defect levels suggest "
            "processing issues at the washing station."
        )

    elif primary_pts <= 8:
        grade = "Grade 5"
        grade_class = "Commercial"
        export_eligible = True
        basis = (
            f"{primary_pts} primary defect points, {secondary_pts} secondary defect points "
            f"(threshold: ≤ 8 primary). Calculated via {calc_method}."
        )
        recommendations.append(
            "Low commercial grade. Acceptable for bulk commodity blending only."
        )
        recommendations.append(
            "Significant sorting and processing improvements required before next harvest."
        )

    else:
        grade = "Ungraded"
        grade_class = "Local Market Only"
        export_eligible = False
        basis = (
            f"{primary_pts} primary defect points, {secondary_pts} secondary defect points "
            f"— exceeds all ECX export thresholds. Calculated via {calc_method}."
        )
        recommendations.append(
            "Defect concentration exceeds Grade 5 tolerance. Not eligible for ECX export."
        )
        recommendations.append(
            "This lot is suitable for local market only. Conduct a full post-harvest audit "
            "to identify the source of defects (cherry quality, fermentation, drying)."
        )

    return GradingResult(
        estimated_grade=grade,
        grade_class=grade_class,
        is_export_eligible=export_eligible,
        grade_basis=basis,
        primary_defect_points=primary_pts,
        secondary_defect_points=secondary_pts,
        recommendations=recommendations,
        detection_note=detection_note,
    )
