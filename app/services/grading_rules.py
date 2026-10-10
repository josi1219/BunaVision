from typing import List, Tuple
from app.schemas import DefectBreakdown, DefectItem

def calculate_grade(
    total_beans: int,
    primary_defects_count: int,
    secondary_defects_count: int
) -> Tuple[str, bool, List[str]]:
    """
    Evaluates coffee grade based on Ethiopian ECX & SCA defect thresholds.
    Standard green sample estimation (adapted for sample counts):
    - Grade 1 (Specialty): 0 primary defects, <= 3% secondary defects
    - Grade 2 (Specialty): 0 primary defects, <= 8% secondary defects
    - Grade 3 (Commercial Export): <= 2 primary defects, <= 15% secondary defects
    - Grade 4 (Commercial Standard): <= 4 primary defects, <= 25% secondary defects
    - Grade 5 / Local (Undergrade): > 4 primary defects or > 25% secondary defects
    """
    if total_beans == 0:
        return "Unknown (No beans detected)", False, ["Please provide an image containing green coffee beans."]

    sec_rate = (secondary_defects_count / total_beans) * 100.0

    recommendations = []

    if primary_defects_count == 0 and sec_rate <= 3.0:
        grade = "Grade 1 (Specialty - Outstanding)"
        export_eligible = True
        recommendations.append("Exceptional quality sample. Suitable for premium single-origin specialty export.")
    elif primary_defects_count == 0 and sec_rate <= 8.0:
        grade = "Grade 2 (Specialty - Premium)"
        export_eligible = True
        recommendations.append("High specialty grade. Minor secondary defect sorting will elevate it to Grade 1.")
    elif primary_defects_count <= 2 and sec_rate <= 15.0:
        grade = "Grade 3 (Commercial Export)"
        export_eligible = True
        recommendations.append("Commercial export grade. Eliminate primary defects via density or hand sorting.")
    elif primary_defects_count <= 4 and sec_rate <= 25.0:
        grade = "Grade 4 (Commercial Standard)"
        export_eligible = True
        recommendations.append("Standard commercial grade. Requires extensive secondary sorting before auction.")
    else:
        grade = "Grade 5 / Local Market (Undergrade)"
        export_eligible = False
        recommendations.append("High defect concentration. Not eligible for standard export; sold on local market.")

    return grade, export_eligible, recommendations
