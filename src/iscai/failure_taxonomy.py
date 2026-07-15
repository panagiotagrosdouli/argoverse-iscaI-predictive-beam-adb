from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class FailureTaxonomyThresholds:
    direction_error_deg: float = 20.0
    small_direction_error_deg: float = 5.0
    large_fde_m: float = 8.0
    good_ade_m: float = 2.0
    topk_target: float = 0.95
    adb_target: float = 0.95
    high_uncertainty_deg: float = 10.0


TAXONOMY_ORDER = (
    "direction_failure",
    "range_failure",
    "beam_allocation_failure",
    "uncertainty_failure",
    "mixed_failure",
    "no_system_failure",
)


def classify_failure(
    row: Mapping[str, object],
    thresholds: FailureTaxonomyThresholds | None = None,
) -> str:
    """Assign one mutually exclusive metric-based failure class.

    The taxonomy is intentionally conservative and uses only the aggregate
    per-target metrics produced by Exp06/Exp07. It does not claim semantic scene
    causes such as occlusion, turning, or lane changes.
    """
    t = thresholds or FailureTaxonomyThresholds()

    angular = float(row["cv_angular_error_deg"])
    ade = float(row["cv_ade_m"])
    fde = float(row["cv_fde_m"])
    topk = float(row["topk_coverage"])
    adb = float(row["adb_shadow_coverage"])
    uncertainty = float(row["mean_angular_std_deg"])

    beam_failure = topk < t.topk_target
    adb_failure = adb < t.adb_target
    prediction_failure = fde >= t.large_fde_m or angular >= t.direction_error_deg

    if angular >= t.direction_error_deg and fde >= t.large_fde_m:
        return "direction_failure"

    if angular <= t.small_direction_error_deg and fde >= t.large_fde_m:
        return "range_failure"

    if ade <= t.good_ade_m and beam_failure:
        return "beam_allocation_failure"

    if ade <= t.good_ade_m and (adb_failure or uncertainty >= t.high_uncertainty_deg):
        return "uncertainty_failure"

    if prediction_failure or beam_failure or adb_failure:
        return "mixed_failure"

    return "no_system_failure"


def taxonomy_description(label: str) -> str:
    descriptions = {
        "direction_failure": "Large directional and final-displacement error.",
        "range_failure": "Small angular error but large final-displacement error.",
        "beam_allocation_failure": "Low Cartesian error but insufficient Top-K coverage.",
        "uncertainty_failure": "Low Cartesian error but inadequate ADB coverage or high predicted uncertainty.",
        "mixed_failure": "Failure with no single dominant metric-based mechanism.",
        "no_system_failure": "Meets the configured failure thresholds.",
    }
    return descriptions[label]
