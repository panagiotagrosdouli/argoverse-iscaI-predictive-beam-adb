from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RiskConfig:
    """Weights and operating limits for inference-time trajectory risk.

    The estimator uses only quantities available before the future ground truth is
    observed: predicted angular uncertainty, recent motion non-linearity, target
    range, and proximity to the beam codebook field-of-view boundary.
    """

    angular_std_reference_deg: float = 8.0
    acceleration_reference_mps2: float = 4.0
    turn_rate_reference_deg_s: float = 20.0
    close_range_reference_m: float = 15.0
    fov_margin_reference_deg: float = 10.0
    angular_weight: float = 0.35
    acceleration_weight: float = 0.20
    turn_rate_weight: float = 0.20
    close_range_weight: float = 0.15
    fov_edge_weight: float = 0.10


@dataclass(frozen=True)
class RiskFeatures:
    angular_uncertainty: float
    acceleration: float
    turn_rate: float
    close_range: float
    fov_edge: float


@dataclass(frozen=True)
class RiskEstimate:
    score: float
    features: RiskFeatures
    coverage_threshold: float
    adb_sigma_scale: float


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def observed_motion_features(observed_xy: np.ndarray, dt: float = 0.1) -> tuple[float, float]:
    """Return recent acceleration magnitude and turn rate from observed positions."""
    points = np.asarray(observed_xy, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("observed_xy must have shape [time, 2]")
    if len(points) < 4:
        return 0.0, 0.0

    velocity = np.diff(points, axis=0) / dt
    acceleration = np.diff(velocity, axis=0) / dt
    acceleration_magnitude = float(np.linalg.norm(acceleration[-3:], axis=1).mean())

    headings = np.unwrap(np.arctan2(velocity[:, 1], velocity[:, 0]))
    heading_rate = np.diff(headings) / dt
    turn_rate_deg_s = float(np.rad2deg(np.abs(heading_rate[-3:])).mean())
    return acceleration_magnitude, turn_rate_deg_s


def estimate_risk(
    observed_xy: np.ndarray,
    predicted_xy: np.ndarray,
    angular_std_rad: float,
    min_fov_rad: float,
    max_fov_rad: float,
    config: RiskConfig | None = None,
) -> RiskEstimate:
    """Estimate normalized motion/control risk using inference-time information."""
    cfg = config or RiskConfig()
    prediction = np.asarray(predicted_xy, dtype=float)
    if prediction.shape != (2,):
        raise ValueError("predicted_xy must have shape [2]")

    acceleration, turn_rate = observed_motion_features(observed_xy)
    range_m = float(np.linalg.norm(prediction))
    angle_rad = float(np.arctan2(prediction[1], prediction[0]))
    edge_margin_rad = min(angle_rad - min_fov_rad, max_fov_rad - angle_rad)

    angular_feature = _clip01(
        np.rad2deg(max(float(angular_std_rad), 0.0)) / cfg.angular_std_reference_deg
    )
    acceleration_feature = _clip01(acceleration / cfg.acceleration_reference_mps2)
    turn_rate_feature = _clip01(turn_rate / cfg.turn_rate_reference_deg_s)
    close_range_feature = _clip01(
        (cfg.close_range_reference_m - range_m) / cfg.close_range_reference_m
    )
    fov_edge_feature = _clip01(
        (cfg.fov_margin_reference_deg - np.rad2deg(edge_margin_rad))
        / cfg.fov_margin_reference_deg
    )

    features = RiskFeatures(
        angular_uncertainty=angular_feature,
        acceleration=acceleration_feature,
        turn_rate=turn_rate_feature,
        close_range=close_range_feature,
        fov_edge=fov_edge_feature,
    )
    score = _clip01(
        cfg.angular_weight * features.angular_uncertainty
        + cfg.acceleration_weight * features.acceleration
        + cfg.turn_rate_weight * features.turn_rate
        + cfg.close_range_weight * features.close_range
        + cfg.fov_edge_weight * features.fov_edge
    )

    if score < 0.25:
        coverage = 0.90
        adb_sigma_scale = 1.64
    elif score < 0.50:
        coverage = 0.95
        adb_sigma_scale = 1.96
    elif score < 0.75:
        coverage = 0.98
        adb_sigma_scale = 2.33
    else:
        coverage = 0.995
        adb_sigma_scale = 2.58

    return RiskEstimate(
        score=score,
        features=features,
        coverage_threshold=coverage,
        adb_sigma_scale=adb_sigma_scale,
    )
