from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .beam import angular_variance_from_xy


@dataclass(frozen=True)
class ShadowInterval:
    center_rad: float
    lower_rad: float
    upper_rad: float
    geometric_margin_rad: float
    uncertainty_margin_rad: float

    @property
    def width_rad(self) -> float:
        return self.upper_rad - self.lower_rad


def predictive_shadow_interval(
    mean_xy: np.ndarray,
    covariance_xy: np.ndarray | None = None,
    object_width_m: float = 2.0,
    base_margin_deg: float = 0.5,
    confidence_z: float = 1.96,
) -> ShadowInterval:
    """Construct an uncertainty-aware angular shadow zone for one actor.

    The geometric half-width is approximated from actor width and range. The
    uncertainty term is propagated from Cartesian covariance to azimuth.
    """
    mean = np.asarray(mean_xy, dtype=float)
    if mean.shape != (2,):
        raise ValueError("mean_xy must have shape (2,)")
    range_m = float(np.linalg.norm(mean))
    if range_m <= 1e-6:
        raise ValueError("Actor range must be positive")

    center = float(np.arctan2(mean[1], mean[0]))
    geometric = float(np.arctan2(object_width_m / 2.0, range_m))
    base = float(np.deg2rad(base_margin_deg))

    uncertainty = 0.0
    if covariance_xy is not None:
        angular_variance = angular_variance_from_xy(mean, covariance_xy)
        uncertainty = confidence_z * float(np.sqrt(angular_variance))

    half_width = geometric + base + uncertainty
    return ShadowInterval(
        center_rad=center,
        lower_rad=center - half_width,
        upper_rad=center + half_width,
        geometric_margin_rad=geometric + base,
        uncertainty_margin_rad=uncertainty,
    )


def shadow_contains(interval: ShadowInterval, actor_angle_rad: float) -> bool:
    return interval.lower_rad <= actor_angle_rad <= interval.upper_rad
