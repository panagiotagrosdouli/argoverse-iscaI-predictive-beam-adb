from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2


@dataclass(frozen=True)
class CoveragePoint:
    nominal: float
    observed: float
    absolute_error: float


@dataclass(frozen=True)
class CalibrationSummary:
    points: tuple[CoveragePoint, ...]
    expected_calibration_error: float
    maximum_calibration_error: float
    mean_nll: float
    mean_mahalanobis_sq: float


def mahalanobis_squared(
    truth_xy: np.ndarray,
    mean_xy: np.ndarray,
    covariance_xy: np.ndarray,
    jitter: float = 1e-6,
) -> np.ndarray:
    """Return squared Mahalanobis distances for 2D Gaussian predictions."""
    truth = np.asarray(truth_xy, dtype=float)
    mean = np.asarray(mean_xy, dtype=float)
    covariance = np.asarray(covariance_xy, dtype=float)
    if truth.shape != mean.shape or truth.ndim != 2 or truth.shape[1] != 2:
        raise ValueError("truth_xy and mean_xy must have shape (T, 2)")
    if covariance.shape != (len(truth), 2, 2):
        raise ValueError("covariance_xy must have shape (T, 2, 2)")

    values: list[float] = []
    for residual, cov in zip(truth - mean, covariance):
        stabilized = cov + jitter * np.eye(2)
        values.append(float(residual @ np.linalg.solve(stabilized, residual)))
    return np.asarray(values)


def gaussian_nll_2d(
    truth_xy: np.ndarray,
    mean_xy: np.ndarray,
    covariance_xy: np.ndarray,
    jitter: float = 1e-6,
) -> np.ndarray:
    """Return per-step negative log likelihood under 2D Gaussian predictions."""
    truth = np.asarray(truth_xy, dtype=float)
    mean = np.asarray(mean_xy, dtype=float)
    covariance = np.asarray(covariance_xy, dtype=float)
    distances = mahalanobis_squared(truth, mean, covariance, jitter=jitter)

    values: list[float] = []
    for distance, cov in zip(distances, covariance):
        stabilized = cov + jitter * np.eye(2)
        sign, logdet = np.linalg.slogdet(stabilized)
        if sign <= 0:
            raise ValueError("Covariance must be positive definite")
        values.append(float(np.log(2.0 * np.pi) + 0.5 * logdet + 0.5 * distance))
    return np.asarray(values)


def empirical_coverage(
    mahalanobis_sq: np.ndarray,
    nominal_levels: np.ndarray | list[float] | tuple[float, ...],
) -> tuple[CoveragePoint, ...]:
    """Measure empirical 2D confidence-ellipse coverage at nominal levels."""
    distances = np.asarray(mahalanobis_sq, dtype=float)
    if distances.ndim != 1 or len(distances) == 0:
        raise ValueError("mahalanobis_sq must be a non-empty 1D array")

    points: list[CoveragePoint] = []
    for nominal in nominal_levels:
        level = float(nominal)
        if not 0.0 < level < 1.0:
            raise ValueError("nominal levels must lie in (0, 1)")
        threshold = float(chi2.ppf(level, df=2))
        observed = float(np.mean(distances <= threshold))
        points.append(CoveragePoint(level, observed, abs(observed - level)))
    return tuple(points)


def summarize_calibration(
    truth_xy: np.ndarray,
    mean_xy: np.ndarray,
    covariance_xy: np.ndarray,
    nominal_levels: tuple[float, ...] = (0.50, 0.68, 0.80, 0.90, 0.95, 0.99),
) -> CalibrationSummary:
    distances = mahalanobis_squared(truth_xy, mean_xy, covariance_xy)
    nll = gaussian_nll_2d(truth_xy, mean_xy, covariance_xy)
    points = empirical_coverage(distances, nominal_levels)
    errors = np.asarray([point.absolute_error for point in points], dtype=float)
    return CalibrationSummary(
        points=points,
        expected_calibration_error=float(errors.mean()),
        maximum_calibration_error=float(errors.max()),
        mean_nll=float(nll.mean()),
        mean_mahalanobis_sq=float(distances.mean()),
    )
