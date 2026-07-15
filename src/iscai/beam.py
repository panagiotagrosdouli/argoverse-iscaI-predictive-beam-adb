from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass(frozen=True)
class BeamCodebook:
    min_angle_rad: float
    max_angle_rad: float
    num_beams: int

    def __post_init__(self) -> None:
        if self.num_beams <= 0:
            raise ValueError("num_beams must be positive")
        if self.max_angle_rad <= self.min_angle_rad:
            raise ValueError("max_angle_rad must exceed min_angle_rad")

    @property
    def edges(self) -> np.ndarray:
        return np.linspace(self.min_angle_rad, self.max_angle_rad, self.num_beams + 1)

    @property
    def centers(self) -> np.ndarray:
        edges = self.edges
        return 0.5 * (edges[:-1] + edges[1:])

    def angle_to_index(self, angle_rad: float | np.ndarray) -> np.ndarray:
        angle = np.asarray(angle_rad, dtype=float)
        raw = np.searchsorted(self.edges, angle, side="right") - 1
        return np.clip(raw, 0, self.num_beams - 1)


def angular_variance_from_xy(mean_xy: np.ndarray, covariance_xy: np.ndarray) -> float:
    """Approximate azimuth variance using first-order covariance propagation."""
    x, y = np.asarray(mean_xy, dtype=float)
    covariance = np.asarray(covariance_xy, dtype=float)
    radius_sq = x * x + y * y
    if radius_sq <= 1e-12:
        return float(np.pi**2)
    jacobian = np.array([-y / radius_sq, x / radius_sq])
    return float(max(0.0, jacobian @ covariance @ jacobian.T))


def gaussian_beam_probabilities(
    mean_angle_rad: float,
    std_angle_rad: float,
    codebook: BeamCodebook,
) -> np.ndarray:
    """Integrate a Gaussian azimuth distribution over each beam interval."""
    std = max(float(std_angle_rad), 1e-6)
    z = (codebook.edges - float(mean_angle_rad)) / std
    probabilities = np.diff(norm.cdf(z))
    probabilities = np.maximum(probabilities, 0.0)
    total = probabilities.sum()
    if total <= 0:
        probabilities[codebook.angle_to_index(mean_angle_rad)] = 1.0
        return probabilities
    return probabilities / total


def adaptive_topk(probabilities: np.ndarray, coverage_threshold: float = 0.95) -> np.ndarray:
    """Return the smallest beam-index set reaching the requested probability mass."""
    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim != 1 or len(probs) == 0:
        raise ValueError("probabilities must be a non-empty 1D array")
    if not 0 < coverage_threshold <= 1:
        raise ValueError("coverage_threshold must be in (0, 1]")
    total = probs.sum()
    if total <= 0:
        raise ValueError("probabilities must have positive mass")
    normalized = probs / total
    order = np.argsort(normalized)[::-1]
    cumulative = np.cumsum(normalized[order])
    count = int(np.searchsorted(cumulative, coverage_threshold, side="left") + 1)
    return np.sort(order[:count])


def topk_contains(indices: np.ndarray, true_beam_index: int) -> bool:
    return bool(np.any(np.asarray(indices, dtype=int) == int(true_beam_index)))
