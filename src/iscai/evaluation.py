from __future__ import annotations

import numpy as np

from .geometry import polar_from_xy, wrap_angle


def _validate_pair(predicted_xy: np.ndarray, target_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    predicted = np.asarray(predicted_xy, dtype=float)
    target = np.asarray(target_xy, dtype=float)
    if predicted.shape != target.shape or predicted.ndim != 2 or predicted.shape[1] != 2:
        raise ValueError("predicted_xy and target_xy must have equal shape (T, 2)")
    return predicted, target


def ade(predicted_xy: np.ndarray, target_xy: np.ndarray) -> float:
    predicted, target = _validate_pair(predicted_xy, target_xy)
    return float(np.linalg.norm(predicted - target, axis=1).mean())


def fde(predicted_xy: np.ndarray, target_xy: np.ndarray) -> float:
    predicted, target = _validate_pair(predicted_xy, target_xy)
    return float(np.linalg.norm(predicted[-1] - target[-1]))


def mean_angular_error_deg(predicted_xy: np.ndarray, target_xy: np.ndarray) -> float:
    predicted, target = _validate_pair(predicted_xy, target_xy)
    _, predicted_angle = polar_from_xy(predicted)
    _, target_angle = polar_from_xy(target)
    error = np.abs(wrap_angle(predicted_angle - target_angle))
    return float(np.rad2deg(error).mean())


def interval_iou(predicted: tuple[float, float], target: tuple[float, float]) -> float:
    pred_low, pred_high = predicted
    true_low, true_high = target
    intersection = max(0.0, min(pred_high, true_high) - max(pred_low, true_low))
    union = max(pred_high, true_high) - min(pred_low, true_low)
    return float(intersection / union) if union > 0 else 1.0
