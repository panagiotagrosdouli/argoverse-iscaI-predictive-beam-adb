from __future__ import annotations

import numpy as np


def rotation_matrix(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s], [s, c]], dtype=float)


def to_ego_coordinates(
    actor_xy: np.ndarray,
    ego_xy: np.ndarray,
    ego_heading_rad: float,
) -> np.ndarray:
    """Transform global/local-map points to the ego vehicle coordinate frame.

    The returned x-axis points along the ego heading and the y-axis to the left.
    ``ego_xy`` may be one point of shape ``(2,)`` or one point per actor sample.
    """
    actor = np.asarray(actor_xy, dtype=float)
    ego = np.asarray(ego_xy, dtype=float)
    if actor.shape[-1] != 2 or ego.shape[-1] != 2:
        raise ValueError("actor_xy and ego_xy must end in dimension 2")
    relative = actor - ego
    return relative @ rotation_matrix(-ego_heading_rad).T


def polar_from_xy(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(xy, dtype=float)
    if points.shape[-1] != 2:
        raise ValueError("xy must end in dimension 2")
    ranges = np.linalg.norm(points, axis=-1)
    angles = np.arctan2(points[..., 1], points[..., 0])
    return ranges, angles


def wrap_angle(angle_rad: np.ndarray | float) -> np.ndarray:
    angle = np.asarray(angle_rad, dtype=float)
    return (angle + np.pi) % (2 * np.pi) - np.pi
