from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GaussianTrajectory:
    mean: np.ndarray
    covariance: np.ndarray


def constant_position(observed_xy: np.ndarray, future_steps: int) -> np.ndarray:
    observed = np.asarray(observed_xy, dtype=float)
    if observed.ndim != 2 or observed.shape[1] != 2 or len(observed) < 1:
        raise ValueError("observed_xy must have shape (T, 2)")
    return np.repeat(observed[-1][None, :], future_steps, axis=0)


def constant_velocity(
    observed_xy: np.ndarray,
    future_steps: int,
    dt: float = 0.1,
    velocity_window: int = 5,
) -> np.ndarray:
    """Predict a trajectory using the mean recent velocity."""
    observed = np.asarray(observed_xy, dtype=float)
    if observed.ndim != 2 or observed.shape[1] != 2 or len(observed) < 2:
        raise ValueError("observed_xy must have shape (T, 2) with T >= 2")
    window = min(max(2, velocity_window), len(observed))
    velocity = np.diff(observed[-window:], axis=0).mean(axis=0) / dt
    horizons = np.arange(1, future_steps + 1, dtype=float)[:, None] * dt
    return observed[-1] + horizons * velocity


def kalman_constant_velocity(
    observed_xy: np.ndarray,
    future_steps: int,
    dt: float = 0.1,
    process_variance: float = 1.0,
    measurement_variance: float = 0.25,
) -> GaussianTrajectory:
    """Fit a linear constant-velocity Kalman model and forecast mean/covariance."""
    z = np.asarray(observed_xy, dtype=float)
    if z.ndim != 2 or z.shape[1] != 2 or len(z) < 2:
        raise ValueError("observed_xy must have shape (T, 2) with T >= 2")

    f = np.array(
        [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]],
        dtype=float,
    )
    h = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
    q = process_variance * np.array(
        [
            [dt**4 / 4, 0, dt**3 / 2, 0],
            [0, dt**4 / 4, 0, dt**3 / 2],
            [dt**3 / 2, 0, dt**2, 0],
            [0, dt**3 / 2, 0, dt**2],
        ],
        dtype=float,
    )
    r = measurement_variance * np.eye(2)

    initial_velocity = (z[1] - z[0]) / dt
    x = np.array([z[0, 0], z[0, 1], initial_velocity[0], initial_velocity[1]])
    p = np.diag([1.0, 1.0, 10.0, 10.0])
    identity = np.eye(4)

    for measurement in z:
        x = f @ x
        p = f @ p @ f.T + q
        innovation = measurement - h @ x
        s = h @ p @ h.T + r
        k = p @ h.T @ np.linalg.inv(s)
        x = x + k @ innovation
        p = (identity - k @ h) @ p

    means: list[np.ndarray] = []
    covariances: list[np.ndarray] = []
    for _ in range(future_steps):
        x = f @ x
        p = f @ p @ f.T + q
        means.append(x[:2].copy())
        covariances.append(p[:2, :2].copy())

    return GaussianTrajectory(np.stack(means), np.stack(covariances))
