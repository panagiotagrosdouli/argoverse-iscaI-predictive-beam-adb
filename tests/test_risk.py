import numpy as np

from iscai.risk import estimate_risk, observed_motion_features


def test_straight_constant_velocity_has_low_motion_risk() -> None:
    observed = np.column_stack([np.linspace(-5.0, 0.0, 50), np.zeros(50)])
    acceleration, turn_rate = observed_motion_features(observed)
    assert acceleration < 1e-6
    assert turn_rate < 1e-6


def test_risk_increases_with_angular_uncertainty() -> None:
    observed = np.column_stack([np.linspace(-5.0, 0.0, 50), np.zeros(50)])
    low = estimate_risk(
        observed_xy=observed,
        predicted_xy=np.array([30.0, 0.0]),
        angular_std_rad=np.deg2rad(0.5),
        min_fov_rad=np.deg2rad(-60.0),
        max_fov_rad=np.deg2rad(60.0),
    )
    high = estimate_risk(
        observed_xy=observed,
        predicted_xy=np.array([30.0, 0.0]),
        angular_std_rad=np.deg2rad(12.0),
        min_fov_rad=np.deg2rad(-60.0),
        max_fov_rad=np.deg2rad(60.0),
    )
    assert high.score > low.score
    assert high.coverage_threshold >= low.coverage_threshold
    assert high.adb_sigma_scale >= low.adb_sigma_scale


def test_fov_edge_increases_risk() -> None:
    observed = np.column_stack([np.linspace(-5.0, 0.0, 50), np.zeros(50)])
    center = estimate_risk(
        observed_xy=observed,
        predicted_xy=np.array([30.0, 0.0]),
        angular_std_rad=np.deg2rad(1.0),
        min_fov_rad=np.deg2rad(-60.0),
        max_fov_rad=np.deg2rad(60.0),
    )
    edge_angle = np.deg2rad(58.0)
    edge = estimate_risk(
        observed_xy=observed,
        predicted_xy=np.array([30.0 * np.cos(edge_angle), 30.0 * np.sin(edge_angle)]),
        angular_std_rad=np.deg2rad(1.0),
        min_fov_rad=np.deg2rad(-60.0),
        max_fov_rad=np.deg2rad(60.0),
    )
    assert edge.score > center.score
