import numpy as np

from iscai.adb import predictive_shadow_interval, shadow_contains
from iscai.beam import BeamCodebook, adaptive_topk, gaussian_beam_probabilities
from iscai.evaluation import ade, fde
from iscai.geometry import polar_from_xy, to_ego_coordinates
from iscai.prediction import constant_velocity, kalman_constant_velocity


def test_ego_transform_and_polar() -> None:
    actor = np.array([[11.0, 5.0]])
    ego = np.array([10.0, 5.0])
    transformed = to_ego_coordinates(actor, ego, 0.0)
    ranges, angles = polar_from_xy(transformed)
    assert np.allclose(transformed, [[1.0, 0.0]])
    assert np.allclose(ranges, [1.0])
    assert np.allclose(angles, [0.0])


def test_constant_velocity_prediction() -> None:
    observed = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    predicted = constant_velocity(observed, future_steps=2, dt=1.0)
    expected = np.array([[3.0, 0.0], [4.0, 0.0]])
    assert np.allclose(predicted, expected)
    assert ade(predicted, expected) == 0.0
    assert fde(predicted, expected) == 0.0


def test_kalman_shapes() -> None:
    observed = np.column_stack([np.arange(10, dtype=float), np.zeros(10)])
    forecast = kalman_constant_velocity(observed, future_steps=5, dt=1.0)
    assert forecast.mean.shape == (5, 2)
    assert forecast.covariance.shape == (5, 2, 2)
    assert np.all(np.linalg.eigvalsh(forecast.covariance) >= -1e-10)


def test_adaptive_topk_and_shadow() -> None:
    codebook = BeamCodebook(-np.pi / 3, np.pi / 3, 16)
    probabilities = gaussian_beam_probabilities(0.0, 0.03, codebook)
    selected = adaptive_topk(probabilities, 0.95)
    assert len(selected) >= 1
    assert int(codebook.angle_to_index(0.0)) in selected

    interval = predictive_shadow_interval(
        mean_xy=np.array([30.0, 1.0]),
        covariance_xy=np.diag([0.2, 0.2]),
    )
    true_angle = float(np.arctan2(1.0, 30.0))
    assert shadow_contains(interval, true_angle)
