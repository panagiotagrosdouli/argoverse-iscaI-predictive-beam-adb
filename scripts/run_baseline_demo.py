from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from iscai.adb import predictive_shadow_interval, shadow_contains  # noqa: E402
from iscai.beam import (  # noqa: E402
    BeamCodebook,
    adaptive_topk,
    angular_variance_from_xy,
    gaussian_beam_probabilities,
)
from iscai.data import extract_actor_tracks, find_ego_track, load_scenario_parquet  # noqa: E402
from iscai.evaluation import ade, fde, mean_angular_error_deg  # noqa: E402
from iscai.geometry import polar_from_xy, to_ego_coordinates  # noqa: E402
from iscai.prediction import constant_velocity, kalman_constant_velocity  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an AV2 trajectory-to-beam/ADB baseline")
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/demo"))
    parser.add_argument("--observation-steps", type=int, default=50)
    parser.add_argument("--future-steps", type=int, default=60)
    parser.add_argument("--num-beams", type=int, default=16)
    parser.add_argument("--field-of-view-deg", type=float, default=120.0)
    parser.add_argument("--coverage", type=float, default=0.95)
    return parser.parse_args()


def select_target(frame: pd.DataFrame, tracks: dict, required_steps: int):
    if "track_category" in frame.columns:
        focal_ids = frame.loc[
            frame["track_category"].astype(str).str.upper().eq("FOCAL_TRACK"), "track_id"
        ].astype(str)
        for track_id in focal_ids.unique():
            if track_id in tracks and len(tracks[track_id].position) >= required_steps:
                return tracks[track_id]

    candidates = [
        track
        for track in tracks.values()
        if track.object_type.upper() not in {"AV", "EGO", "AUTONOMOUS_VEHICLE"}
        and len(track.position) >= required_steps
    ]
    if not candidates:
        raise ValueError("No non-ego actor has enough samples for the requested split")
    return max(candidates, key=lambda track: len(track.position))


def state_at(track, timestep: int) -> tuple[np.ndarray, float]:
    matches = np.flatnonzero(track.timestep == timestep)
    if len(matches) == 0:
        nearest = int(np.argmin(np.abs(track.timestep - timestep)))
        index = nearest
    else:
        index = int(matches[0])
    return track.position[index], float(track.heading[index])


def main() -> None:
    args = parse_args()
    required = args.observation_steps + args.future_steps
    frame = load_scenario_parquet(args.scenario)
    tracks = extract_actor_tracks(frame)
    ego = find_ego_track(tracks)
    target = select_target(frame, tracks, required)

    target_position = target.position[:required]
    target_timestep = target.timestep[:required]
    current_timestep = int(target_timestep[args.observation_steps - 1])
    ego_origin, ego_heading = state_at(ego, current_timestep)
    target_ego = to_ego_coordinates(target_position, ego_origin, ego_heading)

    observed = target_ego[: args.observation_steps]
    future = target_ego[args.observation_steps : required]

    cv_prediction = constant_velocity(observed, args.future_steps)
    kalman = kalman_constant_velocity(observed, args.future_steps)

    cv_metrics = {
        "ade_m": ade(cv_prediction, future),
        "fde_m": fde(cv_prediction, future),
        "mean_angular_error_deg": mean_angular_error_deg(cv_prediction, future),
    }
    kalman_metrics = {
        "ade_m": ade(kalman.mean, future),
        "fde_m": fde(kalman.mean, future),
        "mean_angular_error_deg": mean_angular_error_deg(kalman.mean, future),
    }

    half_fov = np.deg2rad(args.field_of_view_deg / 2.0)
    codebook = BeamCodebook(-half_fov, half_fov, args.num_beams)
    _, true_angles = polar_from_xy(future)
    _, predicted_angles = polar_from_xy(kalman.mean)

    topk_coverage: list[bool] = []
    topk_sizes: list[int] = []
    shadow_coverage: list[bool] = []
    predicted_beams: list[int] = []
    true_beams: list[int] = []

    for mean_xy, covariance_xy, predicted_angle, true_angle in zip(
        kalman.mean, kalman.covariance, predicted_angles, true_angles
    ):
        angular_std = np.sqrt(angular_variance_from_xy(mean_xy, covariance_xy))
        probabilities = gaussian_beam_probabilities(predicted_angle, angular_std, codebook)
        selected = adaptive_topk(probabilities, args.coverage)
        true_beam = int(codebook.angle_to_index(true_angle))
        predicted_beam = int(codebook.angle_to_index(predicted_angle))

        topk_coverage.append(bool(true_beam in selected))
        topk_sizes.append(len(selected))
        predicted_beams.append(predicted_beam)
        true_beams.append(true_beam)

        interval = predictive_shadow_interval(mean_xy, covariance_xy)
        shadow_coverage.append(shadow_contains(interval, float(true_angle)))

    beam_metrics = {
        "top1_accuracy": float(np.mean(np.equal(predicted_beams, true_beams))),
        "topk_coverage": float(np.mean(topk_coverage)),
        "average_k": float(np.mean(topk_sizes)),
        "overhead_reduction": float(1.0 - np.mean(topk_sizes) / args.num_beams),
        "adb_shadow_coverage": float(np.mean(shadow_coverage)),
    }

    result = {
        "scenario": str(args.scenario),
        "target_track_id": target.track_id,
        "target_object_type": target.object_type,
        "constant_velocity": cv_metrics,
        "kalman": kalman_metrics,
        "beam_and_adb": beam_metrics,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    figure, axis = plt.subplots(figsize=(9, 7))
    axis.plot(observed[:, 0], observed[:, 1], label="Observed", linewidth=2)
    axis.plot(future[:, 0], future[:, 1], label="Ground truth", linewidth=2)
    axis.plot(cv_prediction[:, 0], cv_prediction[:, 1], label="Constant velocity")
    axis.plot(kalman.mean[:, 0], kalman.mean[:, 1], label="Kalman mean")
    axis.scatter([0], [0], marker="*", s=140, label="Ego origin")
    axis.set_xlabel("Longitudinal x [m]")
    axis.set_ylabel("Lateral y [m]")
    axis.set_title("Trajectory prediction in the current ego frame")
    axis.axis("equal")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(args.output_dir / "trajectory_prediction.png", dpi=180)
    plt.close(figure)

    print(json.dumps(result, indent=2))
    print(f"Outputs saved under: {args.output_dir}")


if __name__ == "__main__":
    main()
