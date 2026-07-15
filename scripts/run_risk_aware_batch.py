from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from iscai.adb import predictive_shadow_interval, shadow_contains
from iscai.beam import BeamCodebook, adaptive_topk, angular_variance_from_xy, gaussian_beam_probabilities
from iscai.data import extract_actor_tracks, find_ego_track, load_scenario_parquet
from iscai.geometry import polar_from_xy, to_ego_coordinates
from iscai.prediction import kalman_constant_velocity
from iscai.risk import estimate_risk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare fixed and risk-adaptive beam/ADB control on AV2"
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-scenarios", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/risk_aware"))
    parser.add_argument("--observation-steps", type=int, default=50)
    parser.add_argument("--future-steps", type=int, default=60)
    parser.add_argument("--num-beams", type=int, default=16)
    parser.add_argument("--field-of-view-deg", type=float, default=120.0)
    parser.add_argument("--fixed-coverage", type=float, default=0.95)
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
        track for track in tracks.values()
        if track.track_id.upper() not in {"AV", "EGO"}
        and len(track.position) >= required_steps
    ]
    if not candidates:
        raise ValueError("No target actor with enough samples")
    return max(candidates, key=lambda track: len(track.position))


def state_at(track, timestep: int) -> tuple[np.ndarray, float]:
    matches = np.flatnonzero(track.timestep == timestep)
    index = int(matches[0]) if len(matches) else int(np.argmin(np.abs(track.timestep - timestep)))
    return track.position[index], float(track.heading[index])


def evaluate_one(path: Path, args: argparse.Namespace) -> dict:
    required = args.observation_steps + args.future_steps
    frame = load_scenario_parquet(path)
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

    prediction = kalman_constant_velocity(observed, args.future_steps)
    half_fov = np.deg2rad(args.field_of_view_deg / 2.0)
    codebook = BeamCodebook(-half_fov, half_fov, args.num_beams)
    _, true_angles = polar_from_xy(future)
    _, predicted_angles = polar_from_xy(prediction.mean)

    fixed_hits: list[bool] = []
    fixed_sizes: list[int] = []
    fixed_adb_hits: list[bool] = []
    risk_hits: list[bool] = []
    risk_sizes: list[int] = []
    risk_adb_hits: list[bool] = []
    risk_scores: list[float] = []
    risk_coverages: list[float] = []

    for mean_xy, cov_xy, pred_angle, true_angle in zip(
        prediction.mean, prediction.covariance, predicted_angles, true_angles
    ):
        angular_std = float(np.sqrt(angular_variance_from_xy(mean_xy, cov_xy)))
        probabilities = gaussian_beam_probabilities(pred_angle, angular_std, codebook)
        true_beam = int(codebook.angle_to_index(true_angle))

        fixed_selected = adaptive_topk(probabilities, args.fixed_coverage)
        fixed_hits.append(bool(true_beam in fixed_selected))
        fixed_sizes.append(len(fixed_selected))
        fixed_interval = predictive_shadow_interval(mean_xy, cov_xy, confidence_z=1.96)
        fixed_adb_hits.append(shadow_contains(fixed_interval, float(true_angle)))

        risk = estimate_risk(
            observed_xy=observed,
            predicted_xy=mean_xy,
            angular_std_rad=angular_std,
            min_fov_rad=-half_fov,
            max_fov_rad=half_fov,
        )
        risk_selected = adaptive_topk(probabilities, risk.coverage_threshold)
        risk_hits.append(bool(true_beam in risk_selected))
        risk_sizes.append(len(risk_selected))
        risk_interval = predictive_shadow_interval(
            mean_xy, cov_xy, confidence_z=risk.adb_sigma_scale
        )
        risk_adb_hits.append(shadow_contains(risk_interval, float(true_angle)))
        risk_scores.append(risk.score)
        risk_coverages.append(risk.coverage_threshold)

    return {
        "scenario": str(path),
        "target_object_type": target.object_type,
        "fixed_topk_coverage": float(np.mean(fixed_hits)),
        "fixed_average_k": float(np.mean(fixed_sizes)),
        "fixed_overhead_reduction": float(1.0 - np.mean(fixed_sizes) / args.num_beams),
        "fixed_adb_coverage": float(np.mean(fixed_adb_hits)),
        "risk_topk_coverage": float(np.mean(risk_hits)),
        "risk_average_k": float(np.mean(risk_sizes)),
        "risk_overhead_reduction": float(1.0 - np.mean(risk_sizes) / args.num_beams),
        "risk_adb_coverage": float(np.mean(risk_adb_hits)),
        "mean_risk_score": float(np.mean(risk_scores)),
        "mean_requested_coverage": float(np.mean(risk_coverages)),
    }


def main() -> None:
    args = parse_args()
    paths = sorted((args.dataset_root / args.split).glob("*/scenario_*.parquet"))
    paths = paths[: args.max_scenarios]
    if not paths:
        raise FileNotFoundError("No scenario parquet files found")

    rows: list[dict] = []
    failures: list[dict] = []
    for index, path in enumerate(paths, start=1):
        try:
            rows.append(evaluate_one(path, args))
            print(f"[{index}/{len(paths)}] OK {path.parent.name}")
        except Exception as error:
            failures.append({"scenario": str(path), "error": str(error)})
            print(f"[{index}/{len(paths)}] SKIP {path.parent.name}: {error}")

    if not rows:
        raise RuntimeError("All scenarios failed")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "per_scenario_risk_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    numeric_keys = [key for key in rows[0] if key not in {"scenario", "target_object_type"}]
    means = {key: float(np.mean([float(row[key]) for row in rows])) for key in numeric_keys}
    summary = {
        "requested_scenarios": len(paths),
        "successful_scenarios": len(rows),
        "failed_scenarios": len(failures),
        "mean_metrics": means,
        "improvement": {
            "topk_coverage_gain": means["risk_topk_coverage"] - means["fixed_topk_coverage"],
            "adb_coverage_gain": means["risk_adb_coverage"] - means["fixed_adb_coverage"],
            "additional_average_beams": means["risk_average_k"] - means["fixed_average_k"],
            "overhead_reduction_change": (
                means["risk_overhead_reduction"] - means["fixed_overhead_reduction"]
            ),
        },
        "failures": failures,
    }
    summary_path = args.output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\n" + json.dumps(summary, indent=2))
    print(f"Saved: {csv_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
