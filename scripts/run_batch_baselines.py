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
from iscai.evaluation import ade, fde, mean_angular_error_deg
from iscai.geometry import polar_from_xy, to_ego_coordinates
from iscai.prediction import constant_velocity, kalman_constant_velocity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch evaluation on AV2 scenarios")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-scenarios", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/batch"))
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

    cv = constant_velocity(observed, args.future_steps)
    kf = kalman_constant_velocity(observed, args.future_steps)

    half_fov = np.deg2rad(args.field_of_view_deg / 2.0)
    codebook = BeamCodebook(-half_fov, half_fov, args.num_beams)
    _, true_angles = polar_from_xy(future)
    _, predicted_angles = polar_from_xy(kf.mean)

    topk_hits, topk_sizes, shadow_hits, top1_hits = [], [], [], []
    for mean_xy, cov_xy, pred_angle, true_angle in zip(
        kf.mean, kf.covariance, predicted_angles, true_angles
    ):
        angular_std = np.sqrt(angular_variance_from_xy(mean_xy, cov_xy))
        probabilities = gaussian_beam_probabilities(pred_angle, angular_std, codebook)
        selected = adaptive_topk(probabilities, args.coverage)
        true_beam = int(codebook.angle_to_index(true_angle))
        predicted_beam = int(codebook.angle_to_index(pred_angle))
        top1_hits.append(predicted_beam == true_beam)
        topk_hits.append(true_beam in selected)
        topk_sizes.append(len(selected))
        interval = predictive_shadow_interval(mean_xy, cov_xy)
        shadow_hits.append(shadow_contains(interval, float(true_angle)))

    return {
        "scenario": str(path),
        "target_object_type": target.object_type,
        "cv_ade_m": ade(cv, future),
        "cv_fde_m": fde(cv, future),
        "cv_angular_error_deg": mean_angular_error_deg(cv, future),
        "kalman_ade_m": ade(kf.mean, future),
        "kalman_fde_m": fde(kf.mean, future),
        "kalman_angular_error_deg": mean_angular_error_deg(kf.mean, future),
        "top1_accuracy": float(np.mean(top1_hits)),
        "topk_coverage": float(np.mean(topk_hits)),
        "average_k": float(np.mean(topk_sizes)),
        "overhead_reduction": float(1.0 - np.mean(topk_sizes) / args.num_beams),
        "adb_shadow_coverage": float(np.mean(shadow_hits)),
    }


def main() -> None:
    args = parse_args()
    scenario_paths = sorted((args.dataset_root / args.split).glob("*/scenario_*.parquet"))
    scenario_paths = scenario_paths[: args.max_scenarios]
    if not scenario_paths:
        raise FileNotFoundError("No scenario parquet files found")

    rows, failures = [], []
    for index, path in enumerate(scenario_paths, start=1):
        try:
            rows.append(evaluate_one(path, args))
            print(f"[{index}/{len(scenario_paths)}] OK {path.parent.name}")
        except Exception as error:
            failures.append({"scenario": str(path), "error": str(error)})
            print(f"[{index}/{len(scenario_paths)}] SKIP {path.parent.name}: {error}")

    if not rows:
        raise RuntimeError("All scenarios failed")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "per_scenario_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    numeric_keys = [key for key in rows[0] if key not in {"scenario", "target_object_type"}]
    summary = {
        "requested_scenarios": len(scenario_paths),
        "successful_scenarios": len(rows),
        "failed_scenarios": len(failures),
        "mean_metrics": {
            key: float(np.mean([float(row[key]) for row in rows])) for key in numeric_keys
        },
        "failures": failures,
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\n" + json.dumps(summary, indent=2))
    print(f"Saved: {csv_path}")
    print(f"Saved: {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
