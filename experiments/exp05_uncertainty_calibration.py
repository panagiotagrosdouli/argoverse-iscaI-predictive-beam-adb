from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from iscai.adb import predictive_shadow_interval, shadow_contains
from iscai.beam import (
    BeamCodebook,
    adaptive_topk,
    angular_variance_from_xy,
    gaussian_beam_probabilities,
)
from iscai.calibration import gaussian_nll_2d, mahalanobis_squared
from iscai.data import extract_actor_tracks, find_ego_track, load_scenario_parquet
from iscai.geometry import polar_from_xy, to_ego_coordinates
from iscai.prediction import kalman_constant_velocity


NOMINAL_LEVELS = (0.50, 0.68, 0.80, 0.90, 0.95, 0.99)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate calibration of Kalman trajectory uncertainty and downstream control"
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-scenarios", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=Path("results/exp05"))
    parser.add_argument("--observation-steps", type=int, default=50)
    parser.add_argument("--future-steps", type=int, default=60)
    parser.add_argument("--num-beams", type=int, default=16)
    parser.add_argument("--field-of-view-deg", type=float, default=120.0)
    parser.add_argument("--beam-coverage", type=float, default=0.95)
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


def entropy(probabilities: np.ndarray) -> float:
    probabilities = np.asarray(probabilities, dtype=float)
    positive = probabilities[probabilities > 0]
    return float(-np.sum(positive * np.log(positive)))


def prepare_one(path: Path, args: argparse.Namespace) -> dict:
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

    distances = mahalanobis_squared(future, prediction.mean, prediction.covariance)
    nll = gaussian_nll_2d(future, prediction.mean, prediction.covariance)

    point_rows: list[dict] = []
    for step, (mean_xy, cov_xy, pred_angle, true_angle, distance, step_nll) in enumerate(
        zip(
            prediction.mean,
            prediction.covariance,
            predicted_angles,
            true_angles,
            distances,
            nll,
        ),
        start=1,
    ):
        angular_std = float(np.sqrt(angular_variance_from_xy(mean_xy, cov_xy)))
        probabilities = gaussian_beam_probabilities(pred_angle, angular_std, codebook)
        selected = adaptive_topk(probabilities, args.beam_coverage)
        true_beam = int(codebook.angle_to_index(true_angle))

        row = {
            "scenario": str(path),
            "target_object_type": target.object_type,
            "future_step": step,
            "mahalanobis_sq": float(distance),
            "nll": float(step_nll),
            "angular_std_deg": float(np.rad2deg(angular_std)),
            "beam_entropy": entropy(probabilities),
            "selected_k": int(len(selected)),
            "beam_hit": int(true_beam in selected),
        }
        for level in NOMINAL_LEVELS:
            z = float(norm.ppf(0.5 + level / 2.0))
            interval = predictive_shadow_interval(mean_xy, cov_xy, confidence_z=z)
            row[f"adb_hit_{int(level * 100)}"] = int(
                shadow_contains(interval, float(true_angle))
            )
            row[f"adb_width_deg_{int(level * 100)}"] = float(
                np.rad2deg(interval.width_rad)
            )
        point_rows.append(row)

    return {"rows": point_rows, "object_type": target.object_type}


def calibration_rows(point_rows: list[dict]) -> list[dict]:
    distances = np.asarray([row["mahalanobis_sq"] for row in point_rows], dtype=float)
    from scipy.stats import chi2

    rows: list[dict] = []
    for level in NOMINAL_LEVELS:
        threshold = float(chi2.ppf(level, df=2))
        observed = float(np.mean(distances <= threshold))
        rows.append(
            {
                "nominal_coverage": level,
                "observed_position_coverage": observed,
                "position_calibration_error": observed - level,
                "observed_adb_coverage": float(
                    np.mean([row[f"adb_hit_{int(level * 100)}"] for row in point_rows])
                ),
                "mean_adb_width_deg": float(
                    np.mean([row[f"adb_width_deg_{int(level * 100)}"] for row in point_rows])
                ),
            }
        )
    return rows


def save_plots(calibration: list[dict], point_rows: list[dict], output_dir: Path) -> None:
    nominal = np.asarray([row["nominal_coverage"] for row in calibration])
    observed = np.asarray([row["observed_position_coverage"] for row in calibration])
    adb = np.asarray([row["observed_adb_coverage"] for row in calibration])

    plt.figure(figsize=(6.5, 5.0))
    plt.plot(nominal, nominal, linestyle="--", label="Ideal")
    plt.plot(nominal, observed, marker="o", label="Position ellipse")
    plt.plot(nominal, adb, marker="s", label="ADB interval")
    plt.xlabel("Nominal confidence")
    plt.ylabel("Observed coverage")
    plt.title("Uncertainty reliability diagram")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "reliability.svg")
    plt.savefig(output_dir / "reliability.png", dpi=200)
    plt.close()

    uncertainty = np.asarray([row["angular_std_deg"] for row in point_rows])
    selected_k = np.asarray([row["selected_k"] for row in point_rows])
    entropy_values = np.asarray([row["beam_entropy"] for row in point_rows])

    plt.figure(figsize=(6.5, 5.0))
    plt.scatter(uncertainty, selected_k, s=8, alpha=0.25)
    plt.xlabel("Predicted angular standard deviation (deg)")
    plt.ylabel("Selected beams K")
    plt.title("Uncertainty vs adaptive beam count")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "uncertainty_vs_topk.svg")
    plt.savefig(output_dir / "uncertainty_vs_topk.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6.5, 5.0))
    plt.scatter(uncertainty, entropy_values, s=8, alpha=0.25)
    plt.xlabel("Predicted angular standard deviation (deg)")
    plt.ylabel("Beam probability entropy (nats)")
    plt.title("Uncertainty vs beam entropy")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "uncertainty_vs_entropy.svg")
    plt.savefig(output_dir / "uncertainty_vs_entropy.png", dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    paths = sorted((args.dataset_root / args.split).glob("*/scenario_*.parquet"))
    paths = paths[: args.max_scenarios]
    if not paths:
        raise FileNotFoundError("No scenario parquet files found")

    point_rows: list[dict] = []
    failures: list[dict] = []
    for index, path in enumerate(paths, start=1):
        try:
            result = prepare_one(path, args)
            point_rows.extend(result["rows"])
            print(f"[{index}/{len(paths)}] OK {path.parent.name}")
        except Exception as error:
            failures.append({"scenario": str(path), "error": str(error)})
            print(f"[{index}/{len(paths)}] SKIP {path.parent.name}: {error}")

    if not point_rows:
        raise RuntimeError("All scenarios failed")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    calibration = calibration_rows(point_rows)

    with (args.output_dir / "point_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(point_rows[0]))
        writer.writeheader()
        writer.writerows(point_rows)

    with (args.output_dir / "calibration.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(calibration[0]))
        writer.writeheader()
        writer.writerows(calibration)

    position_errors = np.asarray(
        [abs(row["position_calibration_error"]) for row in calibration], dtype=float
    )
    uncertainty = np.asarray([row["angular_std_deg"] for row in point_rows], dtype=float)
    selected_k = np.asarray([row["selected_k"] for row in point_rows], dtype=float)
    entropy_values = np.asarray([row["beam_entropy"] for row in point_rows], dtype=float)

    summary = {
        "experiment": "exp05_uncertainty_calibration",
        "split": args.split,
        "requested_scenarios": len(paths),
        "successful_scenarios": len(paths) - len(failures),
        "failed_scenarios": len(failures),
        "num_prediction_points": len(point_rows),
        "position_expected_calibration_error": float(position_errors.mean()),
        "position_maximum_calibration_error": float(position_errors.max()),
        "mean_gaussian_nll": float(np.mean([row["nll"] for row in point_rows])),
        "mean_mahalanobis_sq": float(
            np.mean([row["mahalanobis_sq"] for row in point_rows])
        ),
        "mean_angular_std_deg": float(uncertainty.mean()),
        "mean_beam_entropy": float(entropy_values.mean()),
        "mean_selected_k": float(selected_k.mean()),
        "beam_coverage": float(np.mean([row["beam_hit"] for row in point_rows])),
        "uncertainty_k_correlation": float(np.corrcoef(uncertainty, selected_k)[0, 1]),
        "uncertainty_entropy_correlation": float(
            np.corrcoef(uncertainty, entropy_values)[0, 1]
        ),
        "coverage_curve": calibration,
        "failures": failures,
    }

    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    save_plots(calibration, point_rows, args.output_dir)
    print("\n" + json.dumps(summary, indent=2))
    print(f"Saved experiment outputs under: {args.output_dir}")


if __name__ == "__main__":
    main()
