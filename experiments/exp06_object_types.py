from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from iscai.adb import predictive_shadow_interval, shadow_contains
from iscai.beam import (
    BeamCodebook,
    adaptive_topk,
    angular_variance_from_xy,
    gaussian_beam_probabilities,
)
from iscai.data import extract_actor_tracks, find_ego_track, load_scenario_parquet
from iscai.evaluation import ade, fde, mean_angular_error_deg
from iscai.geometry import polar_from_xy, to_ego_coordinates
from iscai.prediction import constant_velocity, kalman_constant_velocity
from iscai.target_selector import OBJECT_TYPE_ALIASES, select_targets_by_object_type


DEFAULT_CLASSES = ("vehicle", "pedestrian", "cyclist", "motorcyclist")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exp06: class-wise AV2 trajectory, beam, and predictive ADB evaluation"
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-scenarios", type=int, default=100)
    parser.add_argument("--object-types", nargs="+", default=list(DEFAULT_CLASSES))
    parser.add_argument("--observation-steps", type=int, default=50)
    parser.add_argument("--future-steps", type=int, default=60)
    parser.add_argument("--num-beams", type=int, default=16)
    parser.add_argument("--field-of-view-deg", type=float, default=120.0)
    parser.add_argument("--coverage-threshold", type=float, default=0.95)
    parser.add_argument("--max-targets-per-class", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("results/exp06"))
    return parser.parse_args()


def state_at(track, timestep: int) -> tuple[np.ndarray, float]:
    matches = np.flatnonzero(track.timestep == timestep)
    index = int(matches[0]) if len(matches) else int(np.argmin(np.abs(track.timestep - timestep)))
    return track.position[index], float(track.heading[index])


def evaluate_target(
    scenario_path: Path,
    target,
    ego,
    args: argparse.Namespace,
    codebook: BeamCodebook,
) -> dict:
    required = args.observation_steps + args.future_steps
    target_position = target.position[:required]
    target_timestep = target.timestep[:required]
    current_timestep = int(target_timestep[args.observation_steps - 1])
    ego_origin, ego_heading = state_at(ego, current_timestep)

    target_ego = to_ego_coordinates(target_position, ego_origin, ego_heading)
    observed = target_ego[: args.observation_steps]
    future = target_ego[args.observation_steps : required]

    cv_prediction = constant_velocity(observed, args.future_steps)
    kalman_prediction = kalman_constant_velocity(observed, args.future_steps)

    _, true_angles = polar_from_xy(future)
    _, predicted_angles = polar_from_xy(kalman_prediction.mean)

    top1_hits: list[bool] = []
    topk_hits: list[bool] = []
    selected_sizes: list[int] = []
    adb_hits: list[bool] = []
    angular_stds: list[float] = []

    for mean_xy, covariance_xy, predicted_angle, true_angle in zip(
        kalman_prediction.mean,
        kalman_prediction.covariance,
        predicted_angles,
        true_angles,
    ):
        angular_std = float(np.sqrt(angular_variance_from_xy(mean_xy, covariance_xy)))
        probabilities = gaussian_beam_probabilities(predicted_angle, angular_std, codebook)
        true_beam = int(codebook.angle_to_index(true_angle))
        top1_beam = int(np.argmax(probabilities))
        selected = adaptive_topk(probabilities, args.coverage_threshold)
        interval = predictive_shadow_interval(mean_xy, covariance_xy, confidence_z=1.96)

        top1_hits.append(top1_beam == true_beam)
        topk_hits.append(bool(true_beam in selected))
        selected_sizes.append(len(selected))
        adb_hits.append(shadow_contains(interval, float(true_angle)))
        angular_stds.append(np.rad2deg(angular_std))

    return {
        "scenario_id": scenario_path.parent.name,
        "scenario": str(scenario_path),
        "track_id": target.track_id,
        "object_type": target.object_type,
        "cv_ade_m": ade(cv_prediction, future),
        "cv_fde_m": fde(cv_prediction, future),
        "cv_angular_error_deg": mean_angular_error_deg(cv_prediction, future),
        "kalman_ade_m": ade(kalman_prediction.mean, future),
        "kalman_fde_m": fde(kalman_prediction.mean, future),
        "kalman_angular_error_deg": mean_angular_error_deg(kalman_prediction.mean, future),
        "top1_accuracy": float(np.mean(top1_hits)),
        "topk_coverage": float(np.mean(topk_hits)),
        "average_k": float(np.mean(selected_sizes)),
        "overhead_reduction": float(1.0 - np.mean(selected_sizes) / args.num_beams),
        "adb_shadow_coverage": float(np.mean(adb_hits)),
        "mean_angular_std_deg": float(np.mean(angular_stds)),
    }


def summarize(rows: list[dict], object_types: list[str]) -> list[dict]:
    numeric_keys = [
        "cv_ade_m",
        "cv_fde_m",
        "cv_angular_error_deg",
        "kalman_ade_m",
        "kalman_fde_m",
        "kalman_angular_error_deg",
        "top1_accuracy",
        "topk_coverage",
        "average_k",
        "overhead_reduction",
        "adb_shadow_coverage",
        "mean_angular_std_deg",
    ]
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["canonical_object_type"]].append(row)

    summary_rows: list[dict] = []
    for object_type in object_types:
        class_rows = grouped.get(object_type, [])
        item: dict[str, object] = {
            "object_type": object_type,
            "num_targets": len(class_rows),
            "num_scenarios": len({row["scenario_id"] for row in class_rows}),
        }
        for key in numeric_keys:
            values = [float(row[key]) for row in class_rows]
            item[key] = float(np.mean(values)) if values else None
            item[f"{key}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        summary_rows.append(item)
    return summary_rows


def save_bar_plot(summary_rows: list[dict], metric: str, ylabel: str, output: Path) -> None:
    valid = [row for row in summary_rows if row[metric] is not None]
    if not valid:
        return
    labels = [str(row["object_type"]).title() for row in valid]
    values = [float(row[metric]) for row in valid]
    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    axis.bar(labels, values)
    axis.set_ylabel(ylabel)
    axis.set_xlabel("Object type")
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(output.with_suffix(".png"), dpi=200)
    figure.savefig(output.with_suffix(".svg"))
    plt.close(figure)


def main() -> None:
    args = parse_args()
    unknown = sorted(set(args.object_types).difference(OBJECT_TYPE_ALIASES))
    if unknown:
        raise ValueError(f"Unsupported object types: {unknown}")

    paths = sorted((args.dataset_root / args.split).glob("*/scenario_*.parquet"))
    paths = paths[: args.max_scenarios]
    if not paths:
        raise FileNotFoundError("No scenario parquet files found")

    half_fov = np.deg2rad(args.field_of_view_deg / 2.0)
    codebook = BeamCodebook(-half_fov, half_fov, args.num_beams)
    required = args.observation_steps + args.future_steps

    rows: list[dict] = []
    failures: list[dict] = []
    target_counts: Counter[str] = Counter()

    for scenario_index, path in enumerate(paths, start=1):
        try:
            frame = load_scenario_parquet(path)
            tracks = extract_actor_tracks(frame)
            ego = find_ego_track(tracks)
            scenario_evaluated = 0

            for object_type in args.object_types:
                targets = select_targets_by_object_type(tracks, object_type, required)
                for target in targets:
                    if (
                        args.max_targets_per_class > 0
                        and target_counts[object_type] >= args.max_targets_per_class
                    ):
                        continue
                    row = evaluate_target(path, target, ego, args, codebook)
                    row["canonical_object_type"] = object_type
                    rows.append(row)
                    target_counts[object_type] += 1
                    scenario_evaluated += 1

            print(
                f"[{scenario_index}/{len(paths)}] OK {path.parent.name} "
                f"({scenario_evaluated} eligible targets)"
            )
        except Exception as error:
            failures.append({"scenario": str(path), "error": str(error)})
            print(f"[{scenario_index}/{len(paths)}] SKIP {path.parent.name}: {error}")

    if not rows:
        raise RuntimeError("No eligible targets were evaluated")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_target_path = args.output_dir / "per_target_metrics.csv"
    with per_target_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = summarize(rows, args.object_types)
    summary_csv_path = args.output_dir / "summary.csv"
    with summary_csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    summary = {
        "experiment": "exp06_object_type_evaluation",
        "split": args.split,
        "requested_scenarios": len(paths),
        "scenarios_with_failures": len(failures),
        "num_beams": args.num_beams,
        "future_steps": args.future_steps,
        "horizon_seconds": args.future_steps / 10.0,
        "coverage_threshold": args.coverage_threshold,
        "total_evaluated_targets": len(rows),
        "class_results": summary_rows,
        "failures": failures,
    }
    summary_json_path = args.output_dir / "summary.json"
    with summary_json_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    save_bar_plot(summary_rows, "cv_ade_m", "CV ADE (m)", args.output_dir / "object_type_cv_ade")
    save_bar_plot(summary_rows, "cv_fde_m", "CV FDE (m)", args.output_dir / "object_type_cv_fde")
    save_bar_plot(summary_rows, "top1_accuracy", "Top-1 beam accuracy", args.output_dir / "object_type_top1")
    save_bar_plot(summary_rows, "topk_coverage", "Adaptive Top-K coverage", args.output_dir / "object_type_topk")
    save_bar_plot(summary_rows, "average_k", "Average selected beams", args.output_dir / "object_type_average_k")
    save_bar_plot(summary_rows, "adb_shadow_coverage", "Predictive ADB coverage", args.output_dir / "object_type_adb")

    print("\n" + json.dumps(summary, indent=2))
    print(f"Saved experiment outputs under: {args.output_dir}")


if __name__ == "__main__":
    main()
