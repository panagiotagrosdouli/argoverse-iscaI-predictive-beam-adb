from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
from iscai.geometry import polar_from_xy, to_ego_coordinates
from iscai.prediction import kalman_constant_velocity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 02: evaluate the beam-codebook size trade-off on AV2 scenarios"
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-scenarios", type=int, default=100)
    parser.add_argument("--beam-sizes", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--field-of-view-deg", type=float, default=120.0)
    parser.add_argument("--coverage", type=float, default=0.95)
    parser.add_argument("--observation-steps", type=int, default=50)
    parser.add_argument("--future-steps", type=int, default=60)
    parser.add_argument("--output-dir", type=Path, default=Path("results/exp02"))
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
        raise ValueError("No target actor has enough samples")
    return max(candidates, key=lambda track: len(track.position))


def state_at(track, timestep: int) -> tuple[np.ndarray, float]:
    matches = np.flatnonzero(track.timestep == timestep)
    index = int(matches[0]) if len(matches) else int(np.argmin(np.abs(track.timestep - timestep)))
    return track.position[index], float(track.heading[index])


def prepare_scenario(path: Path, args: argparse.Namespace) -> dict:
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
    kalman = kalman_constant_velocity(observed, args.future_steps)
    _, true_angles = polar_from_xy(future)
    _, predicted_angles = polar_from_xy(kalman.mean)

    angular_stds = np.array(
        [
            np.sqrt(angular_variance_from_xy(mean_xy, covariance_xy))
            for mean_xy, covariance_xy in zip(kalman.mean, kalman.covariance)
        ],
        dtype=float,
    )

    adb_hits = [
        shadow_contains(
            predictive_shadow_interval(mean_xy, covariance_xy),
            float(true_angle),
        )
        for mean_xy, covariance_xy, true_angle in zip(
            kalman.mean, kalman.covariance, true_angles
        )
    ]

    return {
        "scenario": str(path),
        "target_object_type": target.object_type,
        "true_angles": true_angles,
        "predicted_angles": predicted_angles,
        "angular_stds": angular_stds,
        "adb_shadow_coverage": float(np.mean(adb_hits)),
    }


def evaluate_codebook(prepared: list[dict], num_beams: int, args: argparse.Namespace) -> dict:
    half_fov = np.deg2rad(args.field_of_view_deg / 2.0)
    codebook = BeamCodebook(-half_fov, half_fov, num_beams)

    scenario_rows: list[dict] = []
    all_top1: list[bool] = []
    all_topk: list[bool] = []
    all_k: list[int] = []

    for item in prepared:
        top1_hits: list[bool] = []
        topk_hits: list[bool] = []
        topk_sizes: list[int] = []

        for pred_angle, true_angle, angular_std in zip(
            item["predicted_angles"], item["true_angles"], item["angular_stds"]
        ):
            probabilities = gaussian_beam_probabilities(pred_angle, angular_std, codebook)
            selected = adaptive_topk(probabilities, args.coverage)
            predicted_beam = int(codebook.angle_to_index(pred_angle))
            true_beam = int(codebook.angle_to_index(true_angle))

            top1_hits.append(predicted_beam == true_beam)
            topk_hits.append(true_beam in selected)
            topk_sizes.append(len(selected))

        all_top1.extend(top1_hits)
        all_topk.extend(topk_hits)
        all_k.extend(topk_sizes)

        scenario_rows.append(
            {
                "scenario": item["scenario"],
                "target_object_type": item["target_object_type"],
                "num_beams": num_beams,
                "top1_accuracy": float(np.mean(top1_hits)),
                "topk_coverage": float(np.mean(topk_hits)),
                "average_k": float(np.mean(topk_sizes)),
                "overhead_reduction": float(1.0 - np.mean(topk_sizes) / num_beams),
                "adb_shadow_coverage": item["adb_shadow_coverage"],
            }
        )

    return {
        "summary": {
            "num_beams": num_beams,
            "top1_accuracy": float(np.mean(all_top1)),
            "topk_coverage": float(np.mean(all_topk)),
            "average_k": float(np.mean(all_k)),
            "overhead_reduction": float(1.0 - np.mean(all_k) / num_beams),
            "adb_shadow_coverage": float(
                np.mean([item["adb_shadow_coverage"] for item in prepared])
            ),
        },
        "scenario_rows": scenario_rows,
    }


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_tradeoff(summary_rows: list[dict], output_dir: Path) -> None:
    beam_sizes = np.array([row["num_beams"] for row in summary_rows], dtype=int)

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(beam_sizes, [row["top1_accuracy"] for row in summary_rows], marker="o", label="Top-1 accuracy")
    axis.plot(beam_sizes, [row["topk_coverage"] for row in summary_rows], marker="o", label="Adaptive Top-K coverage")
    axis.set_xlabel("Beam codebook size")
    axis.set_ylabel("Coverage / accuracy")
    axis.set_ylim(0.0, 1.02)
    axis.set_xticks(beam_sizes)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "beam_codebook_coverage.svg")
    figure.savefig(output_dir / "beam_codebook_coverage.png", dpi=200)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(beam_sizes, [row["average_k"] for row in summary_rows], marker="o", label="Average selected beams")
    axis.set_xlabel("Beam codebook size")
    axis.set_ylabel("Average K")
    axis.set_xticks(beam_sizes)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "beam_codebook_average_k.svg")
    figure.savefig(output_dir / "beam_codebook_average_k.png", dpi=200)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(beam_sizes, [row["overhead_reduction"] for row in summary_rows], marker="o")
    axis.set_xlabel("Beam codebook size")
    axis.set_ylabel("Overhead reduction")
    axis.set_ylim(0.0, 1.02)
    axis.set_xticks(beam_sizes)
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_dir / "beam_codebook_overhead.svg")
    figure.savefig(output_dir / "beam_codebook_overhead.png", dpi=200)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    scenario_paths = sorted((args.dataset_root / args.split).glob("*/scenario_*.parquet"))
    scenario_paths = scenario_paths[: args.max_scenarios]
    if not scenario_paths:
        raise FileNotFoundError("No AV2 scenario parquet files found")

    prepared: list[dict] = []
    failures: list[dict] = []
    for index, path in enumerate(scenario_paths, start=1):
        try:
            prepared.append(prepare_scenario(path, args))
            print(f"[{index}/{len(scenario_paths)}] PREPARED {path.parent.name}")
        except Exception as error:
            failures.append({"scenario": str(path), "error": str(error)})
            print(f"[{index}/{len(scenario_paths)}] SKIP {path.parent.name}: {error}")

    if not prepared:
        raise RuntimeError("All scenarios failed during preprocessing")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict] = []

    for num_beams in sorted(set(args.beam_sizes)):
        result = evaluate_codebook(prepared, num_beams, args)
        summary_rows.append(result["summary"])
        save_csv(
            args.output_dir / f"per_scenario_beams_{num_beams}.csv",
            result["scenario_rows"],
        )
        print(
            f"N={num_beams:>2} | Top-1={result['summary']['top1_accuracy']:.4f} "
            f"| Top-K={result['summary']['topk_coverage']:.4f} "
            f"| Avg K={result['summary']['average_k']:.3f} "
            f"| Overhead reduction={result['summary']['overhead_reduction']:.4f}"
        )

    save_csv(args.output_dir / "summary.csv", summary_rows)
    summary = {
        "experiment": "exp02_beam_codebook_size",
        "split": args.split,
        "requested_scenarios": len(scenario_paths),
        "successful_scenarios": len(prepared),
        "failed_scenarios": len(failures),
        "coverage_threshold": args.coverage,
        "field_of_view_deg": args.field_of_view_deg,
        "results": summary_rows,
        "failures": failures,
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    plot_tradeoff(summary_rows, args.output_dir)

    print("\n" + json.dumps(summary, indent=2))
    print(f"Saved experiment outputs under: {args.output_dir}")


if __name__ == "__main__":
    main()
