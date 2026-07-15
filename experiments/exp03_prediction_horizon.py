from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 3: evaluate prediction horizon effects on trajectory, beam, and ADB metrics"
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-scenarios", type=int, default=100)
    parser.add_argument("--observation-steps", type=int, default=50)
    parser.add_argument("--horizon-steps", type=int, nargs="+", default=[10, 20, 30, 40, 60])
    parser.add_argument("--sampling-hz", type=float, default=10.0)
    parser.add_argument("--num-beams", type=int, default=16)
    parser.add_argument("--field-of-view-deg", type=float, default=120.0)
    parser.add_argument("--coverage", type=float, default=0.95)
    parser.add_argument("--output-dir", type=Path, default=Path("results/exp03"))
    return parser.parse_args()


def run_batch(args: argparse.Namespace, future_steps: int, output_dir: Path) -> dict:
    command = [
        sys.executable,
        "scripts/run_batch_baselines.py",
        "--dataset-root",
        str(args.dataset_root),
        "--split",
        args.split,
        "--max-scenarios",
        str(args.max_scenarios),
        "--output-dir",
        str(output_dir),
        "--observation-steps",
        str(args.observation_steps),
        "--future-steps",
        str(future_steps),
        "--num-beams",
        str(args.num_beams),
        "--field-of-view-deg",
        str(args.field_of_view_deg),
        "--coverage",
        str(args.coverage),
    ]
    subprocess.run(command, check=True)
    with (output_dir / "summary.json").open("r", encoding="utf-8") as file:
        return json.load(file)


def save_plot(rows: list[dict], x_key: str, y_keys: list[tuple[str, str]], path: Path, ylabel: str) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    x_values = [row[x_key] for row in rows]
    for key, label in y_keys:
        axis.plot(x_values, [row[key] for row in rows], marker="o", label=label)
    axis.set_xlabel("Prediction horizon [s]")
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path.with_suffix(".png"), dpi=180)
    figure.savefig(path.with_suffix(".svg"))
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    failures: list[dict] = []

    for steps in args.horizon_steps:
        run_dir = args.output_dir / f"horizon_{steps}_steps"
        try:
            summary = run_batch(args, steps, run_dir)
            metrics = summary["mean_metrics"]
            row = {
                "future_steps": steps,
                "horizon_seconds": steps / args.sampling_hz,
                "successful_scenarios": summary["successful_scenarios"],
                "failed_scenarios": summary["failed_scenarios"],
                **metrics,
            }
            rows.append(row)
            print(
                f"H={row['horizon_seconds']:.1f}s | "
                f"CV ADE={row['cv_ade_m']:.3f} | CV FDE={row['cv_fde_m']:.3f} | "
                f"Top-1={row['top1_accuracy']:.4f} | Top-K={row['topk_coverage']:.4f} | "
                f"ADB={row['adb_shadow_coverage']:.4f}"
            )
        except Exception as error:
            failures.append({"future_steps": steps, "error": str(error)})
            print(f"FAILED horizon {steps} steps: {error}")

    if not rows:
        raise RuntimeError("All prediction-horizon runs failed")

    rows.sort(key=lambda item: item["future_steps"])

    csv_path = args.output_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    output = {
        "experiment": "exp03_prediction_horizon",
        "split": args.split,
        "sampling_hz": args.sampling_hz,
        "observation_steps": args.observation_steps,
        "num_beams": args.num_beams,
        "coverage_threshold": args.coverage,
        "results": rows,
        "failures": failures,
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)

    save_plot(
        rows,
        "horizon_seconds",
        [("cv_ade_m", "CV ADE"), ("kalman_ade_m", "Kalman ADE")],
        args.output_dir / "horizon_vs_ade",
        "ADE [m]",
    )
    save_plot(
        rows,
        "horizon_seconds",
        [("cv_fde_m", "CV FDE"), ("kalman_fde_m", "Kalman FDE")],
        args.output_dir / "horizon_vs_fde",
        "FDE [m]",
    )
    save_plot(
        rows,
        "horizon_seconds",
        [("top1_accuracy", "Top-1"), ("topk_coverage", "Adaptive Top-K"), ("adb_shadow_coverage", "ADB")],
        args.output_dir / "horizon_vs_control_coverage",
        "Coverage / accuracy",
    )
    save_plot(
        rows,
        "horizon_seconds",
        [("average_k", "Average K")],
        args.output_dir / "horizon_vs_average_k",
        "Average selected beams",
    )

    print(json.dumps(output, indent=2))
    print(f"Saved experiment outputs under: {args.output_dir}")


if __name__ == "__main__":
    main()
