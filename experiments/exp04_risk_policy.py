from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.run_risk_aware_batch import evaluate_one


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 04: fixed versus risk-adaptive beam and ADB control"
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-scenarios", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=Path("results/exp04"))
    parser.add_argument("--observation-steps", type=int, default=50)
    parser.add_argument("--future-steps", type=int, default=60)
    parser.add_argument("--num-beams", type=int, default=16)
    parser.add_argument("--field-of-view-deg", type=float, default=120.0)
    parser.add_argument("--fixed-coverage", type=float, default=0.95)
    return parser.parse_args()


def mean(rows: list[dict], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def save_bar_plot(
    labels: list[str],
    values: list[float],
    ylabel: str,
    title: str,
    output_base: Path,
    percent: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    x = np.arange(len(labels))
    bars = ax.bar(x, values)
    ax.set_xticks(x, labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    if percent:
        ax.set_ylim(0.0, 1.05)
    for bar, value in zip(bars, values):
        label = f"{100.0 * value:.2f}%" if percent else f"{value:.3f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            label,
            ha="center",
            va="bottom",
        )
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".png"), dpi=200)
    fig.savefig(output_base.with_suffix(".svg"))
    plt.close(fig)


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
    per_scenario_path = args.output_dir / "per_scenario_metrics.csv"
    with per_scenario_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fixed_topk = mean(rows, "fixed_topk_coverage")
    risk_topk = mean(rows, "risk_topk_coverage")
    fixed_k = mean(rows, "fixed_average_k")
    risk_k = mean(rows, "risk_average_k")
    fixed_overhead = mean(rows, "fixed_overhead_reduction")
    risk_overhead = mean(rows, "risk_overhead_reduction")
    fixed_adb = mean(rows, "fixed_adb_coverage")
    risk_adb = mean(rows, "risk_adb_coverage")

    summary_rows = [
        {
            "policy": "fixed",
            "topk_coverage": fixed_topk,
            "average_k": fixed_k,
            "overhead_reduction": fixed_overhead,
            "adb_coverage": fixed_adb,
        },
        {
            "policy": "risk_adaptive",
            "topk_coverage": risk_topk,
            "average_k": risk_k,
            "overhead_reduction": risk_overhead,
            "adb_coverage": risk_adb,
        },
    ]
    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    summary = {
        "experiment": "exp04_risk_adaptive_policy",
        "split": args.split,
        "requested_scenarios": len(paths),
        "successful_scenarios": len(rows),
        "failed_scenarios": len(failures),
        "num_beams": args.num_beams,
        "fixed_coverage_threshold": args.fixed_coverage,
        "policies": summary_rows,
        "difference_risk_minus_fixed": {
            "topk_coverage": risk_topk - fixed_topk,
            "average_k": risk_k - fixed_k,
            "overhead_reduction": risk_overhead - fixed_overhead,
            "adb_coverage": risk_adb - fixed_adb,
        },
        "risk_statistics": {
            "mean_score": mean(rows, "mean_risk_score"),
            "mean_requested_coverage": mean(rows, "mean_requested_coverage"),
        },
        "failures": failures,
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    labels = ["Fixed", "Risk-adaptive"]
    save_bar_plot(
        labels,
        [fixed_topk, risk_topk],
        "Top-K coverage",
        "Beam coverage: fixed vs risk-adaptive",
        args.output_dir / "topk_coverage",
        percent=True,
    )
    save_bar_plot(
        labels,
        [fixed_k, risk_k],
        "Average selected beams",
        "Average beam probes",
        args.output_dir / "average_k",
    )
    save_bar_plot(
        labels,
        [fixed_overhead, risk_overhead],
        "Overhead reduction",
        "Beam-search overhead reduction",
        args.output_dir / "overhead_reduction",
        percent=True,
    )
    save_bar_plot(
        labels,
        [fixed_adb, risk_adb],
        "ADB angular coverage",
        "Predictive ADB coverage",
        args.output_dir / "adb_coverage",
        percent=True,
    )

    print("\n" + json.dumps(summary, indent=2))
    print(f"Saved experiment outputs under: {args.output_dir}")


if __name__ == "__main__":
    main()
