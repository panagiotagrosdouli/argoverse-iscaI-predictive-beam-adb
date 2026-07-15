from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from iscai.failure_taxonomy import (
    TAXONOMY_ORDER,
    FailureTaxonomyThresholds,
    classify_failure,
    taxonomy_description,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exp07b: mutually exclusive metric-based failure taxonomy"
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("results/exp07/all_targets_with_failure_flags.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/exp07b"))
    parser.add_argument("--direction-error-deg", type=float, default=20.0)
    parser.add_argument("--small-direction-error-deg", type=float, default=5.0)
    parser.add_argument("--large-fde-m", type=float, default=8.0)
    parser.add_argument("--good-ade-m", type=float, default=2.0)
    parser.add_argument("--topk-target", type=float, default=0.95)
    parser.add_argument("--adb-target", type=float, default=0.95)
    parser.add_argument("--high-uncertainty-deg", type=float, default=10.0)
    return parser.parse_args()


def save_bar(data: pd.Series, ylabel: str, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(8.0, 4.8))
    data.plot.bar(ax=axis)
    axis.set_xlabel("Failure taxonomy")
    axis.set_ylabel(ylabel)
    axis.tick_params(axis="x", rotation=25)
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(output.with_suffix(".png"), dpi=200)
    figure.savefig(output.with_suffix(".svg"))
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")

    frame = pd.read_csv(args.input_csv)
    required = {
        "scenario_id",
        "track_id",
        "canonical_object_type",
        "cv_ade_m",
        "cv_fde_m",
        "cv_angular_error_deg",
        "topk_coverage",
        "adb_shadow_coverage",
        "mean_angular_std_deg",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    thresholds = FailureTaxonomyThresholds(
        direction_error_deg=args.direction_error_deg,
        small_direction_error_deg=args.small_direction_error_deg,
        large_fde_m=args.large_fde_m,
        good_ade_m=args.good_ade_m,
        topk_target=args.topk_target,
        adb_target=args.adb_target,
        high_uncertainty_deg=args.high_uncertainty_deg,
    )

    frame["failure_taxonomy"] = [
        classify_failure(row, thresholds) for row in frame.to_dict("records")
    ]
    frame["taxonomy_description"] = frame["failure_taxonomy"].map(taxonomy_description)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "all_targets_with_taxonomy.csv", index=False)

    counts = (
        frame["failure_taxonomy"]
        .value_counts()
        .reindex(TAXONOMY_ORDER, fill_value=0)
        .rename_axis("failure_taxonomy")
        .reset_index(name="count")
    )
    counts["rate"] = counts["count"] / len(frame)
    counts["description"] = counts["failure_taxonomy"].map(taxonomy_description)
    counts.to_csv(args.output_dir / "taxonomy_counts.csv", index=False)

    by_object = (
        frame.groupby(["canonical_object_type", "failure_taxonomy"])
        .size()
        .rename("count")
        .reset_index()
    )
    class_totals = frame.groupby("canonical_object_type").size().rename("class_total")
    by_object = by_object.join(class_totals, on="canonical_object_type")
    by_object["rate_within_object_type"] = by_object["count"] / by_object["class_total"]
    by_object.to_csv(args.output_dir / "taxonomy_by_object.csv", index=False)

    for label in TAXONOMY_ORDER:
        frame.loc[frame["failure_taxonomy"] == label].to_csv(
            args.output_dir / f"{label}.csv", index=False
        )

    plot_counts = counts.set_index("failure_taxonomy")["count"]
    save_bar(plot_counts, "Number of targets", args.output_dir / "taxonomy_counts")

    failures_only = counts.loc[counts["failure_taxonomy"] != "no_system_failure"].copy()
    if failures_only["count"].sum() > 0:
        figure, axis = plt.subplots(figsize=(7.0, 7.0))
        axis.pie(
            failures_only["count"],
            labels=failures_only["failure_taxonomy"],
            autopct="%1.1f%%",
        )
        axis.set_title("Failure taxonomy among classified failures")
        figure.tight_layout()
        figure.savefig(args.output_dir / "taxonomy_pie.png", dpi=200)
        figure.savefig(args.output_dir / "taxonomy_pie.svg")
        plt.close(figure)

    pivot = by_object.pivot(
        index="canonical_object_type",
        columns="failure_taxonomy",
        values="rate_within_object_type",
    ).fillna(0.0)
    pivot = pivot.reindex(columns=TAXONOMY_ORDER, fill_value=0.0)
    figure, axis = plt.subplots(figsize=(9.0, 5.0))
    pivot.plot.bar(ax=axis)
    axis.set_xlabel("Object type")
    axis.set_ylabel("Within-class fraction")
    axis.grid(axis="y", alpha=0.3)
    axis.legend(title="Taxonomy", fontsize=8)
    figure.tight_layout()
    figure.savefig(args.output_dir / "taxonomy_by_object.png", dpi=200)
    figure.savefig(args.output_dir / "taxonomy_by_object.svg")
    plt.close(figure)

    failure_mask = frame["failure_taxonomy"] != "no_system_failure"
    failure_count = int(failure_mask.sum())
    taxonomy_records = counts.to_dict("records")
    summary = {
        "experiment": "exp07b_failure_taxonomy",
        "input_csv": str(args.input_csv),
        "num_targets": int(len(frame)),
        "num_classified_failures": failure_count,
        "classified_failure_rate": float(failure_count / len(frame)),
        "thresholds": {
            "direction_error_deg": thresholds.direction_error_deg,
            "small_direction_error_deg": thresholds.small_direction_error_deg,
            "large_fde_m": thresholds.large_fde_m,
            "good_ade_m": thresholds.good_ade_m,
            "topk_target": thresholds.topk_target,
            "adb_target": thresholds.adb_target,
            "high_uncertainty_deg": thresholds.high_uncertainty_deg,
        },
        "taxonomy_counts": taxonomy_records,
        "note": (
            "These are mutually exclusive metric-based categories. Semantic root causes "
            "such as turns, crossings, occlusions, or lane changes require visual inspection."
        ),
    }
    with (args.output_dir / "taxonomy_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Saved experiment outputs under: {args.output_dir}")


if __name__ == "__main__":
    main()
