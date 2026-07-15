from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "scenario_id",
    "track_id",
    "canonical_object_type",
    "cv_ade_m",
    "cv_fde_m",
    "cv_angular_error_deg",
    "kalman_ade_m",
    "kalman_fde_m",
    "kalman_angular_error_deg",
    "top1_accuracy",
    "topk_coverage",
    "average_k",
    "adb_shadow_coverage",
    "mean_angular_std_deg",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exp07: failure taxonomy and worst-case analysis from Exp06 outputs"
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("results/exp06_1000/per_target_metrics.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/exp07"))
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--top1-threshold", type=float, default=0.95)
    parser.add_argument("--topk-threshold", type=float, default=0.95)
    parser.add_argument("--adb-threshold", type=float, default=0.95)
    parser.add_argument("--high-k-threshold", type=float, default=6.0)
    parser.add_argument("--high-uncertainty-deg", type=float, default=10.0)
    parser.add_argument("--high-angular-error-deg", type=float, default=5.0)
    parser.add_argument("--large-fde-m", type=float, default=8.0)
    return parser.parse_args()


def validate(frame: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Input CSV is missing columns: {sorted(missing)}")


def add_failure_flags(frame: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    data = frame.copy()
    data["top1_failure"] = data["top1_accuracy"] < args.top1_threshold
    data["topk_failure"] = data["topk_coverage"] < args.topk_threshold
    data["adb_failure"] = data["adb_shadow_coverage"] < args.adb_threshold
    data["high_beam_budget"] = data["average_k"] > args.high_k_threshold
    data["high_uncertainty"] = data["mean_angular_std_deg"] > args.high_uncertainty_deg
    data["high_angular_error"] = (
        data[["cv_angular_error_deg", "kalman_angular_error_deg"]].min(axis=1)
        > args.high_angular_error_deg
    )
    data["large_final_error"] = (
        data[["cv_fde_m", "kalman_fde_m"]].min(axis=1) > args.large_fde_m
    )
    data["vru"] = data["canonical_object_type"].isin(
        ["pedestrian", "cyclist", "motorcyclist"]
    )
    data["kalman_worse_than_cv"] = data["kalman_ade_m"] > data["cv_ade_m"]
    data["cv_kalman_ade_gap_m"] = (data["kalman_ade_m"] - data["cv_ade_m"]).abs()

    flag_columns = [
        "top1_failure",
        "topk_failure",
        "adb_failure",
        "high_beam_budget",
        "high_uncertainty",
        "high_angular_error",
        "large_final_error",
    ]
    data["failure_count"] = data[flag_columns].sum(axis=1)
    data["any_system_failure"] = data[["topk_failure", "adb_failure"]].any(axis=1)

    labels: list[str] = []
    for _, row in data.iterrows():
        causes: list[str] = []
        if row["high_angular_error"]:
            causes.append("high angular error")
        if row["large_final_error"]:
            causes.append("large final displacement error")
        if row["high_uncertainty"]:
            causes.append("high predicted angular uncertainty")
        if row["high_beam_budget"]:
            causes.append("large beam budget")
        if row["vru"]:
            causes.append("vulnerable road user")
        if row["kalman_worse_than_cv"]:
            causes.append("Kalman worse than CV")
        labels.append("; ".join(causes) if causes else "no proxy cause triggered")
    data["proxy_causes"] = labels
    return data


def save_ranked(data: pd.DataFrame, column: str, path: Path, top_n: int) -> None:
    data.nlargest(top_n, column).to_csv(path, index=False)


def save_bar(series: pd.Series, ylabel: str, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8.0, 4.8))
    series.plot(kind="bar", ax=axis)
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", alpha=0.3)
    axis.tick_params(axis="x", rotation=25)
    figure.tight_layout()
    figure.savefig(path.with_suffix(".png"), dpi=200)
    figure.savefig(path.with_suffix(".svg"))
    plt.close(figure)


def safe_rate(mask: pd.Series) -> float:
    return float(mask.mean()) if len(mask) else float("nan")


def main() -> None:
    args = parse_args()
    if args.top_n < 1:
        raise ValueError("--top-n must be positive")
    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")

    frame = pd.read_csv(args.input_csv)
    validate(frame)
    data = add_failure_flags(frame, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data.to_csv(args.output_dir / "all_targets_with_failure_flags.csv", index=False)
    save_ranked(data, "cv_ade_m", args.output_dir / "worst_cv_ade.csv", args.top_n)
    save_ranked(data, "cv_fde_m", args.output_dir / "worst_cv_fde.csv", args.top_n)
    save_ranked(
        data,
        "cv_angular_error_deg",
        args.output_dir / "worst_cv_angular.csv",
        args.top_n,
    )
    save_ranked(data, "kalman_ade_m", args.output_dir / "worst_kalman_ade.csv", args.top_n)
    save_ranked(
        data,
        "cv_kalman_ade_gap_m",
        args.output_dir / "largest_predictor_disagreement.csv",
        args.top_n,
    )

    data[data["top1_failure"]].sort_values("top1_accuracy").to_csv(
        args.output_dir / "top1_failures.csv", index=False
    )
    data[data["topk_failure"]].sort_values("topk_coverage").to_csv(
        args.output_dir / "topk_failures.csv", index=False
    )
    data[data["adb_failure"]].sort_values("adb_shadow_coverage").to_csv(
        args.output_dir / "adb_failures.csv", index=False
    )
    data[data["high_beam_budget"]].sort_values("average_k", ascending=False).to_csv(
        args.output_dir / "high_beam_budget.csv", index=False
    )
    data[data["any_system_failure"]].sort_values(
        ["failure_count", "cv_angular_error_deg"], ascending=False
    ).to_csv(args.output_dir / "system_failures.csv", index=False)

    flag_columns = [
        "top1_failure",
        "topk_failure",
        "adb_failure",
        "high_beam_budget",
        "high_uncertainty",
        "high_angular_error",
        "large_final_error",
    ]
    failure_statistics = pd.DataFrame(
        {
            "failure_type": flag_columns,
            "count": [int(data[column].sum()) for column in flag_columns],
            "rate": [safe_rate(data[column]) for column in flag_columns],
        }
    )
    failure_statistics.to_csv(args.output_dir / "failure_statistics.csv", index=False)

    class_rows: list[dict] = []
    for object_type, group in data.groupby("canonical_object_type", sort=True):
        class_rows.append(
            {
                "object_type": object_type,
                "num_targets": int(len(group)),
                "top1_failure_rate": safe_rate(group["top1_failure"]),
                "topk_failure_rate": safe_rate(group["topk_failure"]),
                "adb_failure_rate": safe_rate(group["adb_failure"]),
                "high_uncertainty_rate": safe_rate(group["high_uncertainty"]),
                "mean_failure_count": float(group["failure_count"].mean()),
            }
        )
    class_summary = pd.DataFrame(class_rows)
    class_summary.to_csv(args.output_dir / "failure_by_object_type.csv", index=False)

    cause_counts = {
        "high angular error": int(data["high_angular_error"].sum()),
        "large final displacement error": int(data["large_final_error"].sum()),
        "high predicted angular uncertainty": int(data["high_uncertainty"].sum()),
        "large beam budget": int(data["high_beam_budget"].sum()),
        "vulnerable road user": int(data["vru"].sum()),
        "Kalman worse than CV": int(data["kalman_worse_than_cv"].sum()),
    }
    pd.DataFrame(
        [{"proxy_cause": key, "count": value} for key, value in cause_counts.items()]
    ).to_csv(args.output_dir / "proxy_cause_statistics.csv", index=False)

    save_bar(
        failure_statistics.set_index("failure_type")["rate"],
        "Fraction of evaluated targets",
        args.output_dir / "failure_rates",
    )
    save_bar(
        class_summary.set_index("object_type")["topk_failure_rate"],
        "Top-K failure rate",
        args.output_dir / "topk_failure_by_object_type",
    )
    save_bar(
        class_summary.set_index("object_type")["adb_failure_rate"],
        "ADB failure rate",
        args.output_dir / "adb_failure_by_object_type",
    )
    save_bar(
        pd.Series(cause_counts).sort_values(ascending=False),
        "Number of targets",
        args.output_dir / "proxy_cause_counts",
    )

    system_failures = data[data["any_system_failure"]]
    summary = {
        "experiment": "exp07_failure_analysis",
        "input_csv": str(args.input_csv),
        "num_targets": int(len(data)),
        "num_scenarios": int(data["scenario_id"].nunique()),
        "num_system_failures": int(len(system_failures)),
        "system_failure_rate": safe_rate(data["any_system_failure"]),
        "top1_failure_rate": safe_rate(data["top1_failure"]),
        "topk_failure_rate": safe_rate(data["topk_failure"]),
        "adb_failure_rate": safe_rate(data["adb_failure"]),
        "high_beam_budget_rate": safe_rate(data["high_beam_budget"]),
        "high_uncertainty_rate": safe_rate(data["high_uncertainty"]),
        "worst_cv_ade_m": float(data["cv_ade_m"].max()),
        "worst_cv_fde_m": float(data["cv_fde_m"].max()),
        "worst_cv_angular_error_deg": float(data["cv_angular_error_deg"].max()),
        "worst_kalman_ade_m": float(data["kalman_ade_m"].max()),
        "thresholds": {
            "top1_accuracy": args.top1_threshold,
            "topk_coverage": args.topk_threshold,
            "adb_shadow_coverage": args.adb_threshold,
            "high_k": args.high_k_threshold,
            "high_uncertainty_deg": args.high_uncertainty_deg,
            "high_angular_error_deg": args.high_angular_error_deg,
            "large_fde_m": args.large_fde_m,
        },
        "failure_by_object_type": class_rows,
        "proxy_cause_counts": cause_counts,
        "note": (
            "Proxy causes are metric-based screening labels, not semantic scene labels. "
            "Scenario visualization is required to confirm causes such as turns, crossings, "
            "occlusion, and lane changes."
        ),
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Saved experiment outputs under: {args.output_dir}")


if __name__ == "__main__":
    main()
