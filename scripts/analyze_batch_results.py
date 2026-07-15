from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze batch trajectory, beam, and ADB metrics")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def paired_bootstrap_ci(values: np.ndarray, n_boot: int = 10000, seed: int = 7) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        means[i] = np.mean(sample)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main() -> None:
    args = parse_args()
    if not args.csv.exists():
        raise FileNotFoundError(args.csv)

    df = pd.read_csv(args.csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    required = {
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
    }
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(f"Missing columns: {sorted(missing)}")

    df["ade_diff_kalman_minus_cv"] = df["kalman_ade_m"] - df["cv_ade_m"]
    df["fde_diff_kalman_minus_cv"] = df["kalman_fde_m"] - df["cv_fde_m"]
    df["angular_diff_kalman_minus_cv"] = (
        df["kalman_angular_error_deg"] - df["cv_angular_error_deg"]
    )

    summary = {
        "num_scenarios": int(len(df)),
        "cv_better_ade_fraction": float((df["cv_ade_m"] < df["kalman_ade_m"]).mean()),
        "cv_better_fde_fraction": float((df["cv_fde_m"] < df["kalman_fde_m"]).mean()),
        "cv_better_angular_fraction": float(
            (df["cv_angular_error_deg"] < df["kalman_angular_error_deg"]).mean()
        ),
        "mean_ade_difference_kalman_minus_cv": float(df["ade_diff_kalman_minus_cv"].mean()),
        "mean_fde_difference_kalman_minus_cv": float(df["fde_diff_kalman_minus_cv"].mean()),
        "mean_angular_difference_kalman_minus_cv": float(
            df["angular_diff_kalman_minus_cv"].mean()
        ),
        "ade_difference_95pct_bootstrap_ci": paired_bootstrap_ci(
            df["ade_diff_kalman_minus_cv"].to_numpy()
        ),
        "fde_difference_95pct_bootstrap_ci": paired_bootstrap_ci(
            df["fde_diff_kalman_minus_cv"].to_numpy()
        ),
        "angular_difference_95pct_bootstrap_ci": paired_bootstrap_ci(
            df["angular_diff_kalman_minus_cv"].to_numpy()
        ),
        "top1_failure_scenarios": int((df["top1_accuracy"] < 1.0).sum()),
        "topk_failure_scenarios": int((df["topk_coverage"] < 1.0).sum()),
        "adb_failure_scenarios": int((df["adb_shadow_coverage"] < 1.0).sum()),
        "top1_accuracy_mean": float(df["top1_accuracy"].mean()),
        "topk_coverage_mean": float(df["topk_coverage"].mean()),
        "average_k_mean": float(df["average_k"].mean()),
        "overhead_reduction_mean": float(df["overhead_reduction"].mean()),
        "adb_shadow_coverage_mean": float(df["adb_shadow_coverage"].mean()),
    }

    with (args.output_dir / "statistical_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    worst_columns = [
        c
        for c in [
            "scenario_id",
            "scenario",
            "target_track_id",
            "target_object_type",
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
        ]
        if c in df.columns
    ]

    df.sort_values("cv_fde_m", ascending=False).head(10)[worst_columns].to_csv(
        args.output_dir / "worst_cv_fde.csv", index=False
    )
    df.sort_values("kalman_fde_m", ascending=False).head(10)[worst_columns].to_csv(
        args.output_dir / "worst_kalman_fde.csv", index=False
    )
    df.sort_values("top1_accuracy", ascending=True).head(10)[worst_columns].to_csv(
        args.output_dir / "worst_top1.csv", index=False
    )
    df.sort_values("topk_coverage", ascending=True).head(10)[worst_columns].to_csv(
        args.output_dir / "worst_topk.csv", index=False
    )
    df.sort_values("adb_shadow_coverage", ascending=True).head(10)[worst_columns].to_csv(
        args.output_dir / "worst_adb.csv", index=False
    )

    print(json.dumps(summary, indent=2))
    print(f"Saved analysis under: {args.output_dir}")


if __name__ == "__main__":
    main()
