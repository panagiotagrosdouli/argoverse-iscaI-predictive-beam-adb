from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


EXPERIMENT_SUMMARIES = {
    "exp05_uncertainty": Path("results/exp05/summary.json"),
    "exp06_object_types": Path("results/exp06_1000/summary.json"),
    "exp07_failures": Path("results/exp07/summary.json"),
    "exp07b_taxonomy": Path("results/exp07b/taxonomy_summary.json"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build consolidated publication-ready tables and figures."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Repository root containing the results directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("publication"),
        help="Directory for consolidated publication outputs.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def flatten(prefix: str, value: Any, output: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            flatten(next_prefix, item, output)
    elif isinstance(value, list):
        output[prefix] = json.dumps(value, ensure_ascii=False)
    else:
        output[prefix] = value


def save_figure(figure: plt.Figure, path_without_suffix: Path) -> None:
    figure.tight_layout()
    figure.savefig(path_without_suffix.with_suffix(".png"), dpi=240)
    figure.savefig(path_without_suffix.with_suffix(".svg"))
    plt.close(figure)


def build_object_type_table(summary: dict[str, Any], tables_dir: Path) -> pd.DataFrame:
    frame = pd.DataFrame(summary.get("class_results", []))
    if frame.empty:
        return frame

    selected = [
        "object_type",
        "num_targets",
        "num_scenarios",
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
    frame = frame[[column for column in selected if column in frame.columns]]
    frame.to_csv(tables_dir / "table_object_types.csv", index=False)
    return frame


def build_failure_table(summary: dict[str, Any], tables_dir: Path) -> pd.DataFrame:
    rows = [
        {"metric": "System failure rate", "value": summary.get("system_failure_rate")},
        {"metric": "Top-1 failure rate", "value": summary.get("top1_failure_rate")},
        {"metric": "Top-K failure rate", "value": summary.get("topk_failure_rate")},
        {"metric": "ADB failure rate", "value": summary.get("adb_failure_rate")},
        {"metric": "High beam-budget rate", "value": summary.get("high_beam_budget_rate")},
        {"metric": "High uncertainty rate", "value": summary.get("high_uncertainty_rate")},
        {"metric": "Worst CV ADE (m)", "value": summary.get("worst_cv_ade_m")},
        {"metric": "Worst CV FDE (m)", "value": summary.get("worst_cv_fde_m")},
        {
            "metric": "Worst CV angular error (deg)",
            "value": summary.get("worst_cv_angular_error_deg"),
        },
    ]
    frame = pd.DataFrame(rows).dropna(subset=["value"])
    frame.to_csv(tables_dir / "table_failure_analysis.csv", index=False)
    return frame


def build_taxonomy_tables(
    taxonomy_summary: dict[str, Any],
    taxonomy_targets_path: Path,
    tables_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_counts = pd.DataFrame(taxonomy_summary.get("taxonomy_counts", []))
    if not all_counts.empty:
        all_counts.to_csv(tables_dir / "table_taxonomy_all_targets.csv", index=False)

    system_counts = pd.DataFrame()
    if taxonomy_targets_path.exists():
        targets = pd.read_csv(taxonomy_targets_path)
        required = {"failure_taxonomy", "any_system_failure"}
        if required.issubset(targets.columns):
            mask = targets["any_system_failure"].astype(str).str.lower().isin({"true", "1"})
            failures = targets.loc[mask].copy()
            system_counts = (
                failures.groupby("failure_taxonomy", dropna=False)
                .size()
                .rename("count")
                .reset_index()
                .sort_values("count", ascending=False)
            )
            denominator = max(len(failures), 1)
            system_counts["rate_within_system_failures"] = system_counts["count"] / denominator
            system_counts.to_csv(
                tables_dir / "table_taxonomy_system_failures.csv", index=False
            )
    return all_counts, system_counts


def plot_object_type_metrics(frame: pd.DataFrame, figures_dir: Path) -> None:
    if frame.empty:
        return
    metrics = [
        ("top1_accuracy", "Top-1 accuracy"),
        ("topk_coverage", "Adaptive Top-K coverage"),
        ("adb_shadow_coverage", "Predictive ADB coverage"),
        ("average_k", "Average selected beams"),
    ]
    labels = frame["object_type"].astype(str).str.title()
    for metric, ylabel in metrics:
        if metric not in frame:
            continue
        figure, axis = plt.subplots(figsize=(7.5, 4.5))
        axis.bar(labels, frame[metric].astype(float))
        axis.set_xlabel("Object type")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.3)
        save_figure(figure, figures_dir / f"object_types_{metric}")


def plot_taxonomy(
    all_counts: pd.DataFrame,
    system_counts: pd.DataFrame,
    figures_dir: Path,
) -> None:
    if not all_counts.empty:
        data = all_counts[all_counts["failure_taxonomy"] != "no_system_failure"]
        figure, axis = plt.subplots(figsize=(8.5, 4.8))
        axis.bar(data["failure_taxonomy"], data["count"])
        axis.set_xlabel("Diagnostic taxonomy")
        axis.set_ylabel("Targets")
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.3)
        save_figure(figure, figures_dir / "taxonomy_all_diagnostics")

    if not system_counts.empty:
        figure, axis = plt.subplots(figsize=(8.5, 4.8))
        axis.bar(system_counts["failure_taxonomy"], system_counts["count"])
        axis.set_xlabel("Taxonomy among end-to-end system failures")
        axis.set_ylabel("Targets")
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.3)
        save_figure(figure, figures_dir / "taxonomy_system_failures")


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_root = (repo_root / args.output_dir).resolve()
    tables_dir = output_root / "tables"
    figures_dir = output_root / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    loaded: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for name, relative_path in EXPERIMENT_SUMMARIES.items():
        path = repo_root / relative_path
        if path.exists():
            loaded[name] = load_json(path)
        else:
            missing.append(str(relative_path))

    if not loaded:
        raise FileNotFoundError(
            "No experiment summaries were found. Run the experiments first."
        )

    flat_rows: list[dict[str, Any]] = []
    for experiment, summary in loaded.items():
        row: dict[str, Any] = {"source": experiment}
        flatten("", summary, row)
        flat_rows.append(row)
    master = pd.DataFrame(flat_rows)
    master.to_csv(output_root / "master_summary.csv", index=False)

    object_types = pd.DataFrame()
    if "exp06_object_types" in loaded:
        object_types = build_object_type_table(
            loaded["exp06_object_types"], tables_dir
        )

    if "exp07_failures" in loaded:
        build_failure_table(loaded["exp07_failures"], tables_dir)

    all_taxonomy = pd.DataFrame()
    system_taxonomy = pd.DataFrame()
    if "exp07b_taxonomy" in loaded:
        all_taxonomy, system_taxonomy = build_taxonomy_tables(
            loaded["exp07b_taxonomy"],
            repo_root / "results/exp07b/all_targets_with_taxonomy.csv",
            tables_dir,
        )

    plot_object_type_metrics(object_types, figures_dir)
    plot_taxonomy(all_taxonomy, system_taxonomy, figures_dir)

    manifest = {
        "builder": "scripts/build_publication_results.py",
        "loaded_summaries": sorted(loaded),
        "missing_optional_inputs": missing,
        "output_directory": str(output_root),
        "important_distinction": {
            "diagnostic_anomaly_rate": loaded.get("exp07b_taxonomy", {}).get(
                "classified_failure_rate"
            ),
            "end_to_end_system_failure_rate": loaded.get("exp07_failures", {}).get(
                "system_failure_rate"
            ),
            "note": (
                "Diagnostic taxonomy conditions and end-to-end beam/ADB failures "
                "are reported separately."
            ),
        },
    }
    with (output_root / "master_summary.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    print(json.dumps(manifest, indent=2))
    print(f"Saved publication package under: {output_root}")


if __name__ == "__main__":
    main()
