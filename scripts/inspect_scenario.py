from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from iscai.data import extract_actor_tracks, load_scenario_parquet  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect one AV2 motion scenario")
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/figures/scenario.png"))
    parser.add_argument("--max-tracks", type=int, default=40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = load_scenario_parquet(args.scenario)
    tracks = extract_actor_tracks(frame)

    figure, axis = plt.subplots(figsize=(9, 8))
    ranked = sorted(tracks.values(), key=lambda track: len(track.position), reverse=True)
    for track in ranked[: args.max_tracks]:
        axis.plot(track.position[:, 0], track.position[:, 1], linewidth=1.2, alpha=0.8)
        axis.scatter(track.position[-1, 0], track.position[-1, 1], s=12)

    axis.set_title(f"AV2 scenario: {args.scenario.stem} ({len(tracks)} tracks)")
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.axis("equal")
    axis.grid(True, alpha=0.25)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(args.output, dpi=180)
    plt.close(figure)

    print(f"Loaded {len(frame)} states from {len(tracks)} tracks")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
