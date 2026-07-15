from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "track_id",
    "object_type",
    "timestep",
    "position_x",
    "position_y",
}


@dataclass(frozen=True)
class ActorTrack:
    """One actor trajectory ordered by timestep."""

    track_id: str
    object_type: str
    timestep: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    heading: np.ndarray


def load_scenario_parquet(path: str | Path) -> pd.DataFrame:
    """Load and validate one Argoverse 2 motion-forecasting scenario parquet."""
    scenario_path = Path(path)
    if not scenario_path.exists():
        raise FileNotFoundError(f"Scenario file not found: {scenario_path}")
    if scenario_path.suffix.lower() != ".parquet":
        raise ValueError("Expected an Argoverse scenario .parquet file")

    frame = pd.read_parquet(scenario_path)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required AV2 columns: {sorted(missing)}")
    return frame.sort_values(["track_id", "timestep"]).reset_index(drop=True)


def _finite_difference(position: np.ndarray, timestep: np.ndarray, dt: float) -> np.ndarray:
    if len(position) < 2:
        return np.zeros_like(position, dtype=float)
    time_s = np.asarray(timestep, dtype=float) * dt
    return np.column_stack(
        [
            np.gradient(position[:, 0], time_s),
            np.gradient(position[:, 1], time_s),
        ]
    )


def extract_actor_tracks(frame: pd.DataFrame, dt: float = 0.1) -> dict[str, ActorTrack]:
    """Convert a scenario dataframe into numeric actor tracks.

    AV2 motion forecasting is sampled at 10 Hz, therefore ``dt=0.1`` by default.
    Existing velocity and heading columns are used when available; otherwise they
    are estimated from position samples.
    """
    tracks: dict[str, ActorTrack] = {}
    for track_id, group in frame.groupby("track_id", sort=False):
        group = group.sort_values("timestep")
        position = group[["position_x", "position_y"]].to_numpy(dtype=float)
        timestep = group["timestep"].to_numpy(dtype=int)

        if {"velocity_x", "velocity_y"}.issubset(group.columns):
            velocity = group[["velocity_x", "velocity_y"]].to_numpy(dtype=float)
            invalid = ~np.isfinite(velocity).all(axis=1)
            if invalid.any():
                estimated = _finite_difference(position, timestep, dt)
                velocity[invalid] = estimated[invalid]
        else:
            velocity = _finite_difference(position, timestep, dt)

        if "heading" in group.columns:
            heading = group["heading"].to_numpy(dtype=float)
        else:
            heading = np.arctan2(velocity[:, 1], velocity[:, 0])

        tracks[str(track_id)] = ActorTrack(
            track_id=str(track_id),
            object_type=str(group["object_type"].iloc[0]),
            timestep=timestep,
            position=position,
            velocity=velocity,
            heading=heading,
        )
    return tracks


def find_ego_track(tracks: dict[str, ActorTrack]) -> ActorTrack:
    """Return the autonomous-vehicle track from an AV2 scenario.

    In AV2 motion-forecasting parquet files, the autonomous vehicle is normally
    identified by ``track_id == 'AV'`` while its object type may simply be
    ``vehicle``. Some converted datasets instead encode ego status in the object
    type, so both conventions are supported.
    """
    for track in tracks.values():
        if track.track_id.strip().upper() in {"AV", "EGO", "AUTONOMOUS_VEHICLE"}:
            return track

    for track in tracks.values():
        if track.object_type.strip().upper() in {
            "AV",
            "EGO",
            "AUTONOMOUS_VEHICLE",
        }:
            return track

    available_ids = sorted(track.track_id for track in tracks.values())[:10]
    raise ValueError(
        "No AV/ego track was found in this scenario. "
        f"First available track IDs: {available_ids}"
    )


def split_observation_future(
    track: ActorTrack,
    observation_steps: int = 50,
    future_steps: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    """Split an actor track into observed and future position arrays."""
    required = observation_steps + future_steps
    if len(track.position) < required:
        raise ValueError(
            f"Track {track.track_id} has {len(track.position)} samples; {required} required"
        )
    return (
        track.position[:observation_steps].copy(),
        track.position[observation_steps:required].copy(),
    )
