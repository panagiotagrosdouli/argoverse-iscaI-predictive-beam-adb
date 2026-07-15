from __future__ import annotations

from collections.abc import Iterable

from .data import ActorTrack


OBJECT_TYPE_ALIASES: dict[str, set[str]] = {
    "vehicle": {
        "vehicle",
        "bus",
        "large_vehicle",
        "truck",
        "trailer",
        "school_bus",
        "articulated_bus",
    },
    "pedestrian": {"pedestrian"},
    "cyclist": {"cyclist", "bicyclist", "bicycle"},
    "motorcyclist": {"motorcyclist", "motorcycle", "motorbike"},
}


def normalize_object_type(value: str) -> str:
    """Normalize AV2 object-type strings for class-wise evaluation."""
    token = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    for canonical, aliases in OBJECT_TYPE_ALIASES.items():
        if token in aliases:
            return canonical
    return token


def select_targets_by_object_type(
    tracks: dict[str, ActorTrack],
    object_type: str,
    required_steps: int,
    *,
    exclude_track_ids: Iterable[str] = ("AV", "EGO", "AUTONOMOUS_VEHICLE"),
) -> list[ActorTrack]:
    """Return eligible actor tracks belonging to one canonical object class.

    Tracks are required to contain at least ``required_steps`` samples. Ego tracks
    are excluded by ID. The output is sorted deterministically by track ID.
    """
    if required_steps <= 0:
        raise ValueError("required_steps must be positive")

    canonical = normalize_object_type(object_type)
    if canonical not in OBJECT_TYPE_ALIASES:
        supported = ", ".join(sorted(OBJECT_TYPE_ALIASES))
        raise ValueError(f"Unsupported object type '{object_type}'. Supported: {supported}")

    excluded = {str(track_id).strip().upper() for track_id in exclude_track_ids}
    selected = [
        track
        for track in tracks.values()
        if track.track_id.strip().upper() not in excluded
        and normalize_object_type(track.object_type) == canonical
        and len(track.position) >= required_steps
    ]
    return sorted(selected, key=lambda track: track.track_id)
