"""Input discovery and spotting JSON loading."""

from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Any

from .models import Event, VideoEvents


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def video_id_from_name(value: str | None, fallback: str) -> str:
    if not value:
        return Path(fallback).stem
    return Path(str(value)).stem


def channel_from_act(value: Any) -> str:
    act = str(value or "").upper()
    if act == "V":
        return "verbal"
    if act == "NV":
        return "nonverbal"
    return "unknown"


def load_spotting_file(path: Path) -> VideoEvents:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")

    video_name = str(payload.get("video") or path.name)
    video_id = video_id_from_name(video_name, path.stem)
    annotations = payload.get("annotations", [])
    if not isinstance(annotations, list):
        raise ValueError(f"{path} has no annotations list")

    events: list[Event] = []
    for index, raw_event in enumerate(annotations):
        if not isinstance(raw_event, dict):
            continue
        start = as_float(raw_event.get("start"))
        end = as_float(raw_event.get("end"), start)
        if end < start:
            start, end = end, start
        event_id = str(raw_event.get("id") or f"event_{index + 1:05d}")
        events.append(
            Event(
                event_id=event_id,
                video_id=video_id,
                start=start,
                end=end,
                channel=channel_from_act(raw_event.get("act")),
                raw=dict(raw_event),
            )
        )

    events.sort(key=lambda event: (event.start, event.end, event.event_id))
    return VideoEvents(video_id=video_id, video_name=video_name, path=path, events=events)


def resolve_path(base_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def discover_input_files(input_spec: dict[str, Any], base_dir: Path) -> list[Path]:
    files: list[Path] = []

    for raw_file in input_spec.get("files", []) or []:
        files.append(resolve_path(base_dir, raw_file))

    raw_glob = input_spec.get("glob")
    if raw_glob:
        pattern = resolve_path(base_dir, raw_glob)
        files.extend(Path(match).resolve() for match in glob.glob(str(pattern)))

    unique: dict[Path, None] = {}
    for path in files:
        unique[path] = None

    discovered = sorted(unique.keys())
    if not discovered:
        raise FileNotFoundError("No input spotting files matched the config")
    return discovered


def load_videos(
    files: list[Path],
    video_ids: list[str] | None = None,
    limit: int | None = None,
) -> list[VideoEvents]:
    allowed = {str(video_id) for video_id in video_ids or []}
    videos: list[VideoEvents] = []

    for path in files:
        video = load_spotting_file(path)
        if allowed and video.video_id not in allowed and path.stem not in allowed:
            continue
        videos.append(video)
        if limit is not None and len(videos) >= limit:
            break

    if not videos:
        raise FileNotFoundError("No input spotting files remained after filtering")
    return videos


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
