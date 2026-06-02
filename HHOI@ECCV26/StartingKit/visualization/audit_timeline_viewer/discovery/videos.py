"""Video file discovery."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Iterable, TextIO


VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".webm", ".mkv"}
CHANNEL_NAMES = ("verbal", "nonverbal")


class VideoDiscoveryService:
    """Resolve audit video IDs to local media files."""

    def __init__(self, output: TextIO = sys.stderr) -> None:
        self.output = output

    def discover(
        self,
        raw_audit: Any,
        video_file: Path | None,
        video_root: Path | None,
    ) -> dict[str, Path]:
        video_ids = extract_video_ids(raw_audit)
        video_map: dict[str, Path] = {}

        if video_file is not None:
            if len(video_ids) == 1:
                video_path = self.resolve_video_file(video_ids[0], video_file)
                if video_path is not None:
                    video_map[video_ids[0]] = video_path
            else:
                print(
                    "--video-file is only mapped automatically when the audit has one video.",
                    file=self.output,
                )

        if video_root is not None and video_root.exists():
            candidates = [
                path.resolve()
                for path in video_root.rglob("*")
                if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
            ]
            for video_id in video_ids:
                if video_id in video_map:
                    continue
                match = first_video_match(video_id, candidates)
                if match is not None:
                    video_map[video_id] = match

        return video_map

    def resolve_video_file(self, video_id: str, video_file: Path) -> Path | None:
        expanded = video_file.expanduser()
        if expanded.exists() and expanded.is_file():
            return expanded.resolve()

        search_root = expanded.parent if expanded.parent != Path("") else Path.cwd()
        if not search_root.exists():
            search_root = Path.cwd()

        candidates = [
            path.resolve()
            for path in search_root.iterdir()
            if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
        ]
        fallback = first_video_match(video_id, candidates)
        if fallback is not None:
            print(
                f"Warning: --video-file {video_file} was not found; using nearby match {fallback}.",
                file=self.output,
            )
            return fallback

        print(f"Warning: --video-file {video_file} was not found; video will not be mapped.", file=self.output)
        return None


def discover_video_map(
    raw_audit: Any,
    video_file: Path | None,
    video_root: Path | None,
) -> dict[str, Path]:
    return VideoDiscoveryService().discover(raw_audit, video_file, video_root)


def resolve_video_file(video_id: str, video_file: Path) -> Path | None:
    return VideoDiscoveryService().resolve_video_file(video_id, video_file)


def extract_video_ids(raw_audit: Any) -> list[str]:
    if isinstance(raw_audit, dict):
        channel_first_ids = extract_channel_first_video_ids(raw_audit)
        if channel_first_ids:
            return channel_first_ids
        return [str(raw_audit.get("video_id", "unknown_video"))]
    if isinstance(raw_audit, list):
        return [
            str(item.get("video_id", "unknown_video"))
            for item in raw_audit
            if isinstance(item, dict)
        ]
    return []


def extract_channel_first_video_ids(raw_audit: dict[str, Any]) -> list[str]:
    video_ids: list[str] = []
    for channel_name in CHANNEL_NAMES:
        channel_map = raw_audit.get(channel_name)
        if not isinstance(channel_map, dict):
            continue
        for video_id, segments in channel_map.items():
            if not isinstance(segments, (dict, list)):
                continue
            normalized_id = str(video_id)
            if normalized_id not in video_ids:
                video_ids.append(normalized_id)
    return video_ids


def first_video_match(video_id: str, candidates: Iterable[Path]) -> Path | None:
    lowered_id = video_id.lower()
    fallback: Path | None = None
    for path in candidates:
        name = path.name.lower()
        stem = path.stem.lower()
        if stem.startswith(lowered_id):
            return path
        if lowered_id in name and fallback is None:
            fallback = path
    return fallback
