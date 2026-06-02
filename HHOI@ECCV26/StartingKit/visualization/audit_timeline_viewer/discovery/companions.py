"""Discovery for annotation JSON and subtitle SRT companion files."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, TextIO

from audit_timeline_viewer.core.annotations import load_annotation_map
from audit_timeline_viewer.discovery.videos import extract_video_ids
from audit_timeline_viewer.subtitles.srt import load_srt_subtitles


class CompanionFileResolver:
    """Resolve optional files that travel alongside a video or audit JSON."""

    def __init__(self, output: TextIO = sys.stderr) -> None:
        self.output = output

    def resolve(
        self,
        video_id: str,
        input_path: Path,
        requested_video_file: Path | None,
        resolved_video_file: Path | None,
        explicit_file: Path | None,
        suffix: str,
    ) -> Path | None:
        if explicit_file is not None:
            expanded = explicit_file.expanduser()
            if expanded.exists() and expanded.is_file():
                return expanded.resolve()
            print(
                f"Warning: {explicit_file} was not found; {suffix} companion will not be mapped.",
                file=self.output,
            )
            return None

        stems = self.companion_stems(video_id, requested_video_file, resolved_video_file)
        search_roots = self.companion_search_roots(input_path, requested_video_file, resolved_video_file)

        for root in search_roots:
            for stem in stems:
                candidate = root / f"{stem}{suffix}"
                if candidate.exists() and candidate.is_file():
                    return candidate.resolve()

        lowered_id = video_id.lower()
        for root in search_roots:
            if not root.exists():
                continue
            for candidate in sorted(root.iterdir()):
                if (
                    candidate.is_file()
                    and candidate.suffix.lower() == suffix
                    and candidate.stem.lower().startswith(lowered_id)
                ):
                    return candidate.resolve()

        return None

    @staticmethod
    def companion_stems(
        video_id: str,
        requested_video_file: Path | None,
        resolved_video_file: Path | None,
    ) -> list[str]:
        stems: list[str] = []

        def add(stem: str) -> None:
            if stem and stem not in stems:
                stems.append(stem)

        if requested_video_file is not None:
            add(requested_video_file.expanduser().stem)
        if resolved_video_file is not None:
            add(resolved_video_file.stem)
        add(video_id)
        return stems

    @staticmethod
    def companion_search_roots(
        input_path: Path,
        requested_video_file: Path | None,
        resolved_video_file: Path | None,
    ) -> list[Path]:
        roots: list[Path] = []

        def add(path: Path) -> None:
            resolved = path.expanduser().resolve()
            if resolved not in roots:
                roots.append(resolved)

        if requested_video_file is not None:
            add(requested_video_file.expanduser().parent)
        if resolved_video_file is not None:
            add(resolved_video_file.parent)
        add(input_path.expanduser().parent)
        add(Path.cwd())
        return roots


class CompanionDiscoveryService:
    """Load all companion metadata maps needed by the viewer."""

    def __init__(self, resolver: CompanionFileResolver | None = None) -> None:
        self.resolver = resolver or CompanionFileResolver()

    def discover_annotation_maps(
        self,
        raw_audit: Any,
        input_path: Path,
        video_file: Path | None,
        video_map: dict[str, Path],
        annotation_file: Path | None,
    ) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, Path]]:
        annotation_maps: dict[str, dict[str, dict[str, Any]]] = {}
        sources: dict[str, Path] = {}

        for video_id in extract_video_ids(raw_audit):
            path = self.resolver.resolve(
                video_id,
                input_path,
                video_file,
                video_map.get(video_id),
                annotation_file,
                ".json",
            )
            if path is None:
                continue
            annotations = load_annotation_map(path)
            if annotations:
                annotation_maps[video_id] = annotations
                sources[video_id] = path

        return annotation_maps, sources

    def discover_subtitle_maps(
        self,
        raw_audit: Any,
        input_path: Path,
        video_file: Path | None,
        video_map: dict[str, Path],
        subtitle_file: Path | None,
        subtitle_root: Path | None = None,
    ) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Path]]:
        subtitle_maps: dict[str, list[dict[str, Any]]] = {}
        sources: dict[str, Path] = {}
        explicit_subtitle_file = subtitle_file
        resolved_subtitle_root = self._resolve_subtitle_root(subtitle_file, subtitle_root)

        if explicit_subtitle_file is not None and explicit_subtitle_file.expanduser().is_dir():
            explicit_subtitle_file = None

        for video_id in extract_video_ids(raw_audit):
            path = None
            if explicit_subtitle_file is not None:
                path = self.resolver.resolve(
                    video_id,
                    input_path,
                    video_file,
                    video_map.get(video_id),
                    explicit_subtitle_file,
                    ".srt",
                )
            elif resolved_subtitle_root is not None:
                path = self.resolve_from_root(video_id, resolved_subtitle_root, ".srt")
            else:
                path = self.resolver.resolve(
                    video_id,
                    input_path,
                    video_file,
                    video_map.get(video_id),
                    None,
                    ".srt",
                )
            if path is None:
                continue
            subtitles = load_srt_subtitles(path)
            if subtitles:
                subtitle_maps[video_id] = subtitles
                sources[video_id] = path

        return subtitle_maps, sources

    def _resolve_subtitle_root(
        self,
        subtitle_file: Path | None,
        subtitle_root: Path | None,
    ) -> Path | None:
        root = subtitle_root
        if root is None and subtitle_file is not None and subtitle_file.expanduser().is_dir():
            root = subtitle_file
        if root is None:
            return None

        expanded = root.expanduser()
        if not expanded.exists() or not expanded.is_dir():
            print(
                f"Warning: subtitle directory {root} was not found; subtitles will not be mapped from it.",
                file=self.resolver.output,
            )
            return None
        return expanded.resolve()

    @staticmethod
    def resolve_from_root(video_id: str, root: Path, suffix: str) -> Path | None:
        exact = root / f"{video_id}{suffix}"
        if exact.exists() and exact.is_file():
            return exact.resolve()

        lowered_id = video_id.lower()
        fallback: Path | None = None
        for candidate in sorted(root.rglob(f"*{suffix}")):
            if not candidate.is_file():
                continue
            stem = candidate.stem.lower()
            name = candidate.name.lower()
            if stem.startswith(lowered_id):
                return candidate.resolve()
            if lowered_id in name and fallback is None:
                fallback = candidate.resolve()
        return fallback


def discover_annotation_maps(
    raw_audit: Any,
    input_path: Path,
    video_file: Path | None,
    video_map: dict[str, Path],
    annotation_file: Path | None,
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, Path]]:
    return CompanionDiscoveryService().discover_annotation_maps(
        raw_audit,
        input_path,
        video_file,
        video_map,
        annotation_file,
    )


def discover_subtitle_maps(
    raw_audit: Any,
    input_path: Path,
    video_file: Path | None,
    video_map: dict[str, Path],
    subtitle_file: Path | None,
    subtitle_root: Path | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Path]]:
    return CompanionDiscoveryService().discover_subtitle_maps(
        raw_audit,
        input_path,
        video_file,
        video_map,
        subtitle_file,
        subtitle_root,
    )


def resolve_companion_file(
    video_id: str,
    input_path: Path,
    requested_video_file: Path | None,
    resolved_video_file: Path | None,
    explicit_file: Path | None,
    suffix: str,
) -> Path | None:
    return CompanionFileResolver().resolve(
        video_id,
        input_path,
        requested_video_file,
        resolved_video_file,
        explicit_file,
        suffix,
    )


def companion_stems(
    video_id: str,
    requested_video_file: Path | None,
    resolved_video_file: Path | None,
) -> list[str]:
    return CompanionFileResolver.companion_stems(video_id, requested_video_file, resolved_video_file)


def companion_search_roots(
    input_path: Path,
    requested_video_file: Path | None,
    resolved_video_file: Path | None,
) -> list[Path]:
    return CompanionFileResolver.companion_search_roots(input_path, requested_video_file, resolved_video_file)
