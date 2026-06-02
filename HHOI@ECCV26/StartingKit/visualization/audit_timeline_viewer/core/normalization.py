"""Normalize raw audit JSON into the web viewer data contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import quote

from audit_timeline_viewer.core.labels import annotation_display_label
from audit_timeline_viewer.core.values import as_float, natural_video_sort_key
from audit_timeline_viewer.core.parsers import get_parser_for_audit


CHANNEL_NAMES = ("verbal", "nonverbal")


class AuditNormalizer:
    """Transform raw audit structures into the browser-facing schema."""

    def __init__(
        self,
        video_map: dict[str, Path],
        annotation_maps: dict[str, dict[str, dict[str, Any]]],
        subtitle_maps: dict[str, list[dict[str, Any]]],
    ) -> None:
        self.video_map = video_map
        self.annotation_maps = annotation_maps
        self.subtitle_maps = subtitle_maps

    def normalize(self, raw_audit: Any) -> dict[str, Any]:
        raw_videos = self._extract_video_audits(raw_audit)
        videos = [self.normalize_video_audit(video_audit) for video_audit in raw_videos]
        videos.sort(key=natural_video_sort_key)

        total_segments = sum(
            len(channel["segments"])
            for video in videos
            for channel in video["channels"].values()
        )
        total_events = sum(
            len(channel["events"])
            for video in videos
            for channel in video["channels"].values()
        )

        return {
            "videos": videos,
            "summary": {
                "num_videos": len(videos),
                "num_segments": total_segments,
                "num_events": total_events,
            },
        }

    def normalize_video_audit(self, video_audit: dict[str, Any]) -> dict[str, Any]:
        video_id = str(video_audit.get("video_id", "unknown_video"))
        annotations = self.annotation_maps.get(video_id, {})

        parser = get_parser_for_audit(video_audit)
        channels = parser.parse_channels(video_audit, annotations)

        max_time = as_float(video_audit.get("video_duration"), 0.0)
        for channel in channels.values():
            for segment in channel["segments"]:
                max_time = max(max_time, segment["t_e"])
            for event in channel["events"]:
                max_time = max(max_time, event["end"])

        video_path = self.video_map.get(video_id)
        return {
            "video_id": video_id,
            "duration": round(max_time, 3),
            "has_video": video_path is not None,
            "video_url": f"/video/{quote(video_id, safe='')}" if video_path is not None else None,
            "subtitles": self.subtitle_maps.get(video_id, []),
            "channels": channels,
            "skipped_count": len(video_audit.get("skipped", []) or []),
            "num_raw_events_used": video_audit.get("num_raw_events_used", 0),
        }

    @staticmethod
    def _extract_video_audits(raw_audit: Any) -> list[dict[str, Any]]:
        if isinstance(raw_audit, dict):
            channel_first_audits = AuditNormalizer._extract_channel_first_video_audits(raw_audit)
            if channel_first_audits:
                return channel_first_audits
            return [raw_audit]
        if isinstance(raw_audit, list):
            return [item for item in raw_audit if isinstance(item, dict)]
        raise ValueError("Audit JSON must be an object or a list of objects.")

    @staticmethod
    def _extract_channel_first_video_audits(raw_audit: dict[str, Any]) -> list[dict[str, Any]]:
        channel_maps = {
            channel_name: channel_map
            for channel_name in CHANNEL_NAMES
            if isinstance((channel_map := raw_audit.get(channel_name)), dict)
        }
        if not channel_maps:
            return []

        video_ids: list[str] = []
        for channel_map in channel_maps.values():
            for video_id, segments in channel_map.items():
                if isinstance(segments, (dict, list)):
                    normalized_id = str(video_id)
                    if normalized_id not in video_ids:
                        video_ids.append(normalized_id)

        video_audits: list[dict[str, Any]] = []
        for video_id in video_ids:
            channels: dict[str, list[dict[str, Any]]] = {}
            num_events = 0
            duration = 0.0

            for channel_name in CHANNEL_NAMES:
                raw_segments = channel_maps.get(channel_name, {}).get(video_id, [])
                segments = AuditNormalizer._channel_first_segments(raw_segments)
                channels[channel_name] = segments
                for segment in segments:
                    duration = max(duration, as_float(segment.get("t_e"), 0.0))
                    raw_events = segment.get("events", []) or []
                    if isinstance(raw_events, list):
                        num_events += len([event for event in raw_events if isinstance(event, dict)])

            video_audits.append(
                {
                    "video_id": video_id,
                    "video_duration": duration,
                    "num_raw_events_used": num_events,
                    "channels": channels,
                }
            )

        return video_audits

    @staticmethod
    def _channel_first_segments(raw_segments: Any) -> list[dict[str, Any]]:
        segments: list[dict[str, Any]] = []

        if isinstance(raw_segments, dict):
            iterable = raw_segments.items()
        elif isinstance(raw_segments, list):
            iterable = (
                (raw_segment.get("segment", f"s_{index + 1:04d}"), raw_segment)
                for index, raw_segment in enumerate(raw_segments)
                if isinstance(raw_segment, dict)
            )
        else:
            return segments

        for fallback_index, (segment_id, raw_segment) in enumerate(iterable, start=1):
            if not isinstance(raw_segment, dict):
                continue
            segment = dict(raw_segment)
            segment["segment"] = str(segment.get("segment") or segment_id or f"s_{fallback_index:04d}")
            segments.append(segment)

        return segments


def normalize_audit(
    raw_audit: Any,
    video_map: dict[str, Path],
    annotation_maps: dict[str, dict[str, dict[str, Any]]],
    subtitle_maps: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    return AuditNormalizer(video_map, annotation_maps, subtitle_maps).normalize(raw_audit)
