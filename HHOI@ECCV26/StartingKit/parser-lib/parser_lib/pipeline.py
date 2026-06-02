"""End-to-end parser pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import ParserConfig, load_config
from .formatting import event_payload, fields_for_channel
from .io import discover_input_files, load_videos, resolve_path, write_json
from .models import Segment, VideoEvents
from .strategies import build_segments, select_events

DEFAULT_CHANNELS = ["verbal", "nonverbal"]


def run_config(
    config_path: str | Path,
    output_dir: str | Path | None = None,
    video_ids: list[str] | None = None,
    limit: int | None = None,
) -> dict[str, Path]:
    config = load_config(config_path)
    files = discover_input_files(config.input, config.base_dir)

    configured_video_ids = [str(video_id) for video_id in config.input.get("video_ids", []) or []]
    selected_video_ids = video_ids if video_ids is not None else configured_video_ids
    selected_limit = limit if limit is not None else config.input.get("limit")
    if selected_limit is not None:
        selected_limit = int(selected_limit)

    videos = load_videos(files, video_ids=selected_video_ids, limit=selected_limit)
    payloads = build_outputs(videos, config)

    output_paths: dict[str, Path] = {}
    base_output_dir = Path(output_dir).resolve() if output_dir else None
    for output_name, payload in payloads.items():
        output_spec = config.outputs[output_name]
        configured_path = str(output_spec.get("path") or f"{output_name}.json")
        output_path = (
            base_output_dir / Path(configured_path).name
            if base_output_dir
            else resolve_path(config.base_dir, configured_path)
        )
        write_json(output_path, payload)
        output_paths[output_name] = output_path

    return output_paths


def build_outputs(videos: list[VideoEvents], config: ParserConfig) -> dict[str, dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    for output_name, output_spec in config.outputs.items():
        segmentation_spec = merged_strategy(config.defaults.get("segmentation"), output_spec.get("segmentation"))
        selection_spec = merged_strategy(config.defaults.get("selection"), output_spec.get("selection"))
        channels = output_channels(config, output_spec)

        payload: dict[str, Any] = {channel: {} for channel in channels}
        for video in sorted(videos, key=lambda item: item.video_id):
            segments = build_segments(video.events, segmentation_spec)
            for channel in channels:
                channel_events = [event for event in video.events if event.channel == channel]
                assignments = select_events(channel_events, segments, selection_spec)
                payload[channel][video.video_id] = format_segments(segments, assignments, output_spec)

        payloads[output_name] = payload

    return payloads


def format_segments(
    segments: list[Segment],
    assignments: dict[str, Any],
    output_spec: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    time_decimals = int(output_spec.get("time_decimals", 3))
    segments_payload: dict[str, dict[str, Any]] = {}

    for segment in segments:
        segment_assignments = assignments.get(segment.segment_id, [])

        events = []
        if output_spec.get("include_events"):
            events = [
                event_payload(
                    assignment.event,
                    fields_for_channel(output_spec, assignment.event.channel),
                    output_spec,
                )
                for assignment in segment_assignments
            ]

        segments_payload[segment.segment_id] = {
            "t_b": round(segment.start, time_decimals),
            "t_e": round(segment.end, time_decimals),
            "events": events,
        }
    return segments_payload


def merged_strategy(default_spec: Any, output_spec: Any) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    if isinstance(default_spec, dict):
        merged.update(default_spec)
    if isinstance(output_spec, dict):
        merged.update(output_spec)
    return merged


def output_channels(config: ParserConfig, output_spec: dict[str, Any]) -> list[str]:
    raw_channels = output_spec.get("channels", config.defaults.get("channels", DEFAULT_CHANNELS))
    if isinstance(raw_channels, list) and raw_channels:
        return [str(channel) for channel in raw_channels]
    return list(DEFAULT_CHANNELS)
