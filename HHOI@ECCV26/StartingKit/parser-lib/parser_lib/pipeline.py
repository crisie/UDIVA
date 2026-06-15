"""End-to-end parser pipeline."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from .config import ParserConfig, load_config
from .formatting import anticipation_event_payload, event_payload, fields_for_channel
from .io import discover_input_files, load_videos, resolve_path, write_json
from .models import Event, Segment, VideoEvents
from .strategies import (
    EPSILON,
    bool_option,
    build_segments,
    float_option,
    option,
    overlap_seconds,
    select_events,
    strategy_name,
)

DEFAULT_CHANNELS = ["verbal", "nonverbal"]
DEFAULT_PARTICIPANTS = ["participant_a", "participant_b"]
ANTICIPATION_SELECTION_STRATEGIES = ["next_events", "prediction_window"]
SEGMENT_OFFSET_OPTION_NAMES = (
    "segment_offset_sec",
    "offset_of_segment_sec",
    "segment_gap_sec",
    "inter_segment_offset_sec",
)


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
    if config.task == "recognition":
        return build_recognition_outputs(videos, config)
    if config.task == "anticipation":
        return build_anticipation_outputs(videos, config)

    raise ValueError("Unknown task '{}'. Known: anticipation, recognition".format(config.task))


def available_anticipation_selection_strategies() -> list[str]:
    return list(ANTICIPATION_SELECTION_STRATEGIES)


def build_recognition_outputs(videos: list[VideoEvents], config: ParserConfig) -> dict[str, dict[str, Any]]:
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


def build_anticipation_outputs(videos: list[VideoEvents], config: ParserConfig) -> dict[str, dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    for output_name, output_spec in config.outputs.items():
        segmentation_spec = merged_strategy(config.defaults.get("segmentation"), output_spec.get("segmentation"))
        selection_spec = merged_strategy(config.defaults.get("selection"), output_spec.get("selection"))
        participants = output_participants(config, output_spec, selection_spec)

        payload: dict[str, Any] = {"anticipation": {}}
        for video in sorted(videos, key=lambda item: item.video_id):
            target_segments = build_anticipation_segments(
                video.events,
                segmentation_spec,
                selection_spec,
                participants,
                video.duration,
            )
            assignments = select_anticipation_events(
                video.events,
                target_segments,
                participants,
                selection_spec,
            )
            payload["anticipation"][video.video_id] = format_anticipation_segments(
                target_segments,
                assignments,
                participants,
                output_spec,
                output_name,
            )

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


def build_anticipation_segments(
    events: list[Event],
    segmentation_spec: dict[str, Any],
    selection_spec: dict[str, Any],
    participants: list[str],
    video_duration: float,
) -> list[Segment]:
    if uses_compound_segment_spacing(segmentation_spec, selection_spec):
        return build_compound_anticipation_segments(
            events,
            segmentation_spec,
            selection_spec,
            participants,
            video_duration,
        )

    context_segments = build_segments(events, segmentation_spec)
    return [
        prediction_segment_from_context(
            segment,
            events,
            selection_spec,
            participants,
            video_duration,
        )
        for segment in context_segments
    ]


def build_compound_anticipation_segments(
    events: list[Event],
    segmentation_spec: dict[str, Any],
    selection_spec: dict[str, Any],
    participants: list[str],
    video_duration: float,
) -> list[Segment]:
    name = anticipation_segmentation_name(segmentation_spec)
    if name not in {"uniform", "sliding_window", "random_delta_window"}:
        raise ValueError(
            "Anticipation segment offsets are supported for uniform, sliding_window, "
            "and random_delta_window segmentation"
        )

    window_sec = _positive_float_option(
        segmentation_spec,
        ("window_sec", "segment_sec", "size_sec"),
        2.0,
    )
    timeline_start, timeline_end = timeline_bounds(events, segmentation_spec)
    include_partial = bool_option(segmentation_spec, "include_partial", True)
    clip = bool_option(segmentation_spec, "clip_to_bounds", True)
    segment_offset = compound_segment_offset(segmentation_spec, selection_spec)
    rng = random.Random(int(float_option(segmentation_spec, ("seed",), 0)))

    target_segments: list[Segment] = []
    current = timeline_start
    while current < timeline_end - EPSILON:
        raw_context_end = current + window_sec
        if not include_partial and raw_context_end > timeline_end + EPSILON:
            break

        context_end = min(raw_context_end, timeline_end) if clip else raw_context_end
        context_segment = Segment(
            segment_id=f"s_{len(target_segments) + 1:04d}",
            start=current,
            end=context_end,
            reason=f"{name}_context",
        )
        prediction_segment = prediction_segment_from_context(
            context_segment,
            events,
            selection_spec,
            participants,
            video_duration,
        )
        target_segments.append(prediction_segment)

        offset_sec = segment_offset
        if offset_sec is None:
            offset_sec = random_delta_segment_offset(segmentation_spec, rng)
        next_start = prediction_segment.end + offset_sec
        if next_start <= current + EPSILON:
            raise ValueError(
                "segment_offset_sec must advance the next observation window; "
                f"got next start {next_start:.6f} after current start {current:.6f}"
            )
        current = next_start

    return target_segments


def prediction_segment_from_context(
    context_segment: Segment,
    events: list[Event],
    selection_spec: dict[str, Any],
    participants: list[str],
    video_duration: float,
) -> Segment:
    offset_sec = float_option(selection_spec, ("offset_sec", "prediction_offset_sec", "gap_sec"), 0.0)
    start_from = str(
        option(
            selection_spec,
            ("prediction_start", "target_start", "start_from"),
            "segment_end",
        )
    ).strip().lower()
    clip = bool_option(selection_spec, "clip_to_bounds", False)

    if start_from in {"segment_start", "start", "window_start"}:
        start = context_segment.start + offset_sec
    elif start_from in {"segment_end", "end", "window_end", "after_context"}:
        start = context_segment.end + offset_sec
    else:
        raise ValueError(
            "Unknown anticipation prediction_start '{}'. Known: segment_end, segment_start".format(
                start_from
            )
        )

    start = max(0.0, start)
    if anticipation_selection_name(selection_spec) == "next_events":
        end = next_events_prediction_end(events, start, selection_spec, participants)
    else:
        horizon_sec = _positive_float_option(
            selection_spec,
            ("prediction_window_sec", "horizon_sec", "prediction_horizon_sec", "target_window_sec"),
            max(context_segment.duration, 0.001),
        )
        end = start + horizon_sec

    if clip:
        end = min(end, video_duration)
        end = max(start, end)

    return Segment(
        segment_id=context_segment.segment_id,
        start=start,
        end=end,
        reason="anticipation",
    )


def select_anticipation_events(
    events: list[Event],
    target_segments: list[Segment],
    participants: list[str],
    selection_spec: dict[str, Any],
) -> dict[str, dict[str, list[Event]]]:
    name = anticipation_selection_name(selection_spec)
    channels = anticipation_channels(selection_spec)
    max_events = int_option(selection_spec, ("max_events", "next_n", "n_events"), None)
    if name == "next_events" and max_events is None:
        max_events = 1

    min_overlap = max(0.0, float_option(selection_spec, ("min_overlap_sec",), 0.0))
    candidates = [
        event
        for event in sorted(events, key=_event_sort_key)
        if event.channel in channels and event_participant(event) in participants
    ]

    assignments: dict[str, dict[str, list[Event]]] = {}
    for segment in target_segments:
        by_participant: dict[str, list[Event]] = {}
        for participant in participants:
            participant_events = [
                event for event in candidates if event_participant(event) == participant
            ]
            if name == "next_events":
                selected = [
                    event
                    for event in participant_events
                    if event.start + EPSILON >= segment.start
                ]
            else:
                selected = [
                    event
                    for event in participant_events
                    if event_matches_prediction_window(event, segment, min_overlap)
                ]

            if max_events is not None:
                selected = selected[: max(0, max_events)]
            by_participant[participant] = selected
        assignments[segment.segment_id] = by_participant

    return assignments


def format_anticipation_segments(
    segments: list[Segment],
    assignments: dict[str, dict[str, list[Event]]],
    participants: list[str],
    output_spec: dict[str, Any],
    output_name: str,
) -> dict[str, dict[str, Any]]:
    time_decimals = int(output_spec.get("time_decimals", 3))
    include_events = bool(output_spec.get("include_events", True))
    include_hypotheses = output_uses_hypotheses(output_name, output_spec)
    segments_payload: dict[str, dict[str, Any]] = {}

    for segment in segments:
        participant_payload: dict[str, dict[str, Any]] = {}
        segment_assignments = assignments.get(segment.segment_id, {})
        for participant in participants:
            participant_events = segment_assignments.get(participant, [])
            events = []
            for event in participant_events:
                payload = anticipation_event_payload(event, output_spec)
                if payload:
                    events.append(payload)
            if not include_events:
                events = []

            if include_hypotheses:
                participant_payload[participant] = {"hypotheses": [{"events": events}]}
            else:
                participant_payload[participant] = {"events": events}

        segments_payload[segment.segment_id] = {
            "t_b": round(segment.start, time_decimals),
            "t_e": round(segment.end, time_decimals),
            "participants": participant_payload,
        }

    return segments_payload


def anticipation_selection_name(spec: dict[str, Any]) -> str:
    name = strategy_name(spec, "prediction_window")
    aliases = {
        "all": "prediction_window",
        "all_matches": "prediction_window",
        "every": "prediction_window",
        "every_overlap": "prediction_window",
        "first": "prediction_window",
        "first_overlap": "prediction_window",
        "future": "prediction_window",
        "future_events": "prediction_window",
        "future_window": "prediction_window",
        "offset": "prediction_window",
        "offset_window": "prediction_window",
        "window": "prediction_window",
        "next": "next_events",
        "next_event": "next_events",
        "next_n": "next_events",
        "next_n_events": "next_events",
    }
    name = aliases.get(name, name)
    if name not in ANTICIPATION_SELECTION_STRATEGIES:
        known = ", ".join(ANTICIPATION_SELECTION_STRATEGIES)
        raise ValueError(
            "Unknown anticipation selection strategy '{}'. Known: {}".format(name, known)
        )
    return name


def uses_compound_segment_spacing(
    segmentation_spec: dict[str, Any],
    selection_spec: dict[str, Any],
) -> bool:
    if bool_option(selection_spec, "compound_segment_spacing", False):
        return True
    if bool_option(segmentation_spec, "compound_segment_spacing", False):
        return True
    return has_any_option(selection_spec, SEGMENT_OFFSET_OPTION_NAMES) or has_any_option(
        segmentation_spec,
        SEGMENT_OFFSET_OPTION_NAMES,
    )


def anticipation_segmentation_name(spec: dict[str, Any]) -> str:
    name = strategy_name(spec, "uniform")
    aliases = {
        "constant": "uniform",
        "fixed": "uniform",
        "fixed_window": "sliding_window",
        "window": "sliding_window",
        "random_delta": "random_delta_window",
        "random_window": "random_delta_window",
    }
    return aliases.get(name, name)


def timeline_bounds(events: list[Event], spec: dict[str, Any]) -> tuple[float, float]:
    start = float_option(spec, ("start_sec", "t_b"), 0.0)
    if "end_sec" in spec:
        end = float_option(spec, ("end_sec", "t_e"), start)
    elif "duration_sec" in spec:
        end = start + max(0.0, float_option(spec, ("duration_sec",), 0.0))
    elif events:
        end = max(event.end for event in events)
    else:
        end = start
    return start, max(start, end)


def compound_segment_offset(
    segmentation_spec: dict[str, Any],
    selection_spec: dict[str, Any],
) -> float | None:
    if has_any_option(selection_spec, SEGMENT_OFFSET_OPTION_NAMES):
        return float_option(selection_spec, SEGMENT_OFFSET_OPTION_NAMES, 0.0)
    if has_any_option(segmentation_spec, SEGMENT_OFFSET_OPTION_NAMES):
        return float_option(segmentation_spec, SEGMENT_OFFSET_OPTION_NAMES, 0.0)
    return None


def random_delta_segment_offset(spec: dict[str, Any], rng: random.Random) -> float:
    if anticipation_segmentation_name(spec) != "random_delta_window":
        return 0.0

    delta_min = float_option(spec, ("delta_min_sec", "d1_sec", "min_delta_sec"), 0.0)
    delta_max = float_option(spec, ("delta_max_sec", "d2_sec", "max_delta_sec"), delta_min)
    if delta_min > delta_max:
        delta_min, delta_max = delta_max, delta_min
    return rng.uniform(delta_min, delta_max)


def next_events_prediction_end(
    events: list[Event],
    prediction_start: float,
    selection_spec: dict[str, Any],
    participants: list[str],
) -> float:
    channels = anticipation_channels(selection_spec)
    max_events = int_option(selection_spec, ("max_events", "next_n", "n_events"), 1)
    selected_events: list[Event] = []
    candidates = [
        event
        for event in sorted(events, key=_event_sort_key)
        if event.channel in channels and event_participant(event) in participants
    ]

    for participant in participants:
        participant_events = [
            event
            for event in candidates
            if event_participant(event) == participant and event.start + EPSILON >= prediction_start
        ]
        selected_events.extend(participant_events[: max(0, max_events or 0)])

    if not selected_events:
        return prediction_start
    return max(max(event.start, event.end) for event in selected_events)


def anticipation_channels(spec: dict[str, Any]) -> set[str]:
    raw = option(spec, ("event_type", "event_types", "type", "channel", "channels"), "all")
    if isinstance(raw, list):
        values = {str(value).strip().lower().replace("-", "_") for value in raw}
    else:
        values = {str(raw).strip().lower().replace("-", "_")}

    if values & {"all", "both", "mixed", "verbal+nonverbal", "verbal_nonverbal"}:
        return set(DEFAULT_CHANNELS)

    channels: set[str] = set()
    if values & {"verbal", "v"}:
        channels.add("verbal")
    if values & {"nonverbal", "non_verbal", "nv"}:
        channels.add("nonverbal")
    if not channels:
        raise ValueError("Anticipation event_type must be one of: all, verbal, nonverbal")
    return channels


def output_participants(
    config: ParserConfig,
    output_spec: dict[str, Any],
    selection_spec: dict[str, Any],
) -> list[str]:
    raw = output_spec.get(
        "participants",
        selection_spec.get("participants", config.defaults.get("participants", DEFAULT_PARTICIPANTS)),
    )
    if isinstance(raw, list) and raw:
        return [str(participant) for participant in raw]
    return list(DEFAULT_PARTICIPANTS)


def output_uses_hypotheses(output_name: str, output_spec: dict[str, Any]) -> bool:
    if "include_hypotheses" in output_spec:
        return bool(output_spec.get("include_hypotheses"))
    output_format = str(output_spec.get("format", "")).strip().lower()
    if output_format in {"hypotheses", "submission", "example", "template"}:
        return True
    if output_format in {"events", "reference"}:
        return False
    return output_name in {"example", "submission", "template"}


def event_matches_prediction_window(event: Event, segment: Segment, min_overlap: float) -> bool:
    if not (segment.start - EPSILON <= event.start <= segment.end + EPSILON):
        return False
    if min_overlap <= 0:
        return True
    if event.duration <= EPSILON:
        return True
    return overlap_seconds(event, segment) + EPSILON >= min_overlap


def event_participant(event: Event) -> str:
    return str(event.raw.get("subject") or "")


def has_any_option(spec: dict[str, Any], names: tuple[str, ...]) -> bool:
    return any(name in spec for name in names)


def int_option(
    spec: dict[str, Any],
    names: tuple[str, ...],
    default: int | None,
) -> int | None:
    value = option(spec, names, default)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _positive_float_option(
    spec: dict[str, Any],
    names: tuple[str, ...],
    default: float,
) -> float:
    value = float_option(spec, names, default)
    if value <= 0:
        raise ValueError(f"{names[0]} must be > 0")
    return value


def _event_sort_key(event: Event) -> tuple[float, float, str]:
    return event.start, event.end, event.event_id


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
