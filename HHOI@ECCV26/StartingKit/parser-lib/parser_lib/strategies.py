"""Segmentation and event-selection strategies."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Callable

from .models import Event, EventAssignment, Segment

EPSILON = 1e-9


def strategy_name(spec: dict[str, Any], default: str) -> str:
    return str(spec.get("name") or default).strip().lower().replace("-", "_")


def option(spec: dict[str, Any], names: tuple[str, ...], default: Any) -> Any:
    for name in names:
        if name in spec:
            return spec[name]
    return default


def float_option(spec: dict[str, Any], names: tuple[str, ...], default: float) -> float:
    try:
        return float(option(spec, names, default))
    except (TypeError, ValueError):
        return default


def bool_option(spec: dict[str, Any], name: str, default: bool) -> bool:
    value = spec.get(name, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def build_segments(events: list[Event], spec: dict[str, Any] | None = None) -> list[Segment]:
    spec = spec or {}
    name = strategy_name(spec, "uniform")
    aliases = {
        "constant": "uniform",
        "fixed": "uniform",
        "fixed_window": "sliding_window",
        "window": "sliding_window",
        "random_delta": "random_delta_window",
        "random_window": "random_delta_window",
        "event": "event_window",
        "event_aligned": "event_window",
        "events_density": "density",
        "event_density": "density",
        "event_centroids": "centroids",
        "centroid": "centroids",
    }
    name = aliases.get(name, name)

    try:
        builder = SEGMENTATION_STRATEGIES[name]
    except KeyError as exc:
        known = ", ".join(sorted(SEGMENTATION_STRATEGIES))
        raise ValueError(f"Unknown segmentation strategy '{name}'. Known: {known}") from exc
    return builder(events, spec)


def select_events(
    events: list[Event],
    segments: list[Segment],
    spec: dict[str, Any] | None = None,
) -> dict[str, list[EventAssignment]]:
    spec = spec or {}
    name = strategy_name(spec, "first_overlap")
    aliases = {
        "first": "first_overlap",
        "first_segment": "first_overlap",
        "first_match": "first_overlap",
        "first_70": "first_min_overlap_ratio",
        "first_70pct": "first_min_overlap_ratio",
        "first_ratio": "first_min_overlap_ratio",
        "threshold": "first_min_overlap_ratio",
        "every": "every_overlap",
        "all": "every_overlap",
        "all_matches": "every_overlap",
    }
    name = aliases.get(name, name)

    try:
        selector = SELECTION_STRATEGIES[name]
    except KeyError as exc:
        known = ", ".join(sorted(SELECTION_STRATEGIES))
        raise ValueError(f"Unknown selection strategy '{name}'. Known: {known}") from exc

    assignments = selector(sorted(events, key=_event_sort_key), segments, spec)
    for segment_id in assignments:
        assignments[segment_id].sort(key=lambda item: _event_sort_key(item.event))
    return {segment.segment_id: assignments.get(segment.segment_id, []) for segment in segments}


def available_segmentation_strategies() -> list[str]:
    return sorted(SEGMENTATION_STRATEGIES)


def available_selection_strategies() -> list[str]:
    return sorted(SELECTION_STRATEGIES)


def uniform_segments(events: list[Event], spec: dict[str, Any]) -> list[Segment]:
    window_sec = _positive(float_option(spec, ("window_sec", "segment_sec", "size_sec"), 2.0), "window_sec")
    start, end = _timeline_bounds(events, spec)
    include_partial = bool_option(spec, "include_partial", True)
    clip = bool_option(spec, "clip_to_bounds", True)

    windows: list[tuple[float, float]] = []
    current = start
    while current < end - EPSILON:
        raw_end = current + window_sec
        if not include_partial and raw_end > end + EPSILON:
            break
        windows.append((current, min(raw_end, end) if clip else raw_end))
        current += window_sec
    return _segments_from_windows(windows, "uniform")


def sliding_window_segments(events: list[Event], spec: dict[str, Any]) -> list[Segment]:
    window_sec = _positive(float_option(spec, ("window_sec", "segment_sec", "size_sec"), 2.0), "window_sec")
    delta_sec = _positive(float_option(spec, ("delta_sec", "stride_sec", "step_sec"), window_sec), "delta_sec")
    start, end = _timeline_bounds(events, spec)
    include_partial = bool_option(spec, "include_partial", True)
    clip = bool_option(spec, "clip_to_bounds", True)

    windows: list[tuple[float, float]] = []
    current = start
    while current < end - EPSILON:
        raw_end = current + window_sec
        if not include_partial and raw_end > end + EPSILON:
            break
        windows.append((current, min(raw_end, end) if clip else raw_end))
        current += delta_sec
    return _segments_from_windows(windows, "sliding_window")


def random_delta_window_segments(events: list[Event], spec: dict[str, Any]) -> list[Segment]:
    window_sec = _positive(float_option(spec, ("window_sec", "segment_sec", "size_sec"), 2.0), "window_sec")
    delta_min = float_option(spec, ("delta_min_sec", "d1_sec", "min_delta_sec"), 1.0)
    delta_max = float_option(spec, ("delta_max_sec", "d2_sec", "max_delta_sec"), delta_min)
    if delta_min > delta_max:
        delta_min, delta_max = delta_max, delta_min
    if delta_min <= -window_sec:
        raise ValueError("delta_min_sec must be greater than -window_sec")

    seed = int(float_option(spec, ("seed",), 0))
    rng = random.Random(seed)
    start, end = _timeline_bounds(events, spec)
    include_partial = bool_option(spec, "include_partial", True)
    clip = bool_option(spec, "clip_to_bounds", True)

    windows: list[tuple[float, float]] = []
    current = start
    while current < end - EPSILON:
        raw_end = current + window_sec
        if not include_partial and raw_end > end + EPSILON:
            break
        windows.append((current, min(raw_end, end) if clip else raw_end))
        delta = rng.uniform(delta_min, delta_max)
        current = raw_end + delta
    return _segments_from_windows(windows, "random_delta_window")


def event_window_segments(events: list[Event], spec: dict[str, Any]) -> list[Segment]:
    if not events:
        return []

    window_sec = _positive(float_option(spec, ("window_sec", "segment_sec", "size_sec"), 2.0), "window_sec")
    offset_sec = float_option(spec, ("offset_sec", "delta_sec", "d_sec"), 0.0)
    anchor = str(spec.get("anchor", "start")).strip().lower()
    seed = int(float_option(spec, ("seed",), 0))
    rng = random.Random(seed)

    has_random_offset = any(name in spec for name in ("delta_min_sec", "d1_sec", "min_delta_sec"))
    delta_min = float_option(spec, ("delta_min_sec", "d1_sec", "min_delta_sec"), offset_sec)
    delta_max = float_option(spec, ("delta_max_sec", "d2_sec", "max_delta_sec"), delta_min)
    if delta_min > delta_max:
        delta_min, delta_max = delta_max, delta_min

    clip = bool_option(spec, "clip_to_bounds", False)
    dedupe = bool_option(spec, "dedupe", True)
    lower, upper = _timeline_bounds(events, spec)

    windows: list[tuple[float, float]] = []
    for event in sorted(events, key=_event_sort_key):
        delta = rng.uniform(delta_min, delta_max) if has_random_offset else offset_sec
        if anchor == "center":
            start = event.midpoint - window_sec / 2.0 + delta
        elif anchor == "end":
            start = event.end - window_sec + delta
        else:
            start = event.start + delta

        end = start + window_sec
        if clip:
            start = max(lower, start)
            end = min(upper, end)
        else:
            start = max(0.0, start)
        windows.append((start, end))
    return _segments_from_windows(windows, "event_window", dedupe=dedupe)


def density_segments(events: list[Event], spec: dict[str, Any]) -> list[Segment]:
    if not events:
        return []

    max_gap = float_option(spec, ("max_gap_sec", "gap_sec"), 2.0)
    padding = max(0.0, float_option(spec, ("padding_sec", "pad_sec"), 0.0))
    min_window = max(0.001, float_option(spec, ("min_window_sec", "window_sec"), 0.5))

    windows: list[tuple[float, float]] = []
    sorted_events = sorted(events, key=_event_sort_key)
    cluster_start = sorted_events[0].start
    cluster_end = sorted_events[0].end

    for event in sorted_events[1:]:
        if event.start <= cluster_end + max_gap:
            cluster_end = max(cluster_end, event.end)
            continue
        windows.append(_expand_window(cluster_start - padding, cluster_end + padding, min_window))
        cluster_start = event.start
        cluster_end = event.end

    windows.append(_expand_window(cluster_start - padding, cluster_end + padding, min_window))
    return _segments_from_windows(windows, "density")


def centroid_segments(events: list[Event], spec: dict[str, Any]) -> list[Segment]:
    if not events:
        return []

    window_sec = _positive(float_option(spec, ("window_sec", "segment_sec", "size_sec"), 2.0), "window_sec")
    max_gap = float_option(spec, ("max_gap_sec", "gap_sec"), 4.0)
    midpoints = sorted(event.midpoint for event in events)

    clusters: list[list[float]] = [[midpoints[0]]]
    for midpoint in midpoints[1:]:
        if midpoint - clusters[-1][-1] <= max_gap:
            clusters[-1].append(midpoint)
        else:
            clusters.append([midpoint])

    windows: list[tuple[float, float]] = []
    for cluster in clusters:
        center = sum(cluster) / len(cluster)
        start = max(0.0, center - window_sec / 2.0)
        windows.append((start, start + window_sec))
    return _segments_from_windows(windows, "centroids")


def first_overlap_selector(
    events: list[Event],
    segments: list[Segment],
    spec: dict[str, Any],
) -> defaultdict[str, list[EventAssignment]]:
    min_overlap = max(0.0, float_option(spec, ("min_overlap_sec",), 0.0))
    assignments: defaultdict[str, list[EventAssignment]] = defaultdict(list)
    for event in events:
        for segment in segments:
            overlap = overlap_seconds(event, segment)
            ratio = overlap_ratio(event, segment)
            if _matches_overlap(event, segment, overlap, ratio, min_overlap):
                assignments[segment.segment_id].append(EventAssignment(event, segment, overlap, ratio))
                break
    return assignments


def first_min_overlap_ratio_selector(
    events: list[Event],
    segments: list[Segment],
    spec: dict[str, Any],
) -> defaultdict[str, list[EventAssignment]]:
    threshold = float_option(spec, ("threshold", "min_ratio", "overlap_ratio"), 0.70)
    assignments: defaultdict[str, list[EventAssignment]] = defaultdict(list)
    for event in events:
        for segment in segments:
            overlap = overlap_seconds(event, segment)
            ratio = overlap_ratio(event, segment)
            if _matches_overlap(event, segment, overlap, ratio, 0.0) and ratio + EPSILON >= threshold:
                assignments[segment.segment_id].append(EventAssignment(event, segment, overlap, ratio))
                break
    return assignments


def every_overlap_selector(
    events: list[Event],
    segments: list[Segment],
    spec: dict[str, Any],
) -> defaultdict[str, list[EventAssignment]]:
    threshold = float_option(spec, ("threshold", "min_ratio", "overlap_ratio"), 0.0)
    min_overlap = max(0.0, float_option(spec, ("min_overlap_sec",), 0.0))
    assignments: defaultdict[str, list[EventAssignment]] = defaultdict(list)
    for event in events:
        for segment in segments:
            overlap = overlap_seconds(event, segment)
            ratio = overlap_ratio(event, segment)
            if _matches_overlap(event, segment, overlap, ratio, min_overlap) and ratio + EPSILON >= threshold:
                assignments[segment.segment_id].append(EventAssignment(event, segment, overlap, ratio))
    return assignments


def overlap_seconds(event: Event, segment: Segment) -> float:
    if event.duration <= EPSILON:
        return 0.0
    return max(0.0, min(event.end, segment.end) - max(event.start, segment.start))


def overlap_ratio(event: Event, segment: Segment) -> float:
    if event.duration <= EPSILON:
        return 1.0 if segment.start <= event.start <= segment.end else 0.0
    return overlap_seconds(event, segment) / event.duration


def _matches_overlap(
    event: Event,
    segment: Segment,
    overlap: float,
    ratio: float,
    min_overlap: float,
) -> bool:
    if event.duration <= EPSILON:
        return ratio > 0.0
    return overlap > max(EPSILON, min_overlap)


def _timeline_bounds(events: list[Event], spec: dict[str, Any]) -> tuple[float, float]:
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


def _segments_from_windows(
    windows: list[tuple[float, float]],
    reason: str,
    dedupe: bool = True,
) -> list[Segment]:
    normalized: list[tuple[float, float]] = []
    seen: set[tuple[float, float]] = set()

    for start, end in sorted(windows):
        start = max(0.0, float(start))
        end = max(start, float(end))
        if end - start <= EPSILON:
            end = start + 0.001
        key = (round(start, 6), round(end, 6))
        if dedupe and key in seen:
            continue
        seen.add(key)
        normalized.append((start, end))

    return [
        Segment(segment_id=f"s_{index + 1:04d}", start=start, end=end, reason=reason)
        for index, (start, end) in enumerate(normalized)
    ]


def _expand_window(start: float, end: float, min_window: float) -> tuple[float, float]:
    start = max(0.0, start)
    if end - start >= min_window:
        return start, end
    center = (start + end) / 2.0
    expanded_start = max(0.0, center - min_window / 2.0)
    return expanded_start, expanded_start + min_window


def _event_sort_key(event: Event) -> tuple[float, float, str]:
    return event.start, event.end, event.event_id


def _positive(value: float, name: str) -> float:
    if value <= 0:
        raise ValueError(f"{name} must be > 0")
    return value


SegmentationBuilder = Callable[[list[Event], dict[str, Any]], list[Segment]]
SelectionBuilder = Callable[
    [list[Event], list[Segment], dict[str, Any]],
    defaultdict[str, list[EventAssignment]],
]

SEGMENTATION_STRATEGIES: dict[str, SegmentationBuilder] = {
    "uniform": uniform_segments,
    "sliding_window": sliding_window_segments,
    "random_delta_window": random_delta_window_segments,
    "event_window": event_window_segments,
    "density": density_segments,
    "centroids": centroid_segments,
}

SELECTION_STRATEGIES: dict[str, SelectionBuilder] = {
    "first_overlap": first_overlap_selector,
    "first_min_overlap_ratio": first_min_overlap_ratio_selector,
    "every_overlap": every_overlap_selector,
}
