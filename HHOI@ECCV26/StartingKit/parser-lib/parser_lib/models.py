"""Core data models used by the parser strategies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Event:
    """One raw spotting annotation normalized enough for strategy code."""

    event_id: str
    video_id: str
    start: float
    end: float
    channel: str
    raw: dict[str, Any]

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def midpoint(self) -> float:
        return self.start + self.duration / 2.0


@dataclass(frozen=True)
class Segment:
    """One parser segment/window."""

    segment_id: str
    start: float
    end: float
    reason: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(frozen=True)
class EventAssignment:
    """Event plus the overlap metrics that caused it to be assigned."""

    event: Event
    segment: Segment
    overlap_sec: float
    overlap_ratio: float


@dataclass(frozen=True)
class VideoEvents:
    """All normalized spotting events for a single input file."""

    video_id: str
    video_name: str
    path: Path
    events: list[Event]

    @property
    def duration(self) -> float:
        if not self.events:
            return 0.0
        return max(event.end for event in self.events)
