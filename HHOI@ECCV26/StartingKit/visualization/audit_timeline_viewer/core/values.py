"""Small value-conversion helpers shared across the viewer."""

from __future__ import annotations

import re
from typing import Any


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def natural_video_sort_key(video: dict[str, Any]) -> tuple[str, str]:
    video_id = str(video.get("video_id", ""))
    match = re.match(r"(\d+)", video_id)
    if match:
        return (match.group(1).zfill(12), video_id)
    return (video_id, video_id)
