"""SRT subtitle parsing for the viewer side panel."""

from __future__ import annotations

import json
import re
from json import JSONDecodeError
from pathlib import Path
from typing import Any

from audit_timeline_viewer.core.io import load_text_with_encoding_fallback
from audit_timeline_viewer.core.labels import annotation_display_label, useful_annotation_value


def load_srt_subtitles(path: Path) -> list[dict[str, Any]]:
    text = load_text_with_encoding_fallback(path)
    blocks = re.split(r"\n\s*\n", text.strip())
    subtitles: list[dict[str, Any]] = []

    for fallback_index, block in enumerate(blocks, start=1):
        lines = [line.rstrip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue

        cue_index = fallback_index
        time_line_index = 0
        if len(lines) > 1 and re.fullmatch(r"\d+", lines[0].strip()):
            cue_index = int(lines[0].strip())
            time_line_index = 1

        if time_line_index >= len(lines):
            continue
        match = re.match(
            r"(.+?)\s*-->\s*(.+?)(?:\s+.*)?$",
            lines[time_line_index].strip(),
        )
        if not match:
            continue

        start = parse_srt_time(match.group(1))
        end = parse_srt_time(match.group(2))
        body = "\n".join(lines[time_line_index + 1:]).strip()
        payload: dict[str, Any] = {}
        if body.startswith("{") and body.endswith("}"):
            try:
                parsed = json.loads(body)
                if isinstance(parsed, dict):
                    payload = parsed
            except JSONDecodeError:
                payload = {}

        subtitles.append(build_subtitle_entry(cue_index, start, end, body, payload))

    subtitles.sort(key=lambda subtitle: (subtitle["start"], subtitle["end"], subtitle["index"]))
    return subtitles


def parse_srt_time(value: str) -> float:
    match = re.match(r"(\d+):(\d{2}):(\d{2})[,.](\d{1,3})", value.strip())
    if not match:
        return 0.0
    hours, minutes, seconds, millis = match.groups()
    return (
        int(hours) * 3600
        + int(minutes) * 60
        + int(seconds)
        + int(millis.ljust(3, "0")[:3]) / 1000
    )


def build_subtitle_entry(
    index: int,
    start: float,
    end: float,
    body: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    if payload:
        label, label_kind = annotation_display_label(payload, payload, "")
        subject = useful_annotation_value(payload.get("subject")) or ""
        act = useful_annotation_value(payload.get("act")) or ""
        target = useful_annotation_value(payload.get("target")) or ""
        low_level = useful_annotation_value(payload.get("low_level_action")) or ""
        other = useful_annotation_value(payload.get("other"))
        why_started = useful_annotation_value(payload.get("why_started"))
        text = other or why_started or " ".join(part for part in (low_level, target) if part) or label
        meta_parts = [part for part in (subject, act, label) if part]
        return {
            "index": index,
            "id": useful_annotation_value(payload.get("id")) or str(index),
            "start": round(start, 3),
            "end": round(end, 3),
            "label": label,
            "label_kind": label_kind,
            "subject": subject,
            "act": act,
            "target": target,
            "text": text,
            "meta": " · ".join(meta_parts),
        }

    return {
        "index": index,
        "id": str(index),
        "start": round(start, 3),
        "end": round(end, 3),
        "label": f"subtitle {index}",
        "label_kind": "subtitle",
        "subject": "",
        "act": "",
        "target": "",
        "text": body,
        "meta": "",
    }
