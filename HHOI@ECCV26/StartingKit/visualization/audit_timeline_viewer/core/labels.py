"""Label selection rules for audit events and subtitle payloads."""

from __future__ import annotations

from typing import Any


def annotation_display_label(
    raw_event: dict[str, Any],
    annotation: dict[str, Any],
    channel_name: str,
) -> tuple[str, str]:
    act = str(annotation.get("act", raw_event.get("act", ""))).upper()
    if not act:
        act = "V" if channel_name == "verbal" else "NV"

    if act == "V":
        label = useful_annotation_value(annotation.get("utterance_type"))
        if label is not None:
            return label, "utterance_type"
        label = useful_annotation_value(annotation.get("high_level_action"))
        if label is not None:
            return label, "high_level_action"
    else:
        label = useful_annotation_value(annotation.get("high_level_action"))
        if label is not None:
            return label, "high_level_action"
        label = useful_annotation_value(annotation.get("utterance_type"))
        if label is not None:
            return label, "utterance_type"

    label = useful_annotation_value(annotation.get("low_level_action"))
    if label is not None:
        return label, "low_level_action"

    return str(raw_event.get("raw_id", annotation.get("id", "event"))), "raw_id"


def useful_annotation_value(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text or text.lower() == "none":
        return None
    return text
