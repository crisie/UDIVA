"""Output event formatting driven by config fields."""

from __future__ import annotations

from typing import Any

from .models import Event

DEFAULT_FIELDS = {
    "verbal": ["subject", "utterance_type", "target", "modifier"],
    "nonverbal": ["subject", "highlevel_action", "lowlevel_action", "target", "modifier"],
}

FIELD_ALIASES = {
    "highlevel_action": ("high_level_action", "highlevel_action"),
    "lowlevel_action": ("low_level_action", "lowlevel_action"),
    "utterance_type": ("utterance_type",),
    "subject": ("subject",),
    "modifier": ("modifier",),
}


def fields_for_channel(output_spec: dict[str, Any], channel: str) -> list[str]:
    raw_fields = output_spec.get("fields", {})
    if isinstance(raw_fields, dict):
        fields = raw_fields.get(channel)
        if isinstance(fields, list) and fields:
            return [str(field) for field in fields]
    return list(DEFAULT_FIELDS.get(channel, []))


def event_payload(event: Event, fields: list[str], output_spec: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for field in fields:
        payload[field] = field_value(event, field, output_spec)

    if bool(output_spec.get("include_score", False)):
        score_field = str(output_spec.get("score_field", "score"))
        score_default = float(output_spec.get("score_default", 1.0))
        score_value = event.raw.get(score_field, event.raw.get("score", score_default))
        payload[score_field] = rounded_score(
            score_value,
            int(output_spec.get("score_decimals", 3)),
            score_default,
        )
    return payload


def field_value(event: Event, field: str, output_spec: dict[str, Any]) -> Any:
    if field == "target":
        return target_value(event.raw, output_spec)

    aliases = FIELD_ALIASES.get(field, (field,))
    for key in aliases:
        if key in event.raw:
            return normalize_value(event.raw.get(key), output_spec)
    return "none"


def target_value(raw: dict[str, Any], output_spec: dict[str, Any]) -> str:
    value = raw.get("target_filtered")
    if _is_empty_or_none(value):
        value = raw.get("target")
    normalized = normalize_value(value, output_spec)
    if isinstance(normalized, str) and bool(output_spec.get("strip_target_ids", True)):
        return normalized.split("#", 1)[0]
    return str(normalized)


def normalize_value(value: Any, output_spec: dict[str, Any]) -> Any:
    if value is None:
        return "none"

    if isinstance(value, list):
        values = [item for item in value if not _is_empty_or_none(item)]
        if not values:
            return "none"
        list_mode = str(output_spec.get("list_mode", "first")).lower()
        if list_mode == "join":
            separator = str(output_spec.get("list_separator", "|"))
            return separator.join(str(item) for item in values)
        return normalize_value(values[0], output_spec)

    if isinstance(value, str):
        value = value.strip()
        return value if value else "none"

    return value


def rounded_score(value: Any, decimals: int, default: float = 1.0) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = default
    return round(score, decimals)


def _is_empty_or_none(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "none", "null"}
    if isinstance(value, list):
        return all(_is_empty_or_none(item) for item in value)
    return False
