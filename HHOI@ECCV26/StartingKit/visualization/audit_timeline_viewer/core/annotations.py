"""Annotation metadata loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from audit_timeline_viewer.core.io import load_json_with_encoding_fallback
from audit_timeline_viewer.core.labels import useful_annotation_value


def load_annotation_map(path: Path) -> dict[str, dict[str, Any]]:
    raw = load_json_with_encoding_fallback(path)
    if isinstance(raw, dict):
        raw_annotations = raw.get("annotations", [])
    elif isinstance(raw, list):
        raw_annotations = raw
    else:
        raw_annotations = []

    annotations: dict[str, dict[str, Any]] = {}
    for annotation in raw_annotations:
        if not isinstance(annotation, dict):
            continue
        raw_id = useful_annotation_value(annotation.get("id"))
        if raw_id is not None:
            annotations[raw_id] = annotation
    return annotations
