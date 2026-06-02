"""Encoding-safe file loading helpers."""

from __future__ import annotations

import json
from json import JSONDecodeError
from pathlib import Path
from typing import Any


TEXT_ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin-1")


def load_text_with_encoding_fallback(path: Path) -> str:
    raw_bytes = path.read_bytes()
    errors: list[str] = []

    for encoding in TEXT_ENCODINGS:
        try:
            return raw_bytes.decode(encoding)
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: decode failed at byte {exc.start}")

    raise ValueError(
        f"Could not load text file {path}. Tried {', '.join(TEXT_ENCODINGS)}. "
        f"Details: {'; '.join(errors)}"
    )


def load_json_with_encoding_fallback(path: Path) -> Any:
    raw_bytes = path.read_bytes()
    errors: list[str] = []

    for encoding in TEXT_ENCODINGS:
        try:
            text = raw_bytes.decode(encoding)
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: decode failed at byte {exc.start}")
            continue

        try:
            return json.loads(text)
        except JSONDecodeError as exc:
            errors.append(f"{encoding}: JSON parse failed at line {exc.lineno}, column {exc.colno}")

    raise ValueError(
        f"Could not load JSON file {path}. Tried {', '.join(TEXT_ENCODINGS)}. "
        f"Details: {'; '.join(errors)}"
    )
