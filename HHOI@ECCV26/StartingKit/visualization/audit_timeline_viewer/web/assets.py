"""Static asset loading for the web viewer."""

from __future__ import annotations

from pathlib import Path


def load_index_html() -> str:
    return (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")
