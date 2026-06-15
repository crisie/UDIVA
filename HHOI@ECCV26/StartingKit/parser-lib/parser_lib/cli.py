"""Command-line interface for parser-lib."""

from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import available_anticipation_selection_strategies, run_config
from .strategies import available_segmentation_strategies, available_selection_strategies


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build parsed spotting outputs from a strategy config.")
    parser.add_argument("--config", type=Path, help="TOML, JSON, or YAML config file.")
    parser.add_argument("--output-dir", type=Path, help="Directory for generated output files.")
    parser.add_argument("--video-id", action="append", help="Video id to parse; may be repeated.")
    parser.add_argument("--limit", type=int, help="Limit number of input videos after filtering.")
    parser.add_argument(
        "--list-strategies",
        action="store_true",
        help="Print registered segmentation and selection strategies.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.list_strategies:
        print("Segmentation strategies:")
        for name in available_segmentation_strategies():
            print(f"  - {name}")
        print("Selection strategies:")
        for name in available_selection_strategies():
            print(f"  - {name}")
        print("Anticipation selection strategies:")
        for name in available_anticipation_selection_strategies():
            print(f"  - {name}")
        return 0

    if args.config is None:
        parser.error("--config is required unless --list-strategies is used")

    paths = run_config(
        args.config,
        output_dir=args.output_dir,
        video_ids=args.video_id,
        limit=args.limit,
    )
    for output_name, path in paths.items():
        print(f"Wrote {output_name}: {path}")
    return 0
