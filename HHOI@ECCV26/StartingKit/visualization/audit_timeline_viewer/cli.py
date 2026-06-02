"""Command-line orchestration for the audit timeline viewer."""

from __future__ import annotations

import argparse
import webbrowser
from dataclasses import dataclass
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any, Sequence

from audit_timeline_viewer.core.io import load_json_with_encoding_fallback
from audit_timeline_viewer.core.normalization import AuditNormalizer
from audit_timeline_viewer.discovery.companions import CompanionDiscoveryService
from audit_timeline_viewer.discovery.videos import VideoDiscoveryService
from audit_timeline_viewer.web.server import ServerState, make_handler


@dataclass(frozen=True)
class ViewerConfig:
    input: Path
    host: str = "127.0.0.1"
    port: int = 8765
    video_file: Path | None = None
    video_root: Path | None = None
    subtitle_file: Path | None = None
    subtitle_root: Path | None = None
    annotation_file: Path | None = None
    open_browser: bool = False

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> "ViewerConfig":
        return cls(
            input=args.input,
            host=args.host,
            port=args.port,
            video_file=args.video_file,
            video_root=args.video_root,
            subtitle_file=args.subtitle_file,
            subtitle_root=args.subtitle_root,
            annotation_file=args.annotation_file,
            open_browser=args.open,
        )


@dataclass
class ViewerDataset:
    data: dict[str, Any]
    video_map: dict[str, Path]
    annotation_sources: dict[str, Path]
    subtitle_sources: dict[str, Path]


class ViewerDataBuilder:
    """Load files and compose the normalized dataset consumed by the web UI."""

    def __init__(
        self,
        video_discovery: VideoDiscoveryService | None = None,
        companion_discovery: CompanionDiscoveryService | None = None,
    ) -> None:
        self.video_discovery = video_discovery or VideoDiscoveryService()
        self.companion_discovery = companion_discovery or CompanionDiscoveryService()

    def build(self, config: ViewerConfig) -> ViewerDataset:
        raw_audit = load_json_with_encoding_fallback(config.input)
        video_map = self.video_discovery.discover(raw_audit, config.video_file, config.video_root)
        annotation_maps, annotation_sources = self.companion_discovery.discover_annotation_maps(
            raw_audit,
            config.input,
            config.video_file,
            video_map,
            config.annotation_file,
        )
        subtitle_maps, subtitle_sources = self.companion_discovery.discover_subtitle_maps(
            raw_audit,
            config.input,
            config.video_file,
            video_map,
            config.subtitle_file,
            config.subtitle_root,
        )
        data = AuditNormalizer(video_map, annotation_maps, subtitle_maps).normalize(raw_audit)
        return ViewerDataset(
            data=data,
            video_map=video_map,
            annotation_sources=annotation_sources,
            subtitle_sources=subtitle_sources,
        )


class ViewerApplication:
    """Own the top-level application workflow without embedding parsing details."""

    def __init__(
        self,
        config: ViewerConfig,
        data_builder: ViewerDataBuilder | None = None,
    ) -> None:
        self.config = config
        self.data_builder = data_builder or ViewerDataBuilder()

    def run(self) -> None:
        dataset = self.data_builder.build(self.config)
        state = ServerState(data=dataset.data, video_map=dataset.video_map)
        handler = make_handler(state)
        server = ThreadingHTTPServer((self.config.host, self.config.port), handler)
        url = f"http://{self.config.host}:{self.config.port}"

        print(f"Serving {self.config.input} at {url}")
        self._print_sources("Mapped videos:", dataset.video_map)
        self._print_sources("Mapped annotation metadata:", dataset.annotation_sources)
        self._print_sources("Mapped subtitles:", dataset.subtitle_sources)

        if self.config.open_browser:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nStopping server")

    @staticmethod
    def _print_sources(title: str, sources: dict[str, Path]) -> None:
        if not sources:
            return
        print(title)
        for video_id, path in sorted(sources.items()):
            print(f"  {video_id}: {path}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve an interactive timeline for groundtruth.audit.json.")
    parser.add_argument("--input", required=True, type=Path, help="Path to groundtruth.audit.json.")
    parser.add_argument("--host", default="127.0.0.1", help="Local server host.")
    parser.add_argument("--port", default=8765, type=int, help="Local server port.")
    parser.add_argument("--video-file", type=Path, help="Optional video file for single-video audits.")
    parser.add_argument("--video-root", type=Path, help="Optional directory to scan for videos named with the video_id.")
    parser.add_argument("--subtitle-file", type=Path, help="Optional SRT subtitle file for single-video audits.")
    parser.add_argument("--subtitle-root", type=Path, help="Optional directory to scan for SRT subtitles named with the video_id.")
    parser.add_argument("--annotation-file", type=Path, help="Optional source annotation JSON used to label events.")
    parser.add_argument("--open", action="store_true", help="Open the viewer in the default browser.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    ViewerApplication(ViewerConfig.from_namespace(args)).run()
