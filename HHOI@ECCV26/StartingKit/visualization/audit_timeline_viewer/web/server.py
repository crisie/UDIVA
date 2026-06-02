"""HTTP serving for the browser UI and local video files."""

from __future__ import annotations

import json
import mimetypes
import re
import sys
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, ClassVar
from urllib.parse import unquote, urlparse

from audit_timeline_viewer.web.assets import load_index_html


@dataclass
class ServerState:
    data: dict[str, Any]
    video_map: dict[str, Path]


class AuditViewerHandler(BaseHTTPRequestHandler):
    """Request handler for viewer data, UI, and byte-range video streaming."""

    server_state: ClassVar[ServerState]

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format % args}", file=sys.stderr)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_text(load_index_html(), "text/html; charset=utf-8")
            return
        if parsed.path == "/data.json":
            payload = json.dumps(self.server_state.data, ensure_ascii=False).encode("utf-8")
            self.send_bytes(payload, "application/json; charset=utf-8")
            return
        if parsed.path.startswith("/video/"):
            video_id = unquote(parsed.path.split("/", 2)[2])
            self.send_video(video_id)
            return
        if parsed.path == "/favicon.ico":
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def send_text(self, text: str, content_type: str) -> None:
        self.send_bytes(text.encode("utf-8"), content_type)

    def send_bytes(self, payload: bytes, content_type: str) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def send_video(self, video_id: str) -> None:
        path = self.server_state.video_map.get(video_id)
        if path is None:
            self.send_error(HTTPStatus.NOT_FOUND, "Video not mapped")
            return
        if not path.exists():
            self.send_error(HTTPStatus.NOT_FOUND, f"Video file not found: {path}")
            return

        file_size = path.stat().st_size
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        range_header = self.headers.get("Range")
        start = 0
        end = file_size - 1
        status = HTTPStatus.OK

        if range_header:
            match = re.match(r"bytes=(\d*)-(\d*)", range_header)
            if match:
                if match.group(1):
                    start = int(match.group(1))
                if match.group(2):
                    end = int(match.group(2))
                status = HTTPStatus.PARTIAL_CONTENT

        start = max(0, min(start, file_size - 1))
        end = max(start, min(end, file_size - 1))
        length = end - start + 1

        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.end_headers()

        with path.open("rb") as handle:
            handle.seek(start)
            remaining = length
            while remaining > 0:
                chunk = handle.read(min(1024 * 512, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    break
                remaining -= len(chunk)


def make_handler(server_state: ServerState) -> type[AuditViewerHandler]:
    class BoundAuditViewerHandler(AuditViewerHandler):
        pass

    BoundAuditViewerHandler.server_state = server_state
    return BoundAuditViewerHandler
