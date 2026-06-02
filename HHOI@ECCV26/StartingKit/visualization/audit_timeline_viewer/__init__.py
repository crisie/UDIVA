"""Interactive audit timeline viewer package."""

from audit_timeline_viewer.cli import ViewerApplication, ViewerConfig, ViewerDataBuilder, ViewerDataset, main, parse_args
from audit_timeline_viewer.core.annotations import load_annotation_map
from audit_timeline_viewer.core.io import load_json_with_encoding_fallback, load_text_with_encoding_fallback
from audit_timeline_viewer.core.labels import annotation_display_label, useful_annotation_value
from audit_timeline_viewer.core.normalization import AuditNormalizer, normalize_audit
from audit_timeline_viewer.core.values import as_float, natural_video_sort_key
from audit_timeline_viewer.discovery.companions import (
    companion_search_roots,
    companion_stems,
    discover_annotation_maps,
    discover_subtitle_maps,
    resolve_companion_file,
)
from audit_timeline_viewer.discovery.videos import (
    VIDEO_SUFFIXES,
    discover_video_map,
    extract_video_ids,
    first_video_match,
    resolve_video_file,
)
from audit_timeline_viewer.subtitles.srt import build_subtitle_entry, load_srt_subtitles, parse_srt_time
from audit_timeline_viewer.web.server import ServerState, make_handler

__all__ = [
    "AuditNormalizer",
    "ServerState",
    "VIDEO_SUFFIXES",
    "ViewerApplication",
    "ViewerConfig",
    "ViewerDataBuilder",
    "ViewerDataset",
    "annotation_display_label",
    "as_float",
    "build_subtitle_entry",
    "companion_search_roots",
    "companion_stems",
    "discover_annotation_maps",
    "discover_subtitle_maps",
    "discover_video_map",
    "extract_video_ids",
    "first_video_match",
    "load_annotation_map",
    "load_json_with_encoding_fallback",
    "load_srt_subtitles",
    "load_text_with_encoding_fallback",
    "main",
    "make_handler",
    "natural_video_sort_key",
    "normalize_audit",
    "parse_args",
    "parse_srt_time",
    "resolve_companion_file",
    "resolve_video_file",
    "useful_annotation_value",
]
