"""Parsers for automatically recognizing and extracting different audit JSON formats."""

from __future__ import annotations

from typing import Any

from audit_timeline_viewer.core.labels import annotation_display_label, useful_annotation_value
from audit_timeline_viewer.core.values import as_float


class BaseParser:
    """Base class for audit parsers."""

    def parse_channels(
        self,
        video_audit: dict[str, Any],
        annotations: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        raise NotImplementedError


class LegacyParser(BaseParser):
    """Parser for the old multi-file format where annotations were separate."""

    def parse_channels(
        self,
        video_audit: dict[str, Any],
        annotations: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        raw_channels = video_audit.get("channels")
        if isinstance(raw_channels, dict):
            return {
                "verbal": self._normalize_channel("verbal", raw_channels.get("verbal", {}), annotations),
                "nonverbal": self._normalize_channel("nonverbal", raw_channels.get("nonverbal", {}), annotations),
            }
        else:
            return self._normalize_legacy_channels(video_audit, annotations)

    def _normalize_channel(
        self,
        channel_name: str,
        channel_audit: dict[str, Any],
        annotations: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        raw_segments = channel_audit.get("segments", []) if isinstance(channel_audit, dict) else []
        segments: list[dict[str, Any]] = []
        events: list[dict[str, Any]] = []

        for segment_index, raw_segment in enumerate(raw_segments):
            if not isinstance(raw_segment, dict):
                continue

            segment_id = str(raw_segment.get("segment", f"s_{segment_index + 1:03d}"))
            t_b = as_float(raw_segment.get("t_b"))
            t_e = as_float(raw_segment.get("t_e"))
            raw_events = raw_segment.get("events", []) or []

            event_ids: list[str] = []
            visible_values: list[float] = []
            loss_values: list[float] = []

            for event_index, raw_event in enumerate(raw_events):
                if not isinstance(raw_event, dict):
                    continue
                raw_id = str(raw_event.get("raw_id", f"event_{event_index + 1:03d}"))
                start = as_float(raw_event.get("start"))
                end = as_float(raw_event.get("end"), start)
                visible_ratio = as_float(raw_event.get("visible_ratio"), 0.0)
                visible_loss_ratio = as_float(
                    raw_event.get("visible_loss_ratio"),
                    max(0.0, 1.0 - visible_ratio),
                )
                annotation = annotations.get(raw_id, {})
                label, label_kind = annotation_display_label(raw_event, annotation, channel_name)
                event_ids.append(raw_id)
                visible_values.append(visible_ratio)
                loss_values.append(visible_loss_ratio)

                events.append(
                    {
                        "uid": f"{channel_name}:{segment_id}:{raw_id}:{event_index}",
                        "raw_id": raw_id,
                        "label": label,
                        "label_kind": label_kind,
                        "segment_id": segment_id,
                        "channel": channel_name,
                        "act": raw_event.get("act", "V" if channel_name == "verbal" else "NV"),
                        "subject": annotation.get("subject", ""),
                        "target": annotation.get("target", ""),
                        "modifier": annotation.get("modifier", ""),
                        "start": round(start, 3),
                        "end": round(end, 3),
                        "duration_sec": round(
                            as_float(raw_event.get("duration_sec"), max(0.0, end - start)),
                            3,
                        ),
                        "segment_t_b": round(t_b, 3),
                        "segment_t_e": round(t_e, 3),
                        "overlap_sec": round(as_float(raw_event.get("overlap_sec")), 3),
                        "max_visible_sec": round(as_float(raw_event.get("max_visible_sec")), 3),
                        "visible_ratio": round(visible_ratio, 3),
                        "visible_loss_sec": round(as_float(raw_event.get("visible_loss_sec")), 3),
                        "visible_loss_ratio": round(visible_loss_ratio, 3),
                        "actual_outside_sec": round(as_float(raw_event.get("actual_outside_sec")), 3),
                        "prefix_lost_sec": round(as_float(raw_event.get("prefix_lost_sec")), 3),
                        "suffix_lost_sec": round(as_float(raw_event.get("suffix_lost_sec")), 3),
                    }
                )

            segments.append(
                {
                    "uid": f"{channel_name}:{segment_id}",
                    "segment": segment_id,
                    "channel": channel_name,
                    "t_b": round(t_b, 3),
                    "t_e": round(t_e, 3),
                    "reason": raw_segment.get("reason", ""),
                    "raw_event_ids": event_ids,
                    "event_count": len(event_ids),
                    "min_visible_ratio": round(min(visible_values), 3) if visible_values else 1.0,
                    "max_visible_loss_ratio": round(max(loss_values), 3) if loss_values else 0.0,
                }
            )

        events.sort(key=lambda event: (event["start"], event["end"], event["raw_id"]))
        segments.sort(key=lambda segment: (segment["t_b"], segment["t_e"], segment["segment"]))
        return {
            "channel": channel_name,
            "segments": segments,
            "events": events,
            "num_segments": len(segments),
            "num_events": len(events),
        }

    def _normalize_legacy_channels(
        self,
        video_audit: dict[str, Any],
        annotations: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        raw_segments = video_audit.get("segments", []) if isinstance(video_audit, dict) else []
        channel_buckets: dict[str, dict[str, Any]] = {
            "verbal": {"segments": []},
            "nonverbal": {"segments": []},
        }

        for raw_segment in raw_segments:
            if not isinstance(raw_segment, dict):
                continue
            by_channel = {"verbal": [], "nonverbal": []}
            for raw_event in raw_segment.get("events", []) or []:
                channel = str(raw_event.get("channel", "")).lower()
                if channel not in by_channel:
                    channel = "verbal" if str(raw_event.get("act", "")).upper() == "V" else "nonverbal"
                by_channel[channel].append(raw_event)
            for channel, events in by_channel.items():
                if not events:
                    continue
                copied = dict(raw_segment)
                copied["events"] = events
                channel_buckets[channel]["segments"].append(copied)

        return {
            channel: self._normalize_channel(channel, audit, annotations)
            for channel, audit in channel_buckets.items()
        }


class LegosParser(BaseParser):
    """Parser for inline segment formats where events live inside each segment."""

    def parse_channels(
        self,
        video_audit: dict[str, Any],
        annotations: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        raw_channels = video_audit.get("channels", {})
        return {
            "verbal": self._parse_channel("verbal", raw_channels.get("verbal", [])),
            "nonverbal": self._parse_channel("nonverbal", raw_channels.get("nonverbal", [])),
        }

    def _parse_channel(self, channel_name: str, raw_segments: list[Any]) -> dict[str, Any]:
        segments: list[dict[str, Any]] = []
        events: list[dict[str, Any]] = []

        for segment_index, raw_segment in enumerate(raw_segments):
            if not isinstance(raw_segment, dict):
                continue

            segment_id = str(raw_segment.get("segment", f"s_{segment_index + 1:03d}"))
            t_b = as_float(raw_segment.get("t_b"))
            t_e = as_float(raw_segment.get("t_e"))
            raw_events = raw_segment.get("events", []) or []

            event_ids: list[str] = []

            for event_index, raw_event in enumerate(raw_events):
                if not isinstance(raw_event, dict):
                    continue

                raw_id = str(
                    raw_event.get("raw_id")
                    or raw_event.get("id")
                    or f"{segment_id}_event_{event_index + 1:03d}"
                )
                has_event_start = raw_event.get("start") is not None
                start = as_float(raw_event.get("start"), t_b)
                end = as_float(raw_event.get("end"), t_e if not has_event_start else start)
                end = max(start, end)

                # Map legos specific actions to standard ones for annotation_display_label
                annotation_proxy = dict(raw_event)
                if "highlevel_action" in raw_event:
                    annotation_proxy["high_level_action"] = raw_event["highlevel_action"]
                if "lowlevel_action" in raw_event:
                    annotation_proxy["low_level_action"] = raw_event["lowlevel_action"]

                label, label_kind = annotation_display_label(raw_event, annotation_proxy, channel_name)

                event_ids.append(raw_id)
                events.append(
                    {
                        "uid": f"{channel_name}:{segment_id}:{raw_id}:{event_index}",
                        "raw_id": raw_id,
                        "label": label,
                        "label_kind": label_kind,
                        "segment_id": segment_id,
                        "channel": channel_name,
                        "act": "V" if channel_name == "verbal" else "NV",
                        "subject": raw_event.get("subject", ""),
                        "target": raw_event.get("target", ""),
                        "modifier": raw_event.get("modifier", ""),
                        "start": round(start, 3),
                        "end": round(end, 3),
                        "duration_sec": round(max(0.0, end - start), 3),
                        "segment_t_b": round(t_b, 3),
                        "segment_t_e": round(t_e, 3),
                        "overlap_sec": 0.0,
                        "max_visible_sec": 0.0,
                        "visible_ratio": 1.0,
                        "visible_loss_sec": 0.0,
                        "visible_loss_ratio": 0.0,
                        "actual_outside_sec": 0.0,
                        "prefix_lost_sec": 0.0,
                        "suffix_lost_sec": 0.0,
                        "time_source": "event" if has_event_start or raw_event.get("end") is not None else "segment",
                    }
                )

            segments.append(
                {
                    "uid": f"{channel_name}:{segment_id}",
                    "segment": segment_id,
                    "channel": channel_name,
                    "t_b": round(t_b, 3),
                    "t_e": round(t_e, 3),
                    "reason": raw_segment.get("reason", ""),
                    "raw_event_ids": event_ids,
                    "event_count": len(event_ids),
                    "min_visible_ratio": 1.0,
                    "max_visible_loss_ratio": 0.0,
                }
            )

        events.sort(key=lambda event: (event["start"], event["end"], event["raw_id"]))
        segments.sort(key=lambda segment: (segment["t_b"], segment["t_e"], segment["segment"]))
        return {
            "channel": channel_name,
            "segments": segments,
            "events": events,
            "num_segments": len(segments),
            "num_events": len(events),
        }


def get_parser_for_audit(video_audit: dict[str, Any]) -> BaseParser:
    """Automatically recognize the format and return the appropriate parser."""
    raw_channels = video_audit.get("channels")

    # In the legos format, channels.verbal and channels.nonverbal are lists.
    # In the legacy format, they are dicts (containing a 'segments' key).
    if isinstance(raw_channels, dict):
        sample_channel = raw_channels.get("verbal") or raw_channels.get("nonverbal")
        if isinstance(sample_channel, list):
            return LegosParser()

    return LegacyParser()
