from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "parser-lib"))

from parser_lib.models import Event, Segment
from parser_lib.pipeline import run_config
from parser_lib.strategies import build_segments, select_events


class SelectionStrategyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.event = Event(
            event_id="evt",
            video_id="video",
            start=1.5,
            end=2.5,
            channel="verbal",
            raw={"id": "evt", "act": "V"},
        )
        self.segments = [
            Segment("s_001", 0.0, 2.0),
            Segment("s_002", 2.0, 4.0),
        ]

    def test_first_overlap_uses_first_matching_segment(self) -> None:
        assignments = select_events([self.event], self.segments, {"name": "first_overlap"})
        self.assertEqual([item.event.event_id for item in assignments["s_001"]], ["evt"])
        self.assertEqual(assignments["s_002"], [])

    def test_first_min_overlap_ratio_can_reject_partial_boundary_event(self) -> None:
        assignments = select_events(
            [self.event],
            self.segments,
            {"name": "first_min_overlap_ratio", "threshold": 0.70},
        )
        self.assertEqual(assignments["s_001"], [])
        self.assertEqual(assignments["s_002"], [])

    def test_every_overlap_assigns_to_all_matching_segments(self) -> None:
        assignments = select_events([self.event], self.segments, {"name": "every_overlap"})
        self.assertEqual([item.event.event_id for item in assignments["s_001"]], ["evt"])
        self.assertEqual([item.event.event_id for item in assignments["s_002"]], ["evt"])


class PipelineConfigTest(unittest.TestCase):
    def test_two_video_config_writes_groundtruth_and_submission(self) -> None:
        config = ROOT / "parser-lib" / "configs" / "two_videos.toml"
        with tempfile.TemporaryDirectory() as tmp_dir:
            paths = run_config(config, output_dir=tmp_dir)
            self.assertEqual(set(paths), {"groundtruth", "submission"})

            groundtruth = json.loads(paths["groundtruth"].read_text(encoding="utf-8"))
            submission = json.loads(paths["submission"].read_text(encoding="utf-8"))

        for payload in (groundtruth, submission):
            self.assertEqual(set(payload), {"verbal", "nonverbal"})
            self.assertIn("001080", payload["verbal"])
            self.assertIn("042084", payload["verbal"])
            self.assertIn("001080", payload["nonverbal"])
            self.assertIn("042084", payload["nonverbal"])

        gt_event = first_event(groundtruth, "verbal")
        submission_event = first_event(submission, "verbal")
        self.assertNotIn("score", gt_event)
        self.assertIn("score", submission_event)
        self.assertEqual(submission_event["score"], 1.0)

    def test_random_delta_zero_gap_matches_uniform_window_starts(self) -> None:
        event = Event(
            event_id="evt",
            video_id="video",
            start=0.0,
            end=6.0,
            channel="verbal",
            raw={"id": "evt", "act": "V"},
        )
        segments = build_segments(
            [event],
            {
                "name": "random_delta_window",
                "window_sec": 2.0,
                "d1_sec": 0.0,
                "d2_sec": 0.0,
                "seed": 2026,
            },
        )
        self.assertEqual([segment.start for segment in segments], [0.0, 2.0, 4.0])

    def test_random_delta_window_allows_negative_gap_overlap(self) -> None:
        event = Event(
            event_id="evt",
            video_id="video",
            start=0.0,
            end=5.0,
            channel="verbal",
            raw={"id": "evt", "act": "V"},
        )
        segments = build_segments(
            [event],
            {
                "name": "random_delta_window",
                "window_sec": 2.0,
                "d1_sec": -1.0,
                "d2_sec": -1.0,
                "include_partial": False,
            },
        )
        self.assertEqual([segment.start for segment in segments], [0.0, 1.0, 2.0, 3.0])


def first_event(payload: dict[str, object], channel: str) -> dict[str, object]:
    videos = payload[channel]
    assert isinstance(videos, dict)
    for segments in videos.values():
        assert isinstance(segments, dict)
        for segment in segments.values():
            assert isinstance(segment, dict)
            events = segment.get("events")
            assert isinstance(events, list)
            if events:
                return events[0]
    raise AssertionError(f"No events found for {channel}")


if __name__ == "__main__":
    unittest.main()
