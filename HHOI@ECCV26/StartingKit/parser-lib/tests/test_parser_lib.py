from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "parser-lib"))

from parser_lib.models import Event, Segment
from parser_lib.pipeline import build_anticipation_segments, run_config, select_anticipation_events
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


class AnticipationSelectionTest(unittest.TestCase):
    def test_prediction_window_segment_offset_controls_next_observation_start(self) -> None:
        events = [
            anticipation_event("evt", 0.0, 0.1, "participant_a", "verbal"),
        ]
        segmentation = {
            "name": "uniform",
            "window_sec": 2.0,
            "start_sec": 0.0,
            "end_sec": 12.0,
            "include_partial": False,
        }
        participants = ["participant_a", "participant_b"]

        zero_offset = build_anticipation_segments(
            events,
            segmentation,
            {
                "name": "prediction_window",
                "prediction_offset_sec": 0.0,
                "prediction_window_sec": 2.0,
                "segment_offset_sec": 0.0,
            },
            participants,
            video_duration=12.0,
        )
        negative_offset = build_anticipation_segments(
            events,
            segmentation,
            {
                "name": "prediction_window",
                "prediction_offset_sec": 0.0,
                "prediction_window_sec": 2.0,
                "segment_offset_sec": -1.0,
            },
            participants,
            video_duration=12.0,
        )
        positive_offset = build_anticipation_segments(
            events,
            segmentation,
            {
                "name": "prediction_window",
                "prediction_offset_sec": 0.0,
                "prediction_window_sec": 2.0,
                "segment_offset_sec": 1.0,
            },
            participants,
            video_duration=12.0,
        )

        self.assertAlmostEqual(zero_offset[1].start - 2.0, zero_offset[0].end)
        self.assertLess(negative_offset[1].start - 2.0, negative_offset[0].end)
        self.assertGreater(positive_offset[1].start - 2.0, positive_offset[0].end)

    def test_next_events_uses_prediction_offset_n_and_event_type(self) -> None:
        events = [
            anticipation_event("a_before", 2.5, 2.6, "participant_a", "verbal"),
            anticipation_event("a_v1", 3.0, 3.1, "participant_a", "verbal"),
            anticipation_event("a_nv", 3.2, 3.3, "participant_a", "nonverbal"),
            anticipation_event("a_v2", 4.0, 4.1, "participant_a", "verbal"),
            anticipation_event("a_v3", 5.0, 5.1, "participant_a", "verbal"),
            anticipation_event("b_v1", 3.4, 3.5, "participant_b", "verbal"),
        ]
        selection = {
            "name": "next_events",
            "prediction_offset_sec": 1.0,
            "n_events": 2,
            "event_type": "verbal",
            "segment_offset_sec": 0.0,
        }
        segments = build_anticipation_segments(
            events,
            {
                "name": "uniform",
                "window_sec": 2.0,
                "start_sec": 0.0,
                "end_sec": 8.0,
                "include_partial": False,
            },
            selection,
            ["participant_a", "participant_b"],
            video_duration=8.0,
        )
        assignments = select_anticipation_events(
            events,
            [segments[0]],
            ["participant_a", "participant_b"],
            selection,
        )

        self.assertAlmostEqual(segments[0].start, 3.0)
        self.assertAlmostEqual(segments[0].end, 4.1)
        self.assertEqual(
            [event.event_id for event in assignments["s_0001"]["participant_a"]],
            ["a_v1", "a_v2"],
        )
        self.assertEqual(
            [event.event_id for event in assignments["s_0001"]["participant_b"]],
            ["b_v1"],
        )


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
            self.assertIn("181182", payload["verbal"])
            self.assertIn("001080", payload["nonverbal"])
            self.assertIn("181182", payload["nonverbal"])

        gt_event = first_event(groundtruth, "verbal")
        submission_event = first_event(submission, "verbal")
        self.assertNotIn("score", gt_event)
        self.assertIn("score", submission_event)
        self.assertEqual(submission_event["score"], 1.0)

    def test_anticipation_config_writes_participant_hypothesis_outputs(self) -> None:
        config = ROOT / "parser-lib" / "configs" / "random_delta_181182_001080_anticipation.toml"
        with tempfile.TemporaryDirectory() as tmp_dir:
            paths = run_config(config, output_dir=tmp_dir)
            self.assertEqual(set(paths), {"reference", "example", "template"})

            reference = json.loads(paths["reference"].read_text(encoding="utf-8"))
            example = json.loads(paths["example"].read_text(encoding="utf-8"))
            template = json.loads(paths["template"].read_text(encoding="utf-8"))

        self.assertEqual(set(reference), {"anticipation"})
        self.assertIn("001080", reference["anticipation"])
        first_segment_id = sorted(reference["anticipation"]["001080"])[0]

        reference_segment = reference["anticipation"]["001080"][first_segment_id]
        example_segment = example["anticipation"]["001080"][first_segment_id]
        template_segment = template["anticipation"]["001080"][first_segment_id]

        self.assertEqual(
            set(reference_segment["participants"]),
            {"participant_a", "participant_b"},
        )
        self.assertIn("events", reference_segment["participants"]["participant_a"])
        self.assertIn("hypotheses", example_segment["participants"]["participant_a"])
        self.assertIn("hypotheses", template_segment["participants"]["participant_a"])
        self.assertEqual(set(reference_segment["participants"]["participant_a"]), {"events"})
        self.assertEqual(set(example_segment["participants"]["participant_a"]), {"hypotheses"})
        self.assertEqual(
            template_segment["participants"]["participant_a"]["hypotheses"],
            [{"events": []}],
        )
        event = first_anticipation_event(reference)
        self.assertIn(len(event), {2, 3})

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


def first_anticipation_event(payload: dict[str, object]) -> list[object]:
    videos = payload["anticipation"]
    assert isinstance(videos, dict)
    for segments in videos.values():
        assert isinstance(segments, dict)
        for segment in segments.values():
            assert isinstance(segment, dict)
            participants = segment.get("participants")
            assert isinstance(participants, dict)
            for participant in participants.values():
                assert isinstance(participant, dict)
                events = participant.get("events")
                assert isinstance(events, list)
                if events:
                    event = events[0]
                    assert isinstance(event, list)
                    return event
    raise AssertionError("No anticipation events found")


def anticipation_event(
    event_id: str,
    start: float,
    end: float,
    participant: str,
    channel: str,
) -> Event:
    raw = {
        "id": event_id,
        "subject": participant,
        "target_filtered": ["brick_3001"],
        "target": ["brick_3001#1"],
    }
    if channel == "verbal":
        raw.update(
            {
                "act": "V",
                "utterance_type": "suggest",
                "high_level_action": "none",
                "low_level_action": "none",
            }
        )
    else:
        raw.update(
            {
                "act": "NV",
                "utterance_type": "none",
                "high_level_action": "select",
                "low_level_action": "pick_up",
            }
        )
    return Event(
        event_id=event_id,
        video_id="video",
        start=start,
        end=end,
        channel=channel,
        raw=raw,
    )


if __name__ == "__main__":
    unittest.main()
