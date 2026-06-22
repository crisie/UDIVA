from __future__ import annotations

import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import pip._vendor.tomli as tomllib

import pytest

STARTING_KIT = Path(__file__).resolve().parents[2]
PARSER_LIB = STARTING_KIT / "parser-lib"
SCORING_PROGRAM = STARTING_KIT / "codabench" / "anticipation" / "scoring_program"
sys.path.insert(0, str(PARSER_LIB))
sys.path.insert(0, str(SCORING_PROGRAM))

from evaluation.evaluation_utils import validate_reference, validate_submission  # noqa: E402
from evaluation.metrics import substitution_cost  # noqa: E402
from parser_lib.formatting import (  # noqa: E402
    anticipation_event_payload,
    event_payload,
    normalize_value,
    target_value,
)
from parser_lib.models import Event, Segment  # noqa: E402
from parser_lib.pipeline import (  # noqa: E402
    build_anticipation_segments,
    format_anticipation_segments,
)


def make_event(
    event_id: str,
    participant: str,
    channel: str,
    target_filtered,
    target=None,
    start: float = 4.2,
    end: float = 4.4,
) -> Event:
    raw = {
        "id": event_id,
        "subject": participant,
        "target_filtered": target_filtered,
        "target": target if target is not None else target_filtered,
        "modifier": ["slow", "fast"],
    }
    if channel == "verbal":
        raw.update(
            {
                "act": "V",
                "utterance_type": "request",
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


def test_normalize_value_preserves_existing_first_and_join_modes():
    values = ["partial_model", "brick_3001"]
    assert normalize_value(values, {}) == "partial_model"
    assert normalize_value(
        values,
        {"list_mode": "join", "list_separator": ",", "list_sort": "asc"},
    ) == "brick_3001,partial_model"


def test_normalize_value_list_mode_returns_a_json_compatible_list():
    values = ["partial_model", "brick_3001"]
    assert normalize_value(
        values,
        {"list_mode": "list", "list_sort": "asc"},
    ) == ["brick_3001", "partial_model"]
    assert normalize_value(
        values,
        {"list_mode": "list", "list_sort": "desc"},
    ) == ["partial_model", "brick_3001"]


def test_target_value_preserves_multiple_filtered_targets_and_strips_ids():
    raw = {
        "target_filtered": ["brick_3001#2", "partial_model#1"],
        "target": ["fallback#1"],
    }
    assert target_value(
        raw,
        {"list_mode": "list", "list_sort": "asc", "strip_target_ids": True},
    ) == ["brick_3001", "partial_model"]


def test_target_value_collapses_singleton_and_falls_back_to_target():
    raw = {
        "target_filtered": [],
        "target": ["brick_3001#2"],
    }
    assert target_value(
        raw,
        {"list_mode": "list", "strip_target_ids": True},
    ) == "brick_3001"


def test_recognition_join_format_is_unchanged():
    event = make_event(
        "v1",
        "participant_a",
        "verbal",
        ["partial_model", "brick_3001"],
    )
    output_spec = {
        "list_mode": "join",
        "list_separator": ",",
        "list_sort": "asc",
        "strip_target_ids": True,
    }
    payload = event_payload(
        event,
        ["subject", "utterance_type", "target", "modifier"],
        output_spec,
    )
    assert payload == {
        "subject": "participant_a",
        "utterance_type": "request",
        "target": "brick_3001,partial_model",
        "modifier": "fast,slow",
    }


def test_reference_example_and_template_match_codabench_contract():
    segment = Segment("s_0001", 4.0, 6.0, "anticipation")
    event_a = make_event(
        "v1",
        "participant_a",
        "verbal",
        ["brick_3001", "partial_model"],
    )
    event_b = make_event(
        "nv1",
        "participant_b",
        "nonverbal",
        ["brick_3001"],
    )
    assignments = {
        "s_0001": {
            "participant_a": [event_a],
            "participant_b": [event_b],
        }
    }
    participants = ["participant_a", "participant_b"]

    reference_segments = format_anticipation_segments(
        [segment],
        assignments,
        participants,
        {
            "include_events": True,
            "include_hypotheses": False,
            "list_mode": "list",
            "list_sort": "asc",
            "strip_target_ids": True,
            "time_decimals": 3,
        },
        "reference",
    )
    example_segments = format_anticipation_segments(
        [segment],
        assignments,
        participants,
        {
            "include_events": True,
            "include_hypotheses": True,
            "list_mode": "first",
            "strip_target_ids": True,
            "time_decimals": 3,
        },
        "example",
    )
    template_segments = format_anticipation_segments(
        [segment],
        assignments,
        participants,
        {
            "include_events": False,
            "include_hypotheses": True,
            "time_decimals": 3,
        },
        "template",
    )

    reference = {"anticipation": {"video": reference_segments}}
    example = {"anticipation": {"video": example_segments}}
    template = {"anticipation": {"video": template_segments}}

    assert reference_segments["s_0001"]["participants"]["participant_a"]["events"] == [
        ["request", ["brick_3001", "partial_model"]]
    ]
    assert example_segments["s_0001"]["participants"]["participant_a"]["hypotheses"] == [
        {"events": [["request", "brick_3001"]]}
    ]
    assert template_segments["s_0001"]["participants"]["participant_a"]["hypotheses"] == [
        {"events": []}
    ]

    for payload in (reference_segments, example_segments, template_segments):
        assert payload["s_0001"]["t_b"] == 4.0
        assert payload["s_0001"]["t_e"] == 6.0

    validate_reference(reference)
    validate_submission(example)
    validate_submission(template)

    gt_event = reference_segments["s_0001"]["participants"]["participant_a"]["events"][0]
    assert substitution_cost(gt_event, ["request", "brick_3001"]) == pytest.approx(0.0)
    assert substitution_cost(gt_event, ["request", "partial_model"]) == pytest.approx(0.0)
    assert substitution_cost(gt_event, ["request", "leaflet"]) == pytest.approx(0.2)


def test_random_delta_anticipation_windows_start_at_four_seconds():
    timeline_event = make_event(
        "end",
        "participant_a",
        "verbal",
        ["brick_3001"],
        start=20.0,
        end=20.0,
    )
    segments = build_anticipation_segments(
        [timeline_event],
        {
            "name": "random_delta_window",
            "window_sec": 2.0,
            "d1_sec": 0.5,
            "d2_sec": 3.5,
            "seed": 2026,
            "start_sec": 2.0,
            "include_partial": False,
        },
        {
            "name": "prediction_window",
            "prediction_start": "segment_end",
            "prediction_offset_sec": 0.0,
            "prediction_window_sec": 2.0,
            "event_type": "all",
        },
        ["participant_a", "participant_b"],
        video_duration=20.0,
    )

    assert segments[0].start == pytest.approx(4.0)
    assert segments[0].end == pytest.approx(6.0)
    for previous, current in zip(segments, segments[1:]):
        assert 2.5 <= current.start - previous.start <= 5.5
        assert current.end - current.start == pytest.approx(2.0)


def test_official_anticipation_configs_use_output_specific_list_modes():
    configs = [
        PARSER_LIB / "configs" / "uniform_181182_001080_anticipation.toml",
        PARSER_LIB / "configs" / "random_delta_181182_001080_anticipation.toml",
    ]
    for path in configs:
        with path.open("rb") as handle:
            config = tomllib.load(handle)
        assert config["task"] == "anticipation"
        assert config["input"]["glob"] == "../data/spotting/*.json"
        assert config["defaults"]["selection"]["prediction_start"] == "segment_end"
        assert "segment_offset_sec" not in config["defaults"]["selection"]
        assert config["outputs"]["reference"]["list_mode"] == "list"
        assert config["outputs"]["example"]["list_mode"] == "first"
        assert config["outputs"]["template"]["include_events"] is False
