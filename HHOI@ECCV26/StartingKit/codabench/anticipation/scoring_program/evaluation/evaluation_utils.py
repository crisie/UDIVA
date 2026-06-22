"""Validation and alignment utilities for UDIVA-HHOI Event Anticipation."""

import json
import sys
from numbers import Real

from evaluation.constants import MAX_HYPOTHESES, PARTICIPANTS, ROOT_KEY


def load_json(path):
    """Load a JSON file or exit with a participant-facing error."""
    try:
        with open(path) as file:
            return json.load(file)
    except FileNotFoundError:
        sys.exit(f"[ERROR] File not found: {path}")
    except json.JSONDecodeError as error:
        sys.exit(f"[ERROR] Could not parse {path}: {error}")


def _validate_event_shape(event, context):
    if not isinstance(event, list) or len(event) not in (2, 3):
        sys.exit(
            f"[ERROR] Event in {context} must be a list with 2 elements "
            "(verbal) or 3 elements (non-verbal)."
        )


def _validate_reference_event(event, context):
    """Validate a hidden reference event.

    Every non-target attribute must be a string. The final target attribute may
    be either one string or a non-empty list of strings representing alternative
    acceptable targets.
    """
    _validate_event_shape(event, context)
    if not all(isinstance(attribute, str) for attribute in event[:-1]):
        sys.exit(f"[ERROR] Every non-target event attribute in {context} must be a string.")

    target = event[-1]
    valid_target = isinstance(target, str) or (
        isinstance(target, list)
        and len(target) > 0
        and all(isinstance(value, str) for value in target)
    )
    if not valid_target:
        sys.exit(
            f"[ERROR] Target in {context} must be a string or a non-empty list of strings."
        )


def _validate_submission_event(event, context):
    """Validate a participant event; predictions must contain one target string."""
    _validate_event_shape(event, context)
    if not all(isinstance(attribute, str) for attribute in event):
        sys.exit(f"[ERROR] Every submitted event attribute in {context} must be a string.")


def _validate_segment_metadata(segment, context):
    for key in ("t_b", "t_e"):
        if key not in segment or not isinstance(segment[key], Real):
            sys.exit(f"[ERROR] Segment {context} must contain numeric '{key}'.")


def validate_reference(reference):
    """Validate the hidden reference structure before scoring."""
    if not isinstance(reference, dict) or ROOT_KEY not in reference:
        sys.exit(f"[ERROR] Reference must contain top-level '{ROOT_KEY}'.")
    for video_id, segments in reference[ROOT_KEY].items():
        if not isinstance(segments, dict):
            sys.exit(f"[ERROR] Reference '{video_id}' must contain segments.")
        for segment_id, segment in segments.items():
            context = f"reference ({video_id}, {segment_id})"
            if not isinstance(segment, dict):
                sys.exit(f"[ERROR] Invalid {context}.")
            _validate_segment_metadata(segment, context)
            participants = segment.get("participants")
            if not isinstance(participants, dict):
                sys.exit(f"[ERROR] Missing 'participants' in {context}.")
            for participant in PARTICIPANTS:
                record = participants.get(participant)
                if not isinstance(record, dict) or not isinstance(record.get("events"), list):
                    sys.exit(f"[ERROR] Missing event list for {participant} in {context}.")
                for event in record["events"]:
                    _validate_reference_event(event, f"{context}, {participant}")


def validate_submission(submission):
    """Validate a submission.

    A submission may omit entire videos or segments; omitted segments are scored
    as empty predictions. Any submitted segment must contain both participants,
    each providing one to five alternative ordered sequences.
    """
    if not isinstance(submission, dict) or ROOT_KEY not in submission:
        sys.exit(f"[ERROR] Submission must contain top-level '{ROOT_KEY}'.")
    for video_id, segments in submission[ROOT_KEY].items():
        if not isinstance(segments, dict):
            sys.exit(f"[ERROR] Submission '{video_id}' must contain segments.")
        for segment_id, segment in segments.items():
            context = f"submission ({video_id}, {segment_id})"
            if not isinstance(segment, dict):
                sys.exit(f"[ERROR] Invalid {context}.")
            _validate_segment_metadata(segment, context)
            participants = segment.get("participants")
            if not isinstance(participants, dict):
                sys.exit(f"[ERROR] Missing 'participants' in {context}.")
            for participant in PARTICIPANTS:
                record = participants.get(participant)
                if not isinstance(record, dict):
                    sys.exit(f"[ERROR] Missing {participant} in {context}.")
                hypotheses = record.get("hypotheses")
                if not isinstance(hypotheses, list) or not 1 <= len(hypotheses) <= MAX_HYPOTHESES:
                    sys.exit(
                        f"[ERROR] {participant} in {context} must submit between "
                        f"1 and {MAX_HYPOTHESES} hypotheses."
                    )
                for index, hypothesis in enumerate(hypotheses):
                    if not isinstance(hypothesis, dict) or not isinstance(hypothesis.get("events"), list):
                        sys.exit(f"[ERROR] Invalid hypothesis {index} for {participant} in {context}.")
                    for event in hypothesis["events"]:
                        _validate_submission_event(event, f"{context}, {participant}, hypothesis {index}")


def _empty_prediction_segment(gt_segment):
    return {
        "t_b": gt_segment["t_b"],
        "t_e": gt_segment["t_e"],
        "participants": {
            participant: {"hypotheses": [{"events": []}]}
            for participant in PARTICIPANTS
        },
    }


def collect_segments_aligned(submission, reference):
    """Align by (video_id, segment_id), defaulting omitted predictions to empty.

    This prevents participants from avoiding penalties by omitting difficult
    segments. Unknown submitted videos or segments are rejected.
    """
    sub_root = submission[ROOT_KEY]
    ref_root = reference[ROOT_KEY]
    aligned = []
    for video_id, segments in ref_root.items():
        pred_segments = sub_root.get(video_id, {})
        for segment_id, gt_segment in segments.items():
            pred_segment = pred_segments.get(segment_id)
            if pred_segment is None:
                pred_segment = _empty_prediction_segment(gt_segment)
            elif (pred_segment["t_b"] != gt_segment["t_b"]
                  or pred_segment["t_e"] != gt_segment["t_e"]):
                sys.exit(
                    f"[ERROR] Time boundaries do not match the template for "
                    f"({video_id}, {segment_id})."
                )
            aligned.append((gt_segment, pred_segment))

    for video_id, segments in sub_root.items():
        if video_id not in ref_root:
            sys.exit(f"[ERROR] Unknown video_id in submission: {video_id}.")
        for segment_id in segments:
            if segment_id not in ref_root[video_id]:
                sys.exit(f"[ERROR] Unknown segment_id in submission: ({video_id}, {segment_id}).")
    return aligned
