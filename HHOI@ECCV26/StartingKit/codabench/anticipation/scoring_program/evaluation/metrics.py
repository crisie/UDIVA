"""Structured Damerau-Levenshtein evaluation for UDIVA-HHOI Event Anticipation.

Events are submitted as ordered JSON lists:
    verbal:     [utterance_type, target]
    non-verbal: [highlevel_action, lowlevel_action, target]

Participant predictions contain exactly one target string. Hidden reference events may
contain either one target string or a list of acceptable target strings.

The best hypothesis is selected independently per participant
and per subtask.
"""

from evaluation.constants import (
    DELETION_COST,
    INSERTION_COST,
    NONVERBAL_EVENT_LENGTH,
    NONVERBAL_WEIGHTS,
    PARTICIPANTS,
    SUBTASKS,
    TRANSPOSITION_COST,
    VERBAL_EVENT_LENGTH,
    VERBAL_WEIGHTS,
    VERBAL_ATTRIBUTE_NAMES,
    NONVERBAL_ATTRIBUTE_NAMES,
    VERBAL_WILDCARDS,
    NONVERBAL_WILDCARDS,
)


def event_type(event):
    """Return ``verbal`` or ``nonverbal`` according to event tuple arity."""
    if len(event) == VERBAL_EVENT_LENGTH:
        return "verbal"
    if len(event) == NONVERBAL_EVENT_LENGTH:
        return "nonverbal"
    raise ValueError(f"Invalid event length {len(event)}: expected 2 or 3.")


def attribute_matches(gt_attr, pred_attr, attr_name, wildcards):
    """Return whether a prediction matches one ground-truth attribute.

    Targets may be represented in the hidden reference as a list of acceptable
    strings. Participant predictions remain single strings. A ground-truth
    wildcard, including one contained in an acceptable-target list, matches any
    prediction.
    """
    wildcard_values = wildcards.get(attr_name, [])
    if attr_name == "target" and isinstance(gt_attr, list):
        return any(value in wildcard_values for value in gt_attr) or pred_attr in gt_attr
    return gt_attr in wildcard_values or gt_attr == pred_attr


def substitution_cost(gt_event, pred_event):
    """Compute structured substitution cost in [0, 1] for a pair of events."""
    gt_type = event_type(gt_event)
    if gt_type != event_type(pred_event):
        return 1.0
    weights = VERBAL_WEIGHTS if gt_type == "verbal" else NONVERBAL_WEIGHTS
    wildcards = VERBAL_WILDCARDS if gt_type == "verbal" else NONVERBAL_WILDCARDS
    attribute_names = VERBAL_ATTRIBUTE_NAMES if gt_type == "verbal" else NONVERBAL_ATTRIBUTE_NAMES
    return sum(
        weight for weight, gt_attr, pred_attr, attr_name
        in zip(weights, gt_event, pred_event, attribute_names)
        if not attribute_matches(gt_attr, pred_attr, attr_name, wildcards)
    )


def events_match_with_wildcards(gt_event, pred_event):
    """Return True if events match after discounting GT wildcard attributes."""
    return (
        event_type(gt_event) == event_type(pred_event)
        and substitution_cost(gt_event, pred_event) == 0.0
    )


def structured_damerau_levenshtein(gt_sequence, pred_sequence):
    """Compute SDL with exact adjacent transposition as specified in the track."""
    m, n = len(gt_sequence), len(pred_sequence)
    dist = [[0.0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        dist[i][0] = i * DELETION_COST
    for j in range(1, n + 1):
        dist[0][j] = j * INSERTION_COST

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            candidates = [
                dist[i - 1][j] + DELETION_COST,
                dist[i][j - 1] + INSERTION_COST,
                dist[i - 1][j - 1] + substitution_cost(gt_sequence[i - 1], pred_sequence[j - 1]),
            ]
            if (
                i > 1 and j > 1
                and events_match_with_wildcards(gt_sequence[i - 1], pred_sequence[j - 2])
                and events_match_with_wildcards(gt_sequence[i - 2], pred_sequence[j - 1])
            ):
                candidates.append(dist[i - 2][j - 2] + TRANSPOSITION_COST)
            dist[i][j] = min(candidates)
    return dist[m][n]


def normalized_sdl_score(gt_sequence, pred_sequence):
    """Return normalized SDL score; two empty sequences receive score one."""
    denominator = max(len(gt_sequence), len(pred_sequence))
    if denominator == 0:
        return 1.0
    distance = structured_damerau_levenshtein(gt_sequence, pred_sequence)
    return max(0.0, min(1.0, 1.0 - distance / denominator))


def project_sequence(sequence, subtask):
    """Extract the sequence evaluated by one of the four anticipation subtasks."""
    if subtask not in SUBTASKS:
        raise ValueError(f"Unknown anticipation subtask: {subtask}")
    if subtask == "next_action":
        return sequence[:1]
    if subtask == "verbal_2s":
        return [event for event in sequence if event_type(event) == "verbal"]
    if subtask == "nonverbal_2s":
        return [event for event in sequence if event_type(event) == "nonverbal"]
    return list(sequence)


def best_of_k_score(gt_events, hypotheses, subtask):
    """Keep the best submitted alternative for a participant and subtask."""
    gt_projected = project_sequence(gt_events, subtask)
    return max(
        normalized_sdl_score(gt_projected, project_sequence(pred_events, subtask))
        for pred_events in hypotheses
    )


def evaluate_subtask(aligned_segments, subtask):
    """Average scores over both participants and all evaluation segments."""
    if not aligned_segments:
        return 0.0
    segment_scores = []
    for gt_segment, pred_segment in aligned_segments:
        scores = []
        for participant in PARTICIPANTS:
            gt_events = gt_segment["participants"][participant]["events"]
            hypotheses = [
                hypothesis["events"]
                for hypothesis in pred_segment["participants"][participant]["hypotheses"]
            ]
            scores.append(best_of_k_score(gt_events, hypotheses, subtask))
        segment_scores.append(sum(scores) / len(PARTICIPANTS))
    return sum(segment_scores) / len(segment_scores)


def evaluate_anticipation(aligned_segments):
    """Compute all four leaderboard score values."""
    return {subtask: evaluate_subtask(aligned_segments, subtask) for subtask in SUBTASKS}
