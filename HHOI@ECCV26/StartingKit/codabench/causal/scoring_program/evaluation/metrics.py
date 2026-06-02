"""
metrics.py – UDIVA-HHOI Causal QA (Task 5)

Implements the two per-question metrics used to score the Causal track.

Each prediction names a single option letter and a single cause timestamp.
Each GT record lists one or more cause windows; every cause carries its own
option letter and an inclusive [t_b, t_e] interval. Both metrics use the
"any-overlap" rule — full credit is awarded as soon as the prediction matches
*any* of the listed causes.

    1. Multiple-Choice Accuracy
        Acc_i = 1[ predicted_option_i ∈ { c.option for c in causes_i } ]

    2. Temporal Localisation Accuracy
        TA_i  = 1[ ∃ c ∈ causes_i : c.t_b ≤ predicted_cause_timestamp_i ≤ c.t_e ]

Both metrics are macro-averaged over all aligned (prediction, GT) pairs.
Predictions that leave `predicted_option` empty or `predicted_cause_timestamp`
null score 0.0 on the corresponding metric.
"""


# ---------------------------------------------------------------------------
# Per-question scoring helpers
# ---------------------------------------------------------------------------

def accuracy_score(predicted_option, gt_causes):
    """
    Binary multiple-choice accuracy for one question.

    Full credit is awarded if the predicted option letter matches the `option`
    field of at least one GT cause. Comparison is exact — labels must agree on
    case and whitespace.

    Args:
        predicted_option (str):        Single predicted option letter, or "" if
                                       no prediction was made.
        gt_causes        (list[dict]): GT cause records, each carrying an
                                       `option` key.

    Returns:
        float: 1.0 if `predicted_option` matches any cause's option, else 0.0.
    """
    if not predicted_option:
        return 0.0
    return 1.0 if any(c["option"] == predicted_option for c in gt_causes) else 0.0


def temporal_accuracy(predicted_cause_timestamp, gt_causes):
    """
    Hard temporal-localisation accuracy for one question.

    Returns 1.0 if the predicted timestamp falls inside the [t_b, t_e] window
    of any GT cause, 0.0 otherwise. Window boundaries are inclusive. A null
    prediction scores 0.0.

    Args:
        predicted_cause_timestamp (float | None): Predicted cause timestamp in
                                                  seconds, or None if no
                                                  prediction was made.
        gt_causes                 (list[dict]):   GT cause records, each
                                                  carrying `t_b` and `t_e`.

    Returns:
        float: 1.0 if the timestamp lies inside any GT window, else 0.0.
    """
    if predicted_cause_timestamp is None:
        return 0.0
    for c in gt_causes:
        if c["t_b"] <= predicted_cause_timestamp <= c["t_e"]:
            return 1.0
    return 0.0


# ---------------------------------------------------------------------------
# Aggregated metrics
# ---------------------------------------------------------------------------

def evaluate_accuracy(pred_records, gt_records):
    """
    Mean multiple-choice accuracy across all aligned questions.

    `pred_records` and `gt_records` must be aligned position-by-position
    (typically via `collect_records_aligned`).

    Args:
        pred_records (list[dict]): Predictions with a `predicted_option` field.
        gt_records   (list[dict]): GT records with a `causes` list.

    Returns:
        float: Mean accuracy in [0, 1]. Returns 0.0 if no aligned records.
    """
    if not pred_records:
        return 0.0

    scores = [
        accuracy_score(p["predicted_option"], g["causes"])
        for p, g in zip(pred_records, gt_records)
    ]
    return sum(scores) / len(scores)


def evaluate_temporal_accuracy(pred_records, gt_records):
    """
    Mean temporal-localisation accuracy across all aligned questions.

    `pred_records` and `gt_records` must be aligned position-by-position.

    Args:
        pred_records (list[dict]): Predictions with a `predicted_cause_timestamp`.
        gt_records   (list[dict]): GT records with a `causes` list.

    Returns:
        float: Mean temporal accuracy in [0, 1]. Returns 0.0 if no records.
    """
    if not pred_records:
        return 0.0

    scores = [
        temporal_accuracy(p["predicted_cause_timestamp"], g["causes"])
        for p, g in zip(pred_records, gt_records)
    ]
    return sum(scores) / len(scores)
