"""
evaluation_utils.py – UDIVA-HHOI Causal QA (Task 5)

Shared helpers used by the Codabench scoring program (codabench_scoring.py)
and by any local evaluation entry points.
"""

import json
import sys
from numbers import Real

from evaluation.constants import (
    CAUSE_REQUIRED_FIELDS,
    GT_REQUIRED_FIELDS,
    PRED_REQUIRED_FIELDS,
    ROOT_KEY
)


def load_json(path):
    """
    Load and parse a JSON file from disk.

    Args:
        path (str): Path to the JSON file.

    Returns:
        dict | list: Parsed JSON content.

    Exits:
        Calls sys.exit() with an error message if the file is not found
        or cannot be parsed as valid JSON.
    """
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        sys.exit(f"[ERROR] File not found: {path}")
    except json.JSONDecodeError as e:
        sys.exit(f"[ERROR] Could not parse {path}: {e}")


def collect_records_aligned(sub_data, ref_data):
    """
    Align predictions with ground truth using (video_id, effect_id) keys.

    Iterates over all effects present in the ground truth, retrieving the
    matching prediction from the submission. Effects missing from the
    submission are evaluated as empty predictions; effects predicted but absent
    from the GT are dropped. Both cases emit a warning.

    Args:
        sub_data (dict): Submission, shaped as
                         `{ROOT_KEY: {video_id: {effect_id: pred_record}}}`.
        ref_data (dict): Ground truth with the same shape.

    Returns:
        tuple[list[dict], list[dict]]:
            - preds: Predictions in GT iteration order.
            - gts:   Ground truth records in the same order.
    """
    sub_root = sub_data.get(ROOT_KEY, {})
    ref_root = ref_data.get(ROOT_KEY, {})

    preds, gts = [], []
    missing_in_sub = 0

    for vid, effects in ref_root.items():
        sub_vid = sub_root.get(vid, {})
        for eid, gt_rec in effects.items():
            pred_rec = sub_vid.get(eid)
            if pred_rec is None:
                missing_in_sub += 1
                pred_rec = {
                    "predicted_cause_timestamp": None,
                    "predicted_option": "",
                }
            preds.append(pred_rec)
            gts.append(gt_rec)

    if missing_in_sub:
        print(f"[WARN] {missing_in_sub} effect(s) in GT but missing from predictions; scored as empty.")

    extra = 0
    for vid, effects in sub_root.items():
        ref_vid = ref_root.get(vid, {})
        for eid in effects:
            if eid not in ref_vid:
                extra += 1
    if extra:
        print(f"[WARN] {extra} effect(s) in predictions but not in GT — skipped.")

    return preds, gts


def validate_submission(sub_data):
    """
    Perform basic structural validation on a submission to catch silent errors
    early.

    Verifies the top-level shape (`{ROOT_KEY: {video_id: {effect_id: record}}}`)
    and that every leaf record carries `predicted_cause_timestamp` (number or
    null) and `predicted_option` (string). Empty / null values are allowed —
    the scorer treats them as zero credit — but missing fields and wrong types
    cause an immediate exit.

    Args:
        sub_data (dict): Parsed submission.

    Exits:
        Calls sys.exit() with an error message identifying the offending record
        if validation fails.
    """
    if not isinstance(sub_data, dict) or ROOT_KEY not in sub_data:
        sys.exit(f"[ERROR] Submission must be a JSON object with a top-level '{ROOT_KEY}' key.")

    for vid, effects in sub_data[ROOT_KEY].items():
        if not isinstance(effects, dict):
            sys.exit(f"[ERROR] Submission '{ROOT_KEY}.{vid}' must be an object keyed by effect_id.")

        for eid, rec in effects.items():
            if not isinstance(rec, dict):
                sys.exit(f"[ERROR] Submission record ({vid}, {eid}) must be an object.")

            for k in PRED_REQUIRED_FIELDS:
                if k not in rec:
                    sys.exit(f"[ERROR] Missing '{k}' in prediction ({vid}, {eid}).")

            ts = rec["predicted_cause_timestamp"]
            if ts is not None and not isinstance(ts, Real):
                sys.exit(f"[ERROR] 'predicted_cause_timestamp' in ({vid}, {eid}) must be a number or null.")

            opt = rec["predicted_option"]
            if not isinstance(opt, str):
                sys.exit(f"[ERROR] 'predicted_option' in ({vid}, {eid}) must be a string.")


def validate_reference(ref_data):
    """
    Perform basic structural validation on the reference (ground truth).

    Verifies the top-level shape and that every leaf record carries a
    non-empty `causes` list whose entries each include `option`, `t_b`, and
    `t_e`.

    Args:
        ref_data (dict): Parsed ground truth.

    Exits:
        Calls sys.exit() with an error message identifying the offending record
        if validation fails.
    """
    if not isinstance(ref_data, dict) or ROOT_KEY not in ref_data:
        sys.exit(f"[ERROR] Reference must be a JSON object with a top-level '{ROOT_KEY}' key.")

    for vid, effects in ref_data[ROOT_KEY].items():
        if not isinstance(effects, dict):
            sys.exit(f"[ERROR] Reference '{ROOT_KEY}.{vid}' must be an object keyed by effect_id.")

        for eid, rec in effects.items():
            if not isinstance(rec, dict):
                sys.exit(f"[ERROR] Reference record ({vid}, {eid}) must be an object.")

            for k in GT_REQUIRED_FIELDS:
                if k not in rec:
                    sys.exit(f"[ERROR] Missing '{k}' in GT record ({vid}, {eid}).")

            causes = rec["causes"]
            if not isinstance(causes, list) or len(causes) == 0:
                sys.exit(f"[ERROR] 'causes' must be a non-empty list in GT record ({vid}, {eid}).")

            for c in causes:
                if not isinstance(c, dict):
                    sys.exit(f"[ERROR] Each cause in GT record ({vid}, {eid}) must be an object.")
                for k in CAUSE_REQUIRED_FIELDS:
                    if k not in c:
                        sys.exit(f"[ERROR] Missing '{k}' in cause of GT record ({vid}, {eid}).")
                if not isinstance(c["t_b"], Real) or not isinstance(c["t_e"], Real):
                    sys.exit(f"[ERROR] Cause windows in ({vid}, {eid}) must have numeric 't_b' and 't_e'.")
