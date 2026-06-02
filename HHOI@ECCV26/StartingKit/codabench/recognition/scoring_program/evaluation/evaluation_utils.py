"""
evaluation_utils.py – UDIVA-HHOI Event Recognition (Task 1 & 2)

Shared helpers used by both the local evaluation script (score.py)
and the Codabench scoring program (score.py).
"""

import json
import sys


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


def collect_segments_aligned(pred_dict, gt_dict, category):
    """
    Align predictions with ground truth segments using (video_id, segment_id) keys.

    Iterates over all segments present in the ground truth for the given category
    and retrieves the corresponding predictions, defaulting to an empty list if
    no predictions exist for a given segment.

    Args:
        pred_dict (dict): Full submission dictionary, keyed by category → video_id → segment_id.
        gt_dict   (dict): Full ground truth dictionary, keyed by category → video_id → segment_id.
        category  (str): Category to extract, either "verbal" or "nonverbal".

    Returns:
        tuple[List[List[dict]], List[List[dict]]]:
            - preds: Predicted events per segment, in GT iteration order.
            - gts:   Ground truth events per segment, in GT iteration order.
    """
    preds, gts = [], []

    gt_cat = gt_dict.get(category, {})
    pred_cat = pred_dict.get(category, {})

    for vid, segs in gt_cat.items():
        for seg_id, seg_data in segs.items():
            gts.append(seg_data.get("events", []))

            seg_data_pred = pred_cat.get(vid, {}).get(seg_id, {})
            pred_events = (seg_data_pred or {}).get("events", [])

            preds.append(pred_events)

    return preds, gts


def validate_submission(sub_data):
    """
    Perform basic structural validation on a submission to catch silent errors early.

    Checks that every predicted event across all categories, videos, and segments
    contains a "score" field. Exits immediately if any event is missing one.

    Args:
        sub_data (dict): Parsed submission dictionary.

    Exits:
        Calls sys.exit() with an error message identifying the offending event
        if a "score" field is missing from any prediction.
    """
    for cat in ["verbal", "nonverbal"]:
        if cat not in sub_data:
            continue

        for vid in sub_data[cat]:
            for seg_id in sub_data[cat][vid]:
                events = sub_data[cat][vid][seg_id].get("events", [])

                for e in events:
                    if "score" not in e:
                        sys.exit(f"[ERROR] Missing 'score' in prediction: {e}")