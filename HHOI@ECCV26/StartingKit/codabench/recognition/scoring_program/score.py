"""
score.py – UDIVA-HHOI Event Recognition (Task 1 & 2)
Codabench Scoring Program

Evaluation script for multimodal event recognition.

- Expects Codabench directory structure: /app/input/ref (reference) and /app/input/res (submission)
- Writes scores to /app/output/scores.json
"""

import os
import sys
import json
import glob

import evaluation.metrics as metrics
import evaluation.constants as constants
from evaluation.evaluation_utils import load_json, collect_segments_aligned, validate_submission


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_json_file(directory):
    """
    Find the first JSON file in a given directory.

    Args:
        directory (str): Path to the directory to search.

    Returns:
        str | None: Path to the first JSON file found, or None if the directory
                    contains no JSON files.
    """
    files = glob.glob(os.path.join(directory, '*.json'))
    return files[0] if files else None


# ---------------------------------------------------------------------------
# Main (Codabench Format)
# ---------------------------------------------------------------------------

def main():
    """
    Entry point for the Codabench scoring program.

    Reads input and output directory paths from command-line arguments following
    the Codabench convention, locates the reference and submission JSON files
    within those directories, validates the submission, computes mAP for each
    available category (verbal and/or non-verbal), and writes the results to
    scores.json in the output directory.

    Command-line Args:
        argv[1] (str): Input directory, expected to contain ref/ (reference) and
                       res/ (submission) subdirectories.
        argv[2] (str): Output directory where scores.json will be written.

    Exits:
        Calls sys.exit() if arguments are missing, no JSON files are found in
        the expected directories, or no valid categories are found in the reference.

    Note:
        The keys written to scores.json must match the column keys defined in
        competition.yaml on Codabench.
    """
    if len(sys.argv) < 3:
        sys.exit("[ERROR] Missing arguments. Usage: python score.py <input_dir> <output_dir>")

    # 1. Parse Codabench standard directories
    input_dir  = sys.argv[1]
    output_dir = sys.argv[2]

    reference_dir  = os.path.join(input_dir, 'ref')
    prediction_dir = os.path.join(input_dir, 'res')

    # 2. Locate JSON files dynamically
    ref_file = get_json_file(reference_dir)
    if not ref_file:
        sys.exit(f"[ERROR] No reference JSON found in {reference_dir}")

    sub_file = get_json_file(prediction_dir)
    if not sub_file:
        sys.exit(f"[ERROR] No submission JSON found in {prediction_dir}")

    ref_data = load_json(ref_file)
    sub_data = load_json(sub_file)

    validate_submission(sub_data)

    scores = {}

    # -----------------------------------------------------------------------
    # VERBAL
    # -----------------------------------------------------------------------
    if "verbal" in ref_data:
        preds_v, gts_v = collect_segments_aligned(sub_data, ref_data, "verbal")

        mAP_v = metrics.evaluate_mAP(
            preds_v,
            gts_v,
            class_key="utterance_type",
            classes=constants.UTTERANCE_TYPES,
            attribute_keys=["subject", "target", "modifier"],
            class_wildcards=constants.WILDCARDS["utterance_type"],
            attribute_wildcards=constants.WILDCARDS
        )

        scores["mAP_verbal"] = round(mAP_v, 4)

    # -----------------------------------------------------------------------
    # NON-VERBAL
    # -----------------------------------------------------------------------
    if "nonverbal" in ref_data:
        preds_n, gts_n = collect_segments_aligned(sub_data, ref_data, "nonverbal")

        mAP_n = metrics.evaluate_mAP(
            preds_n,
            gts_n,
            class_key="highlevel_action",
            classes=constants.HIGHLEVEL_ACTIONS,
            attribute_keys=["subject", "lowlevel_action", "target", "modifier"],
            class_wildcards=constants.WILDCARDS["highlevel_action"],
            attribute_wildcards=constants.WILDCARDS
        )

        scores["mAP_nonverbal"] = round(mAP_n, 4)

    if not scores:
        sys.exit("[ERROR] No valid categories found in submission.")

    scores["mAP"] = round(sum(scores.values()) / len(scores), 4)

    # -----------------------------------------------------------------------
    # Codabench Output
    # -----------------------------------------------------------------------
    # Keys in `scores` must match the column keys defined in competition.yaml
    score_file = os.path.join(output_dir, 'scores.json')

    with open(score_file, 'w') as f:
        json.dump(scores, f)

    print("Evaluation complete. Scores:")
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()