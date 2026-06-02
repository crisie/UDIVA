"""
score.py - UDIVA-HHOI Causal QA (Task 5)
Codabench Scoring Program

Evaluation script for the Causal track on Codabench.

- Expects the Codabench directory layout: /app/input/ref (reference) and
  /app/input/res (submission)
- Writes scores to /app/output/scores.json
"""

import glob
import json
import os
import sys

import evaluation.metrics as metrics
from evaluation.evaluation_utils import (
    collect_records_aligned,
    load_json,
    validate_reference,
    validate_submission,
)


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

    Reads input and output directory paths from command-line arguments
    following the Codabench convention, locates the reference and submission
    JSON files within those directories, validates them, aligns records by
    (video_id, effect_id), computes both metrics, and writes the results to
    scores.json in the output directory.

    Command-line Args:
        argv[1] (str): Input directory, expected to contain ref/ (reference)
                       and res/ (submission) subdirectories.
        argv[2] (str): Output directory where scores.json will be written.

    Exits:
        Calls sys.exit() if arguments are missing, no JSON files are found in
        the expected directories, or no records can be aligned.

    Note:
        The keys written to scores.json must match the column keys defined in
        competition.yaml on Codabench.
    """
    if len(sys.argv) < 3:
        sys.exit("[ERROR] Missing arguments. Usage: python score.py <input_dir> <output_dir>")

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    reference_dir = os.path.join(input_dir, 'ref')
    prediction_dir = os.path.join(input_dir, 'res')

    ref_file = get_json_file(reference_dir)
    if not ref_file:
        sys.exit(f"[ERROR] No reference JSON found in {reference_dir}")

    sub_file = get_json_file(prediction_dir)
    if not sub_file:
        sys.exit(f"[ERROR] No submission JSON found in {prediction_dir}")

    ref_data = load_json(ref_file)
    sub_data = load_json(sub_file)

    validate_submission(sub_data)
    validate_reference(ref_data)

    preds, gts = collect_records_aligned(sub_data, ref_data)

    if not preds:
        sys.exit("[ERROR] No records could be aligned between submission and reference.")

    scores = {
        "accuracy": round(metrics.evaluate_accuracy(preds, gts), 4),
        "temporal_accuracy": round(metrics.evaluate_temporal_accuracy(preds, gts), 4),
    }

    # Keys in `scores` must match the column keys defined in competition.yaml
    score_file = os.path.join(output_dir, 'scores.json')

    with open(score_file, 'w') as f:
        json.dump(scores, f)

    print("Evaluation complete. Scores:")
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
