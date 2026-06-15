"""score.py – UDIVA-HHOI Event Anticipation (Tracks 3 & 4).

Expects the Codabench directory layout:
    <input_dir>/ref/*.json
    <input_dir>/res/*.json
and writes:
    <output_dir>/scores.json
"""

import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from evaluation.metrics import evaluate_anticipation
from evaluation.evaluation_utils import (
    collect_segments_aligned,
    load_json,
    validate_reference,
    validate_submission,
)


def get_json_file(directory):
    """Return the first JSON file in a Codabench input directory."""
    files = glob.glob(os.path.join(directory, "*.json"))
    return files[0] if files else None


def main():
    if len(sys.argv) < 3:
        sys.exit("[ERROR] Missing arguments. Usage: python codabench_scoring.py <input_dir> <output_dir>")

    input_dir, output_dir = sys.argv[1], sys.argv[2]
    reference_file = get_json_file(os.path.join(input_dir, "ref"))
    submission_file = get_json_file(os.path.join(input_dir, "res"))

    if not reference_file:
        sys.exit(f"[ERROR] No reference JSON found in {os.path.join(input_dir, 'ref')}")
    if not submission_file:
        sys.exit(f"[ERROR] No submission JSON found in {os.path.join(input_dir, 'res')}")

    reference = load_json(reference_file)
    submission = load_json(submission_file)

    validate_reference(reference)
    validate_submission(submission)
    aligned_segments = collect_segments_aligned(submission, reference)

    scores = {
        key: round(value, 4)
        for key, value in evaluate_anticipation(aligned_segments).items()
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "scores.json"), "w") as file:
        json.dump(scores, file)

    print("Evaluation complete. Scores:")
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()