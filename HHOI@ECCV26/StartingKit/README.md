# UDIVA-HHOI Starting Kit

This starting kit provides:

- Codabench scoring packages for Recognition (Tracks 1 & 2), Anticipation (Tracks 3 & 4), and Causal (Track 5)
- Parser utilities for generating template/reference/example JSON files
- Visualization tools for timeline auditing with video and subtitles

The challenge defines five tracks across three tasks:

- Recognition (Tracks 1 & 2)
- Anticipation (Tracks 3 & 4)
- Causal (Track 5)

---

## Repository Structure

```
StartingKit/
├── readme.md
├── codabench/
│   ├── anticipation/
│   │   ├── template.json
│   │   ├── input/
│   │   │   ├── ref/
│   │   │   └── res/
│   │   ├── output/
│   │   └── scoring_program/
│   │       └── evaluation/
│   ├── recognition/
│   │   ├── template.json
│   │   ├── input/
│   │   │   ├── ref/
│   │   │   └── res/
│   │   ├── output/
│   │   └── scoring_program/
│   │       └── evaluation/
│   └── causal/
│       ├── template.json
│       ├── input/
│       │   ├── ref/
│       │   └── res/
│       ├── output/
│       └── scoring_program/
│           └── evaluation/
├── parser-lib/
│   ├── README.md
│   ├── configs/
│   ├── parser_lib/
│   ├── tests/
│   └── out/
└── visualization/
    ├── audit_timeline_viewer.py
    ├── README.md
    ├── audit_timeline_viewer/
    ├── data/
    └── preview/
```

---

## Codabench Scoring

The `codabench/` directory mirrors the structure and scripts that will be upload to Codabench for scoring participants. It contains one package per task:

- `recognition/` for Tracks 1 and 2
- `anticipation/` for Tracks 3 and 4
- `causal/` for Track 5

For each task package, the important components are:

- `template.json`: submission template with segments and empty event containers. During the evaluation/test stage, participants will receive a file like this and are expected to fill in the events to produce their example submission file like `input/res/example.json`.
- `input/ref/`: reference (ground truth) file used by the scoring script. It will not be publicly provided for the evaluation/test stage.
- `input/res/`: submission file(s) to evaluate, showing the expected format for predicted events. A sample `example.json` is provided.
- `output/`: destination where `scores.json` is written after scoring.
- `scoring_program/score.py`: Codabench entrypoint, which will run automatically after submission to the public leaderboard (during the evaluation/test stage). It loads one JSON from `input/ref/` and one JSON from `input/res/`, validates the submission, computes metrics, and writes results to `output/scores.json`.
- `scoring_program/evaluation/`: task-specific metric implementation and helper utilities (constants, validation/alignment helpers, and metric functions).

### Local Execution

Run the task scoring script from the repository root:

```bash
python codabench/<anticipation|recognition|causal>/scoring_program/score.py \
    codabench/<anticipation|recognition|causal>/input \
    codabench/<anticipation|recognition|causal>/output
```

---

## Tracks 1 & 2 - Recognition

Recognition evaluates verbal and non-verbal event detection with mean Average Precision (mAP).

### Run Locally

```bash
python codabench/recognition/scoring_program/score.py \
    codabench/recognition/input \
    codabench/recognition/output
```

### Example Output

```json
{ "mAP_verbal": 0.6320, "mAP_nonverbal": 0.5810, "mAP": 0.6065 }
```

### Wildcards

Wildcard values listed in `./src/evaluation/recognition/constants.py` (`WILDCARDS` dictionary) are meant to avoid penalising predictions for events that were ambiguous or uncertain at annotation time. These are:

| Attribute | Wildcard values |
|---|---|
| `utterance_type` | `unintelligible` |
| `highlevel_action` | `unintentional`, `unclear` |
| `lowlevel_action` | `unclear`, `none` |
| `target` | `unclear` |

More precisely, they appear in two different roles in the evaluation:

**Wildcards during the attribute matching for counting TP/FP/FNs.** Wildcard values are accepted as a match for any predicted attribute value during the attribute matching step prior to the AP calculation. Predictions are compared first to ground truth events not containing wildcard values. If unmatched, they are compared against ground truth events containing wildcard values. If matched to the latter, the prediction is silently discarded — counting neither as a true positive nor as a false positive. Only predictions that remain unmatched after this second pass are counted as false positives.

**Wildcards for discarding AP classes.** Both `utterance_type` and `highlevel_action` are attributes that define the set of classes from the point of view of the mAP metric (an AP is computed for each `utterance_type` and `highlevel_action` value, then averaged). If events are annotated with wildcard values for those two attributes, no AP is computed for them, and therefore they do not contribute to the corresponding mAP_verbal or mAP_nonverbal computations.

---

## Tracks 3 & 4 - Anticipation

Anticipation evaluates how well a model predicts upcoming events (verbal and non-verbal sequences).

### Run Locally

```bash
python codabench/anticipation/scoring_program/score.py \
    codabench/anticipation/input \
    codabench/anticipation/output
```

### Example Output

```json
{ "next_action": 1.0, "verbal_2s": 1.0, "nonverbal_2s": 1.0, "verbal_nonverbal_2s": 1.0 }
```

### Wildcards
See explanation for Tracks 1 & 2 above. Note that the metric will be computed differently, this will be updated shortly.

---

## Track 5 - Causal

Causal evaluates cause-effect predictions with:

- `accuracy`
- `temporal_accuracy`

### Run Locally

```bash
python codabench/causal/scoring_program/score.py \
    codabench/causal/input \
    codabench/causal/output
```

---

## Parser Utilities

The parser utilities are under `parser-lib/`.

- Source package: `parser-lib/parser_lib/`
- Configs: `parser-lib/configs/`
- Tests: `parser-lib/tests/`
- Example generated outputs: `parser-lib/out/random_delta_181182_001080/`

See `parser-lib/README.md` for usage details.

---

## Visualization

The starting kit includes an interactive **Timeline Viewer** for auditing ground truth annotations alongside video and subtitles. It supports multi-channel display of verbal and non-verbal events, synchronized video playback, and subtitle integration. The timeline viewer and related assets are under `visualization/`.

- CLI/package code: `visualization/audit_timeline_viewer/`
- Standalone launcher: `visualization/audit_timeline_viewer.py`
- Example data: `visualization/data/`

See `visualization/README.md` for usage details.

---

## License

This project is part of the UDIVA research initiative. 

Contact us at contextus-workshop [at] googlegroups.com or open a Github issue if you find any problems or have any questions.
