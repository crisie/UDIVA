# Parser Lib

Standalone parser builder for UDIVA spotting JSON files.

The parser is configured through TOML files. Two strategy families are used:

- **Segmentation strategies** determine the temporal windows.
- **Selection strategies** determine which annotations are assigned to each window.

The top-level `task` field selects the output schema:

- `task = "recognition"` produces the channel-based format used by Tracks 1 and 2.
- `task = "anticipation"` produces the participant-based format used by Tracks 3 and 4.

## Quick Start

Run commands from the repo root.

Recognition:

```bash
PYTHONPATH=parser-lib python3 -m parser_lib \
  --config parser-lib/configs/uniform_181182_001080_recognition.toml
```

Uniform anticipation:

```bash
PYTHONPATH=parser-lib python3 -m parser_lib \
  --config parser-lib/configs/uniform_181182_001080_anticipation.toml
```

Random-delta anticipation:

```bash
PYTHONPATH=parser-lib python3 -m parser_lib \
  --config parser-lib/configs/random_delta_181182_001080_anticipation.toml
```

Each config generates `reference.json`, `example.json`, and `template.json` in its configured output directory.

## Strategies

Segmentation strategies choose how time windows are created:

- `uniform`: split the video into back-to-back windows of the same length.
- `sliding_window`: create same-size windows at a fixed step size.
- `random_delta_window`: create same-size windows, then start the next window
  after a random gap in `[d1, d2]`; `0` behaves like uniform spacing and
  negative values make windows overlap.
- `event_window`: create one fixed-size window around each event.
- `density`: group events that are close together in time.
- `centroids`: build windows around the centers of event clusters.

Selection strategies choose how events are assigned to windows:

- `first_overlap`: assign an event to the first window that overlaps it.
- `first_min_overlap_ratio`: assign an event to the first window that contains
  at least a configured fraction of it, for example `threshold = 0.70`.
- `every_overlap`: assign an event to every window it overlaps.

## Config Shape

See `configs/random_delta_181182_001080.toml` for a runnable example.
Each output can override the default segmentation and selection strategy, which
makes it easy to produce `reference`, `example`, `template`, or other variants
from the same input files.

## Recognition outputs

Recognition outputs use separate `verbal` and `nonverbal` roots:

```json
{
  "verbal": {
    "001080": {
      "s_001": {
        "t_b": 0.0,
        "t_e": 2.0,
        "events": []
      }
    }
  },
  "nonverbal": {}
}
```

Here, `t_b` is the segment begin time, `t_e` is the segment end time, and
`events` is the list of events assigned to that segment. The usual difference is
the event payload inside `events`: `reference` contains the configured
annotation fields, while `example` can also include a `score` when
`include_score = true`. That `score` is a confidence score that participants are
expected to provide in their submission output.

The output fields and list formatting are controlled independently for every output block (i.e., reference, example, template). Existing recognition configs use:

```toml
list_mode = "join"
list_separator = ","
list_sort = "asc"
```

This converts a list-valued annotation into one joined string and remains unchanged across outputs.

For anticipation configs, `[defaults.segmentation]` still creates the observed
context windows. `[defaults.selection]` then controls the future target:

```toml
task = "anticipation"

[defaults.selection]
name = "prediction_window"    # or "next_events"
prediction_start = "segment_end"
prediction_offset_sec = 0.0
prediction_window_sec = 2.0
segment_offset_sec = 0.0
event_type = "all"            # "verbal", "nonverbal", or "all"
# n_events = 5                # used when name = "next_events"
```

`prediction_offset_sec` is the gap after the observation window before future
events are selected. For `prediction_window`, all events starting inside
`[prediction.t_b, prediction.t_e]` are grouped. For `next_events`, the next
`n_events` after `prediction.t_b` are selected per participant. `event_type`
filters the prediction target to verbal, nonverbal, or all events.

When `segment_offset_sec` is set, anticipation windows are laid out as compound
segments: observation window, prediction offset, prediction target, then segment
offset. A zero segment offset makes the next observation start exactly when the
previous prediction ends; negative values overlap the next observation with the
previous prediction; positive values leave a gap. Please note that, in the 
context of the challenge, the observation window would be considered from the 
start of the video until the start of the prediction window.

The reference format contains one ordered event sequence for each participant:

```json
{
  "anticipation": {
    "001080": {
      "s_0001": {
        "t_b": 4.0,
        "t_e": 6.0,
        "participants": {
          "participant_a": {
            "events": [["request", "brick_3001"]]
          },
          "participant_b": {
            "events": []
          }
        }
      }
    }
  }
}
```

Example and template outputs use (and must include) ranked hypotheses:

```json
{
  "participant_a": {
    "hypotheses": [
      {"events": []}
    ]
  }
}
```
### Multiple acceptable targets

Participants always predict one target string per event. Hidden reference annotations may contain multiple acceptable targets:

```json
["request", ["brick_3001", "partial_model"]]
```

The anticipation scorer considers either target correct. Configure output-specific list behavior as follows:

```toml
[outputs.reference]
list_mode = "list"
list_sort = "asc"
strip_target_ids = true

[outputs.example]
list_mode = "first"
strip_target_ids = true
```

The available list modes are:

- `first`: retain the first non-empty value;
- `join`: concatenate values into one string;
- `list`: preserve values as a JSON list.

When  `task = "anticipation"`, `list_mode = "list"` must be used for targets.

## Output alignment

`reference.json`, `example.json`, and `template.json` generated by one config must share the same video IDs, segment IDs, and `t_b`/`t_e` values. The CodaBench anticipation scorer requires prediction boundaries to match the reference exactly.
