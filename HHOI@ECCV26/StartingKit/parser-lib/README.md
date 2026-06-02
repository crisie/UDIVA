# Parser Lib

Standalone parser builder for `data/spotting/*.json` files.

You configure the parser with two kinds of strategies:

- **Segmentation strategies** decide how each video is split into time windows.
- **Selection strategies** decide which events are assigned to each window.

The output shape is controlled by a TOML config file.

## Quick Start

From the repo root:

```bash
PYTHONPATH=parser-lib python3 -m parser_lib \
  --config parser-lib/configs/random_delta_181182_001080.toml
```

That writes:

- `parser-lib/out/random_delta_181182_001080/groundtruth.json`
- `parser-lib/out/random_delta_181182_001080/submission.json`

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

> As discussed in UDIVA HHOI meeting 2026-05-21, we will likely just use `random_delta_window` and `every_overlap` for the Starting Kit submission, but the other strategies are available for experimentation and future use.

## Config Shape

See `configs/random_delta_181182_001080.toml` for a runnable example. Each output can override
the default segmentation and selection strategy, which makes it easy to produce
`groundtruth`, `submission`, or other variants from the same input files.

Both outputs use the same outer JSON structure:

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
`events` is the list of events assigned to that segment. The outer structure is
the same for `groundtruth` and `submission`; the usual difference is the event
payload inside `events`: `groundtruth` contains the configured annotation
fields, while `submission` can also include a `score` when `include_score = true`.
That `score` is a confidence score that participants are expected to provide in
their submission output.
