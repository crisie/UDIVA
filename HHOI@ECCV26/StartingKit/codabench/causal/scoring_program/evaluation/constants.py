# Top-level key under which causal records are nested.
ROOT_KEY = "causal"

# Fields required in each prediction record (leaf of the submission tree).
PRED_REQUIRED_FIELDS = ["predicted_cause_timestamp", "predicted_option"]

# Fields required from each GT record by the metric functions.
# (GT records also carry `effect` and `options` -- those are not needed for scoring.)
GT_REQUIRED_FIELDS = ["causes"]

# Fields required in each cause-window dict nested inside GT records.
CAUSE_REQUIRED_FIELDS = ["option", "t_b", "t_e"]
