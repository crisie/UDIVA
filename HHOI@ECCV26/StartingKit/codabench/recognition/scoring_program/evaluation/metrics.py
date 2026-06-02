import numpy as np

# ---------------------------------------------------------------------------
# Average Precision (all-point interpolation, unchanged)
# ---------------------------------------------------------------------------

def _compute_ap(tp_flags, confidences, num_gt):
    """
    Compute Average Precision (AP) using all-point interpolation.

    Sorts predictions by descending confidence, computes a precision-recall
    curve, makes it monotonically decreasing, and integrates the area under
    it using the all-point interpolation method.

    Args:
        tp_flags    (np.ndarray): Boolean array where True indicates a True Positive.
                                  Must be parallel to `confidences`.
        confidences (np.ndarray): Confidence scores for each prediction.
                                  Must be parallel to `tp_flags`.
        num_gt      (int):        Total number of ground truth instances for this class,
                                  used as the recall denominator.

    Returns:
        float: Average Precision in [0, 1]. Returns 0.0 if `num_gt` is 0
               or `tp_flags` is empty.
    """
    if num_gt == 0 or len(tp_flags) == 0:
        return 0.0

    # Sort by confidence (descending)
    sort_idx = np.argsort(-confidences)
    tp_flags = tp_flags[sort_idx]

    # Cumulative sums
    tp_cumsum = np.cumsum(tp_flags)
    fp_cumsum = np.cumsum(~tp_flags)

    precision = tp_cumsum / (tp_cumsum + fp_cumsum)
    recall = tp_cumsum / num_gt

    # All-point interpolation
    m_rec = np.concatenate(([0.0], recall, [1.0]))
    m_pre = np.concatenate(([0.0], precision, [0.0]))

    # Make precision monotonically decreasing
    for i in range(len(m_pre) - 2, -1, -1):
        m_pre[i] = max(m_pre[i], m_pre[i + 1])

    # Area under curve
    i = np.where(m_rec[1:] != m_rec[:-1])[0]
    ap = np.sum((m_rec[i + 1] - m_rec[i]) * m_pre[i + 1])

    return ap


# ---------------------------------------------------------------------------
# mAP evaluation (dict-based, segment-aware)
# ---------------------------------------------------------------------------

def _find_match(pred, candidates, matched, attribute_keys, attribute_wildcards):
    """
    Greedily match a prediction to the first unmatched candidate.

    Iterates over `candidates` in order and returns True as soon as a candidate
    is found that (a) has not already been matched and (b) agrees with `pred` on
    all `attribute_keys`. Agreement on a key means either an exact value match or
    the candidate's value being listed as a wildcard for that key in
    `attribute_wildcards`.

    The index of the matched candidate is added to `matched` in-place, ensuring
    one-to-one assignment across successive calls within the same segment.

    Args:
        pred               (dict):      The predicted event to match.
        candidates         (List[dict]): Ordered list of ground truth events to
                                         match against.
        matched            (set):        Set of already-consumed candidate indices.
                                         Modified in-place when a match is found.
        attribute_keys     (List[str]):  Event attributes that must agree for a
                                         match to be valid.
        attribute_wildcards (dict):      Mapping from attribute name to a list of
                                         wildcard values that are accepted as a
                                         match for any prediction value on that key.

    Returns:
        bool: True if a match was found (and `matched` updated), False otherwise.
    """
    def attributes_match(p, g):
        return all(
            p.get(k) == g.get(k) or g.get(k) in attribute_wildcards.get(k, [])
            for k in attribute_keys
        )

    for i, g in enumerate(candidates):
        if i not in matched and attributes_match(pred, g):
            matched.add(i)
            return True

    return False


def evaluate_mAP(all_preds, all_gts, class_key, classes, attribute_keys, class_wildcards, attribute_wildcards, score_key="score"):
    """
    Compute mean Average Precision (mAP) over event predictions.

    Evaluates one AP per class in `classes` (skipping wildcard classes) and
    averages them. For each class, predictions are matched to ground truth events
    using a greedy, confidence-ranked, one-to-one strategy:

    1. A prediction is first matched against ground truth events of the same class.
       A successful match counts as a True Positive.
    2. If no class match is found, the prediction is matched against ground truth
       events annotated with a wildcard class. A wildcard match silently discards
       the prediction — it counts neither as a TP nor as a FP.
    3. If neither match succeeds, the prediction is recorded as a False Positive.

    Wildcard ground truth events are tracked globally across all classes within
    each segment to prevent the same wildcard GT from absorbing predictions from
    more than one class.

    Args:
        all_preds          (List[List[dict]]): Predicted events per segment.
        all_gts            (List[List[dict]]): Ground truth events per segment.
                                               Must be aligned with `all_preds`.
        class_key          (str):              Event attribute that defines the class
                                               (e.g. "utterance_type" or
                                               "highlevel_action").
        classes            (Iterable[str]):    Complete set of classes to evaluate.
                                               Wildcard classes are skipped.
        attribute_keys     (List[str]):        Event attributes that must match
                                               exactly (beyond the class) for a
                                               prediction to be a TP.
        class_wildcards    (Iterable[str]):    Class values treated as wildcards;
                                               GT events with these values can absorb
                                               unmatched predictions of any class.
        attribute_wildcards (dict):            Mapping from attribute name to wildcard
                                               values accepted during matching
                                               (passed through to `_find_match`).
        score_key          (str):              Key in each prediction dict that holds
                                               the confidence score. Defaults to
                                               "score".

    Returns:
        float: mAP averaged over all classes that have at least one GT instance.
               Returns 0.0 if no such class exists.
    """
    aps = []

    # Wildcard GTs are shared across classes to prevent double-consuming
    matched_gt_j_per_seg = [set() for _ in range(len(all_gts))]

    for c in classes:
        if c in class_wildcards:
            continue

        class_preds = []
        num_gt = 0

        for seg_idx, (seg_preds, seg_gts) in enumerate(zip(all_preds, all_gts)):
            gt_c = [e for e in seg_gts if e.get(class_key) == c]
            gt_j = [e for e in seg_gts if any(v in attribute_wildcards.get(k, []) for k,v in e.items())]
            pred_c = sorted(
                [e for e in seg_preds if e.get(class_key) == c],
                key=lambda x: -x.get(score_key, 0.0)
            )

            num_gt += len(gt_c)

            matched_gt_c = set()
            matched_gt_j = matched_gt_j_per_seg[seg_idx]

            for p in pred_c:
                matched_c = _find_match(p, gt_c, matched_gt_c, attribute_keys, attribute_wildcards)
                if not matched_c:
                    matched_j = _find_match(p, gt_j, matched_gt_j, attribute_keys, attribute_wildcards)
                    if matched_j:
                        continue  # Absorbed by wildcard (ignore prediction entirely)

                class_preds.append((matched_c, p.get(score_key, 0.0)))

        if num_gt == 0:
            continue

        if class_preds:
            tp_flags = np.array([x[0] for x in class_preds])
            confs = np.array([x[1] for x in class_preds])
            ap = _compute_ap(tp_flags, confs, num_gt)
        else:
            ap = 0.0

        aps.append(ap)

    return float(np.mean(aps)) if aps else 0.0