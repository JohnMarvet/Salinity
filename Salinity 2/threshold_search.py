"""Tune probability decision threshold on a validation split."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from sklearn.metrics import f1_score, recall_score


def find_best_threshold(
    y_true,
    y_proba_positive,
    *,
    metric: str = "recall",
    min_threshold: float = 0.05,
    max_threshold: float = 0.95,
    step: float = 0.05,
) -> Tuple[float, float]:
    """
    Scan thresholds and return (best_threshold, best_score).
    metric: recall | f1
    """
    best_t = 0.5
    best_score = -1.0

    thresholds = np.arange(min_threshold, max_threshold + step / 2, step)
    for t in thresholds:
        pred = (y_proba_positive >= t).astype(int)
        if metric == "f1":
            score = f1_score(y_true, pred, zero_division=0)
        else:
            score = recall_score(y_true, pred, zero_division=0)
        if score > best_score:
            best_score = float(score)
            best_t = float(t)

    return best_t, best_score
