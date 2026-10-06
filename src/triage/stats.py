"""Paired bootstrap for the macro-F1 difference between two systems on the same test items."""

from __future__ import annotations

import numpy as np


def macro_f1_idx(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    cm = np.bincount(y_true * k + y_pred, minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(float)
    denom = cm.sum(0) + cm.sum(1)  # = 2tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros(k), where=denom > 0)
    return float(f1.mean())


def bootstrap_macro_f1_diff(y_true, pred_a, pred_b, labels: list[str], n_resamples: int = 1000, seed: int = 42) -> dict:
    """CI of macro_F1(a) - macro_F1(b) by resampling test items (paired, with replacement)."""
    idx = {lab: i for i, lab in enumerate(labels)}
    yt = np.array([idx[v] for v in y_true])
    a = np.array([idx[v] for v in pred_a])
    b = np.array([idx[v] for v in pred_b])
    k, n = len(labels), len(yt)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_resamples)
    for i in range(n_resamples):
        s = rng.integers(0, n, n)
        diffs[i] = macro_f1_idx(yt[s], a[s], k) - macro_f1_idx(yt[s], b[s], k)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    point = macro_f1_idx(yt, a, k) - macro_f1_idx(yt, b, k)
    return {
        "metric": "macro_f1(a) - macro_f1(b)",
        "point_estimate": float(point),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "includes_zero": bool(lo <= 0 <= hi),
        "n_resamples": n_resamples,
        "seed": seed,
        "n_items": int(n),
    }
