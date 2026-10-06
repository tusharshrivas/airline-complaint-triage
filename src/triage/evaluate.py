"""Metric computation shared by all models (classical and LLM)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from triage import config


def compute_metrics(y_true, y_pred, labels: list[str]) -> dict:
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    _, _, wf, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    per_class = {
        lab: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])} for i, lab in enumerate(labels)
    }
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(np.mean(p)),
        "macro_recall": float(np.mean(r)),
        "macro_f1": float(np.mean(f)),
        "weighted_f1": float(wf),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": list(labels),
    }


def prediction_path(model_name: str, split: str = "test"):
    return config.PREDICTIONS / f"{model_name}_{split}.csv"


def load_predictions(model_name: str, split: str = "test") -> pd.DataFrame | None:
    path = prediction_path(model_name, split)
    return pd.read_csv(path) if path.exists() else None
