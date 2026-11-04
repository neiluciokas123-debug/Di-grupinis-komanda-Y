"""Bendros metrikų funkcijos visiems modeliams."""
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score)


def best_threshold(y_true, scores) -> float:
    """Slenkstis, maksimizuojantis F1 (renkamas TIK validacijos aibėje, ne teste)."""
    y_true = np.asarray(y_true)
    if y_true.sum() == 0:
        return 0.5
    p, r, t = precision_recall_curve(y_true, scores)
    f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
    i = int(np.nanargmax(f1[:-1])) if len(t) else 0
    return float(t[i]) if len(t) else 0.5


def binary_metrics(y_true, scores, threshold: float) -> dict:
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    y_pred = (scores >= threshold).astype(int)
    two_classes = len(np.unique(y_true)) > 1
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, scores) if two_classes else float("nan"),
        "PR-AUC": average_precision_score(y_true, scores) if two_classes else float("nan"),
        "Slenkstis": threshold,
    }
