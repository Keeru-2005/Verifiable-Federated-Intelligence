"""Small shared helpers for threshold selection under severe class imbalance."""

import numpy as np
from sklearn.metrics import precision_recall_curve


def select_f1_threshold(y_true, scores):
    """Choose the probability threshold that maximizes F1 on the supplied split."""
    y_true = np.asarray(y_true, dtype=np.int32).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if y_true.size == 0 or y_true.size != scores.size:
        raise ValueError("Labels and scores must be nonempty arrays with equal length.")
    if not np.isfinite(scores).all():
        raise ValueError("Prediction scores must all be finite.")
    if np.unique(y_true).size != 2:
        raise ValueError("F1 threshold selection requires positive and negative examples.")

    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if thresholds.size == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    best = int(np.argmax(f1))
    return float(thresholds[best])


def binary_classification_metrics(y_true, scores, threshold):
    """Compute exact binary metrics and confusion counts at a fixed threshold."""
    y_true = np.asarray(y_true, dtype=np.int32).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if y_true.size == 0 or y_true.size != scores.size:
        raise ValueError("Labels and scores must be nonempty arrays with equal length.")
    if not np.isfinite(scores).all() or not np.isfinite(threshold):
        raise ValueError("Scores and threshold must be finite.")

    y_pred = (scores >= float(threshold)).astype(np.int32)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    support = int(y_true.size)
    positives = int(np.sum(y_true == 1))
    return {
        "accuracy": float((tp + tn) / support),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "positives": positives,
        "negatives": int(support - positives),
        "support": support,
        "prevalence": float(positives / support),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "decision_threshold": float(threshold),
    }
