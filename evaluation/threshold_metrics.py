"""Small shared helpers for threshold selection under severe class imbalance."""

import numpy as np
from sklearn.metrics import precision_recall_curve
from scipy.optimize import minimize_scalar


def fit_temperature(logits, y_true):
    """Fit one positive temperature by minimizing validation binary log loss."""
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    if logits.size == 0 or logits.size != y_true.size:
        raise ValueError("Logits and labels must be nonempty arrays with equal length.")
    if not np.isfinite(logits).all() or not np.isfinite(y_true).all():
        raise ValueError("Logits and labels must be finite.")
    if not np.isin(y_true, (0.0, 1.0)).all() or np.unique(y_true).size != 2:
        raise ValueError("Temperature fitting requires binary labels from both classes.")

    def objective(log_temperature):
        scaled = logits / np.exp(log_temperature)
        return float(np.mean(np.logaddexp(0.0, scaled) - y_true * scaled))

    result = minimize_scalar(
        objective,
        bounds=(-5.0, 5.0),
        method="bounded",
        options={"xatol": 1e-5},
    )
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"Temperature scaling did not converge: {result.message}")
    return float(np.exp(result.x))


def select_smoothed_thresholds(y_true, scores, min_precision=0.70, window=3):
    """Select a moving-average F1 threshold and a highest-recall precision threshold."""
    y_true = np.asarray(y_true, dtype=np.int32).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if y_true.size == 0 or y_true.size != scores.size:
        raise ValueError("Labels and scores must be nonempty arrays with equal length.")
    if not np.isfinite(scores).all():
        raise ValueError("Prediction scores must all be finite.")
    if not 0.0 < min_precision <= 1.0:
        raise ValueError("min_precision must be in (0, 1].")
    if window < 1 or window % 2 == 0:
        raise ValueError("Smoothing window must be a positive odd integer.")
    if np.unique(y_true).size != 2:
        raise ValueError("Threshold selection requires positive and negative examples.")

    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if thresholds.size == 0:
        raise ValueError("Precision-recall curve did not produce thresholds.")
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(
        precision[:-1] + recall[:-1], 1e-12
    )
    pad = window // 2
    padded = np.pad(f1, (pad, pad), mode="edge")
    smoothed_f1 = np.convolve(
        padded, np.full(window, 1.0 / window), mode="valid"
    )
    best_index = int(np.argmax(smoothed_f1))

    eligible = np.flatnonzero(precision[:-1] >= min_precision)
    precision_index = None
    if eligible.size:
        best_recall = np.max(recall[eligible])
        best_precision_indexes = eligible[recall[eligible] == best_recall]
        precision_index = int(
            best_precision_indexes[
                np.argmax(precision[best_precision_indexes])
            ]
        )

    return {
        "f1_threshold": float(thresholds[best_index]),
        "smoothed_f1": float(smoothed_f1[best_index]),
        "raw_f1_at_threshold": float(f1[best_index]),
        "precision_threshold": (
            float(thresholds[precision_index])
            if precision_index is not None
            else None
        ),
        "precision_threshold_precision": (
            float(precision[precision_index])
            if precision_index is not None
            else None
        ),
        "precision_threshold_recall": (
            float(recall[precision_index])
            if precision_index is not None
            else None
        ),
        "smoothing_window": int(window),
        "minimum_precision": float(min_precision),
    }


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
