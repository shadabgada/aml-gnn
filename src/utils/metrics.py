"""Evaluation metrics for imbalanced binary edge classification.

Provides AUC-ROC, Precision, Recall, and per-class F1-score, with support
for threshold-dependent and threshold-independent metrics.
"""

from typing import Dict, Tuple

import numpy as np

from sklearn.metrics import (
    auc,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_all_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute comprehensive evaluation metrics for binary classification.

    Args:
        y_true: Ground-truth binary labels, shape (N,).
        y_prob: Predicted probabilities for the positive class, shape (N,).
        threshold: Decision threshold for hard predictions.

    Returns:
        Dictionary mapping metric name to value.
    """
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {}

    # Threshold-independent -----------------------------------------------
    metrics["auc_roc"] = roc_auc_score(y_true, y_prob)

    # Precision-Recall curve AUC -----------------------------------------
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
    metrics["auc_pr"] = auc(recall_curve, precision_curve)

    # Threshold-dependent ------------------------------------------------
    metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
    metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)

    # Per-class F1 --------------------------------------------------------
    metrics["f1_class_0"] = f1_score(y_true, y_pred, pos_label=0, zero_division=0)
    metrics["f1_class_1"] = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

    # Macro & weighted F1 ------------------------------------------------
    metrics["f1_macro"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["f1_weighted"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # Count-based sanity checks ------------------------------------------
    metrics["num_pos"] = float(np.sum(y_true == 1))
    metrics["num_neg"] = float(np.sum(y_true == 0))
    metrics["num_pred_pos"] = float(np.sum(y_pred == 1))

    return metrics


def compute_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute ROC curve points for plotting."""
    return roc_curve(y_true, y_prob)


def compute_pr_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute precision-recall curve points for plotting."""
    return precision_recall_curve(y_true, y_prob)


def calibrate_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f1",
    n_thresholds: int = 100,
) -> tuple[float, Dict[str, float]]:
    """Find the decision threshold that maximises a metric on validation data.

    For extreme class imbalance, threshold=0.5 is almost never optimal.
    This sweeps thresholds in [0.01, 0.99] and returns the best one.

    Args:
        y_true: Ground-truth binary labels.
        y_prob: Predicted probabilities for the positive class.
        metric: Metric to optimise — "f1", "f1_macro", or "f1_weighted".
        n_thresholds: Number of threshold candidates to evaluate.

    Returns:
        (best_threshold, metrics_at_best_threshold)
    """
    if np.sum(y_true == 1) == 0:
        return 0.5, compute_all_metrics(y_true, y_prob, threshold=0.5)

    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    best_thresh = 0.5
    best_score = -1.0
    best_metrics = {}

    for t in thresholds:
        m = compute_all_metrics(y_true, y_prob, threshold=t)
        score = m.get(metric, m.get("f1", 0))
        if score > best_score:
            best_score = score
            best_thresh = t
            best_metrics = m

    return float(best_thresh), best_metrics


def calibrate_threshold_at_recall(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_recall: float = 0.90,
) -> tuple[float, Dict[str, float]]:
    """Find the threshold that achieves at least target_recall while maximising precision.

    Useful when a minimum recall is required (e.g., compliance-driven AML).

    Args:
        y_true: Ground-truth binary labels.
        y_prob: Predicted probabilities for the positive class.
        target_recall: Minimum acceptable recall.

    Returns:
        (threshold, metrics_at_that_threshold)
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)

    # Find thresholds where recall >= target_recall
    valid = recalls >= target_recall
    if not np.any(valid):
        # Fall back to standard calibration
        return calibrate_threshold(y_true, y_prob, metric="f1")

    # Among valid thresholds, pick the one with highest precision
    best_idx = np.argmax(precisions[valid])
    # Map back to the original index
    valid_indices = np.where(valid)[0]
    chosen = valid_indices[best_idx]

    # precision_recall_curve returns n+1 thresholds; align
    # thresholds[i] corresponds to precision[i], recall[i] for i < len(thresholds)
    if chosen < len(thresholds):
        best_t = float(thresholds[chosen])
    else:
        best_t = 0.5

    metrics = compute_all_metrics(y_true, y_prob, threshold=best_t)
    return best_t, metrics


def format_metrics(metrics: Dict[str, float], prefix: str = "") -> str:
    """Format a metrics dict as a single-line string for logging."""
    parts = [f"{prefix}{k}={v:.4f}" for k, v in metrics.items()
             if k.startswith(("auc", "precision", "recall", "f1"))]
    return " | ".join(parts)
