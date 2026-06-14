"""Evaluation harness for all models: LR, RF, XGBoost, GCN, GAT, GraphSAGE,
TemporalGCN, EvolveGCN-H, TGN.

Provides a consistent evaluation interface across all model types, computing
the full metrics suite on train/val/test splits.
"""

import logging
from typing import Dict, Optional

import numpy as np
import torch

from src.utils.metrics import compute_all_metrics, format_metrics

logger = logging.getLogger(__name__)


def evaluate_baseline(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Dict[str, float]]:
    """Evaluate a baseline (tabular) model on train/val/test splits.

    Args:
        model: Object with predict_proba(X) → (N,) array.
        X_train, y_train: Training features and labels.
        X_val, y_val: Validation features and labels.
        X_test, y_test: Test features and labels.
        threshold: Decision threshold for hard predictions.

    Returns:
        Nested dict: {"train": {...}, "val": {...}, "test": {...}}
    """
    results = {}
    for name, X, y in [
        ("train", X_train, y_train),
        ("val", X_val, y_val),
        ("test", X_test, y_test),
    ]:
        y_prob = model.predict_proba(X)
        metrics = compute_all_metrics(y, y_prob, threshold=threshold)
        results[name] = metrics

    return results


def log_baseline_results(
    model_name: str,
    results: Dict[str, Dict[str, float]],
) -> None:
    """Log baseline evaluation results in a compact table format."""
    logger.info("=" * 70)
    logger.info("  %s", model_name)
    logger.info("=" * 70)
    header = f"{'Split':<8} {'AUC-ROC':>8} {'AUC-PR':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'F1-macro':>9}"
    logger.info(header)
    logger.info("-" * 70)
    for split in ("train", "val", "test"):
        m = results[split]
        logger.info(
            f"{split:<8} {m['auc_roc']:>8.4f} {m['auc_pr']:>8.4f} "
            f"{m['precision']:>8.4f} {m['recall']:>8.4f} "
            f"{m['f1']:>8.4f} {m['f1_macro']:>9.4f}"
        )
    logger.info("-" * 70)
    test = results["test"]
    logger.info(
        "  Test set: %d pos / %d neg, predicted %d pos",
        int(test["num_pos"]), int(test["num_neg"]), int(test["num_pred_pos"]),
    )


def evaluate_gnn(
    model: torch.nn.Module,
    data,
    mask: torch.Tensor,
    device: str = "cpu",
) -> Dict[str, float]:
    """Evaluate a GNN model on a specific edge mask.

    Args:
        model: PyTorch GNN model.
        data: PyG Data object with x, edge_index, edge_attr, edge_label.
        mask: Boolean mask over edges to evaluate on.
        device: Device string.

    Returns:
        Metrics dict from compute_all_metrics.
    """
    model.eval()
    with torch.no_grad():
        logits = model(data.x.to(device), data.edge_index.to(device),
                       data.edge_attr.to(device))
        if isinstance(logits, tuple):
            logits = logits[0]  # some GNNs return (logits, attention)
        y_prob = torch.sigmoid(logits[mask]).cpu().numpy()
        y_true = data.edge_label[mask].cpu().numpy()

    return compute_all_metrics(y_true, y_prob)
