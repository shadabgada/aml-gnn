"""Training loop for static GNNs: GCN, GAT, GraphSAGE.

Full-batch edge classification with weighted BCE loss, validation
AUC-ROC early stopping, ReduceLROnPlateau LR scheduling, gradient
clipping, and threshold calibration for imbalanced evaluation.
"""

from __future__ import annotations

import logging
import time
from copy import deepcopy
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.utils.metrics import (
    calibrate_threshold,
    calibrate_threshold_at_recall,
    compute_all_metrics,
    format_metrics,
)

logger = logging.getLogger(__name__)


class GNNTrainer:
    """Full-batch GNN trainer for binary edge classification.

    Args:
        model: PyG GNN model returning edge logits.
        data: PyG Data with (x, edge_index, edge_attr, edge_label, masks).
        pos_weight: Weight for the positive class in BCEWithLogitsLoss.
        pos_weight_multiplier: Scale the pos_weight (e.g., 0.1 to moderate).
            The auto-computed weight is neg/pos (~1243 for HI-Small) which
            creates an extremely high-recall, low-precision classifier.
            Values 0.01–0.5 typically produce better calibrated models.
        lr: Initial learning rate.
        weight_decay: L2 regularisation strength.
        grad_clip: Max gradient norm (0 = no clipping).
        patience: Early-stopping patience (epochs with no val AUC-ROC improvement).
        lr_patience: LR scheduler patience.
        lr_factor: LR reduction factor.
        device: "cpu" or "cuda".
        log_interval: Log metrics every N epochs.
    """

    def __init__(
        self,
        model: nn.Module,
        data,
        pos_weight: float,
        pos_weight_multiplier: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 5e-4,
        grad_clip: float = 1.0,
        patience: int = 25,
        lr_patience: int = 10,
        lr_factor: float = 0.5,
        device: str = "cpu",
        log_interval: int = 10,
    ):
        self.model = model.to(device)
        self.data = data.to(device)
        self.device = device
        self.log_interval = log_interval
        self.grad_clip = grad_clip

        # Loss — moderate the extreme pos_weight ---------------------------
        moderated_pw = pos_weight * pos_weight_multiplier
        logger.info(
            "pos_weight: %.1f (raw) × %.3f (moderator) = %.1f",
            pos_weight, pos_weight_multiplier, moderated_pw,
        )
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([moderated_pw], device=device)
        )

        # Optimiser --------------------------------------------------------
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=lr_factor,
            patience=lr_patience,
        )

        # Early stopping state ---------------------------------------------
        self.patience = patience
        self.best_val_auc = 0.0
        self.best_state = None
        self.epochs_without_improvement = 0

        # Calibrated threshold ---------------------------------------------
        self.calibrated_threshold: float = 0.5
        self.calibrated_threshold_recall90: float = 0.5

        # Logging ----------------------------------------------------------
        self.history: Dict[str, list] = {
            "epoch": [], "train_loss": [], "val_auc": [], "lr": [],
        }

    def train(self, num_epochs: int = 200) -> Dict[str, list]:
        """Run the full training loop.

        After training, calibrates the decision threshold on the
        validation set for imbalanced evaluation.

        Returns:
            history dict with per-epoch metrics.
        """
        logger.info("Starting GNN training on %s (%d epochs)", self.device, num_epochs)
        t_start = time.time()

        for epoch in range(1, num_epochs + 1):
            # --- Train step ---
            train_loss = self._train_epoch()

            # --- Validation ---
            val_metrics = self._evaluate(self.data.val_mask)

            # --- Scheduler ---
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step(val_metrics["auc_roc"])

            # --- Logging ---
            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(train_loss)
            self.history["val_auc"].append(val_metrics["auc_roc"])
            self.history["lr"].append(current_lr)

            if epoch % self.log_interval == 0 or epoch == 1:
                logger.info(
                    "Epoch %3d | loss=%.4f | lr=%.6f | %s",
                    epoch, train_loss, current_lr,
                    format_metrics(val_metrics, prefix="val_"),
                )

            # --- Early stopping ---
            if val_metrics["auc_roc"] > self.best_val_auc:
                self.best_val_auc = val_metrics["auc_roc"]
                self.best_state = deepcopy(self.model.state_dict())
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
                if self.epochs_without_improvement >= self.patience:
                    logger.info(
                        "Early stopping at epoch %d (best val AUC-ROC=%.4f)",
                        epoch, self.best_val_auc,
                    )
                    break

        elapsed = time.time() - t_start
        logger.info("Training finished in %.1f min", elapsed / 60.0)

        # Restore best model -----------------------------------------------
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)

        # --- Threshold calibration on validation set -----------------------
        self._calibrate_threshold()

        return self.history

    def evaluate_test(self) -> Dict[str, float]:
        """Evaluate the (best) model on the test set at default threshold."""
        return self._evaluate(self.data.test_mask)

    def evaluate_test_calibrated(self) -> Dict[str, float]:
        """Evaluate on test set using the calibrated threshold (max F1 on val)."""
        return self._evaluate(
            self.data.test_mask, threshold=self.calibrated_threshold
        )

    def evaluate_test_at_recall(self, target_recall: float = 0.90) -> Dict[str, float]:
        """Evaluate on test set at a threshold tuned for target recall on val."""
        t, _ = calibrate_threshold_at_recall(
            *self._get_val_probs(), target_recall=target_recall,
        )
        return self._evaluate(self.data.test_mask, threshold=t)

    @property
    def calibrated_thresholds(self) -> Dict[str, float]:
        return {
            "f1_optimal": self.calibrated_threshold,
            "recall90": self.calibrated_threshold_recall90,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _train_epoch(self) -> float:
        self.model.train()
        self.optimizer.zero_grad()

        logits = self.model(
            self.data.x, self.data.edge_index, self.data.edge_attr,
        )
        loss = self.criterion(
            logits[self.data.train_mask],
            self.data.edge_label[self.data.train_mask],
        )
        loss.backward()

        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip,
            )

        self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def _evaluate(
        self, mask: torch.Tensor, threshold: float = 0.5
    ) -> Dict[str, float]:
        self.model.eval()
        logits = self.model(
            self.data.x, self.data.edge_index, self.data.edge_attr,
        )
        y_prob = torch.sigmoid(logits[mask]).cpu().numpy()
        y_true = self.data.edge_label[mask].cpu().numpy()
        return compute_all_metrics(y_true, y_prob, threshold=threshold)

    def _get_val_probs(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (y_true, y_prob) for the validation set."""
        self.model.eval()
        with torch.no_grad():
            logits = self.model(
                self.data.x, self.data.edge_index, self.data.edge_attr,
            )
            y_prob = torch.sigmoid(logits[self.data.val_mask]).cpu().numpy()
            y_true = self.data.edge_label[self.data.val_mask].cpu().numpy()
        return y_true, y_prob

    def _calibrate_threshold(self) -> None:
        """Find optimal thresholds using the validation set."""
        y_true, y_prob = self._get_val_probs()

        # F1-optimal threshold
        best_t, best_m = calibrate_threshold(y_true, y_prob, metric="f1")
        self.calibrated_threshold = best_t
        logger.info(
            "Calibrated threshold (F1-optimal on val): %.4f | %s",
            best_t, format_metrics(best_m, prefix="val_cal_"),
        )

        # Recall-90 threshold
        best_t90, best_m90 = calibrate_threshold_at_recall(
            y_true, y_prob, target_recall=0.90,
        )
        self.calibrated_threshold_recall90 = best_t90
        logger.info(
            "Calibrated threshold (recall>=0.90 on val): %.4f | %s",
            best_t90, format_metrics(best_m90, prefix="val_r90_"),
        )
