"""Training loop for TGN on AML edge classification.

Processes transaction edges chronologically. Memory updates happen inside
the forward pass so the message projector receives gradients.
"""

from __future__ import annotations

import logging
import os
import time
from copy import deepcopy
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from src.utils.metrics import (
    calibrate_threshold,
    compute_all_metrics,
    format_metrics,
)

logger = logging.getLogger(__name__)


class TGNTrainer:
    """Trainer for TGN on chronological edge classification.

    Args:
        model: TGNModel instance.
        temporal_data: TemporalData with all edges (src, dst, t, msg, y).
        train_end_idx: Last training edge index (exclusive).
        val_end_idx: Last validation edge index (exclusive).
        pos_weight: Weight for BCEWithLogitsLoss positive class.
        batch_size: Edges per batch.
        lr: Initial learning rate.
        weight_decay: L2 regularisation.
        grad_clip: Max gradient norm.
        patience: Early-stopping patience (val AUC-ROC).
        device: "cpu" or "cuda".
        checkpoint_dir: Directory for checkpoints.
        checkpoint_interval: Save latest checkpoint every N epochs.
    """

    def __init__(
        self,
        model: nn.Module,
        temporal_data,
        train_end_idx: int,
        val_end_idx: int,
        pos_weight: float,
        pos_weight_multiplier: float = 0.1,
        batch_size: int = 2048,
        lr: float = 1e-3,
        weight_decay: float = 5e-4,
        grad_clip: float = 1.0,
        patience: int = 25,
        lr_patience: int = 10,
        lr_factor: float = 0.5,
        device: str = "cpu",
        log_interval: int = 5,
        checkpoint_dir: str = "results/checkpoints",
        checkpoint_interval: int = 10,
    ):
        self.model = model.to(device)
        self.data = temporal_data.to(device)
        self.train_end_idx = train_end_idx
        self.val_end_idx = val_end_idx
        self.device = device
        self.batch_size = batch_size
        self.grad_clip = grad_clip
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = checkpoint_interval
        self.log_interval = log_interval
        self._start_epoch = 1

        # Loss ---------------------------------------------------------------
        moderated_pw = pos_weight * pos_weight_multiplier
        logger.info("pos_weight: %.1f x %.3f = %.1f",
                      pos_weight, pos_weight_multiplier, moderated_pw)
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([moderated_pw], device=device)
        )

        # Optimiser ----------------------------------------------------------
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=lr_factor, patience=lr_patience,
        )

        # Early stopping -----------------------------------------------------
        self.patience = patience
        self.best_val_auc = 0.0
        self.best_state = None
        self.epochs_without_improvement = 0

        # Calibration --------------------------------------------------------
        self.calibrated_threshold: float = 0.5

        # History ------------------------------------------------------------
        self.history: Dict[str, list] = {
            "epoch": [], "train_loss": [], "val_auc": [], "lr": [],
        }

        # Edge ranges for train/val/test ------------------------------------
        self.train_edges = (0, train_end_idx)
        self.val_edges = (train_end_idx, val_end_idx)
        self.test_edges = (val_end_idx, len(self.data.y))

        logger.info(
            "TGN trainer: %d edges (train=%d, val=%d, test=%d), batch=%d",
            len(self.data.y),
            train_end_idx, val_end_idx - train_end_idx,
            len(self.data.y) - val_end_idx,
            batch_size,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, num_epochs: int = 100) -> Dict[str, list]:
        """Run the full training loop."""
        model_name = self.model.__class__.__name__
        ckpt_path = os.path.join(self.checkpoint_dir, f"{model_name}_latest.pt")
        best_path = os.path.join(self.checkpoint_dir, f"{model_name}_best.pt")

        logger.info("Starting TGN training on %s (%d epochs)", self.device, num_epochs)
        t_start = time.time()

        for epoch in range(self._start_epoch, num_epochs + 1):
            train_loss = self._train_epoch(epoch)
            val_metrics = self._evaluate_split(self.val_edges)
            val_auc = val_metrics["auc_roc"]

            current_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step(val_auc)

            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(train_loss)
            self.history["val_auc"].append(val_auc)
            self.history["lr"].append(current_lr)

            if epoch % self.log_interval == 0 or epoch == 1:
                logger.info(
                    "Epoch %3d | loss=%.4f | lr=%.6f | %s",
                    epoch, train_loss, current_lr,
                    format_metrics(val_metrics, prefix="val_"),
                )

            if val_auc > self.best_val_auc:
                self.best_val_auc = val_auc
                self.best_state = deepcopy(self.model.state_dict())
                self.epochs_without_improvement = 0
                self._save_checkpoint(best_path, epoch)
            else:
                self.epochs_without_improvement += 1
                if self.epochs_without_improvement >= self.patience:
                    logger.info(
                        "Early stopping at epoch %d (best val AUC-ROC=%.4f)",
                        epoch, self.best_val_auc,
                    )
                    break

            if epoch % self.checkpoint_interval == 0:
                self._save_checkpoint(ckpt_path, epoch)

        elapsed = time.time() - t_start
        logger.info("Training finished in %.1f min", elapsed / 60.0)

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)

        self._calibrate_threshold()
        return self.history

    def evaluate_test(self) -> Dict[str, float]:
        return self._evaluate_split(self.test_edges, threshold=0.5)

    def evaluate_test_calibrated(self) -> Dict[str, float]:
        return self._evaluate_split(
            self.test_edges, threshold=self.calibrated_threshold,
        )

    def evaluate_per_time_slice(self, num_slices: int = 12) -> list[Dict]:
        """Evaluate on equal-sized time slices for trend analysis."""
        test_start, test_end = self.test_edges
        n_test = test_end - test_start
        slice_size = n_test // num_slices
        results = []
        for i in range(num_slices):
            lo = test_start + i * slice_size
            hi = min(test_start + (i + 1) * slice_size, test_end)
            m = self._evaluate_split((lo, hi), threshold=0.5)
            m["slice_idx"] = i
            results.append(m)
        return results

    # ------------------------------------------------------------------
    # Internal: training
    # ------------------------------------------------------------------

    def _train_epoch(self, epoch: int) -> float:
        """Process training edges chronologically.

        Memory updates happen INSIDE compute_edge_logits (update_memory=True),
        so the message projector receives gradients from the loss.
        """
        self.model.train()
        self.model.reset_memory()

        total_loss = 0.0
        total_edges = 0

        batch_starts = list(range(
            self.train_edges[0], self.train_edges[1], self.batch_size,
        ))
        perm = torch.randperm(len(batch_starts))

        for idx in perm:
            start = batch_starts[idx]
            end = min(start + self.batch_size, self.train_edges[1])
            if end <= start:
                continue

            self.optimizer.zero_grad()

            src, dst, t, msg, y = self._slice(start, end)

            # Forward with memory update (msg projector receives gradients)
            logits = self.model.compute_edge_logits(
                src, dst, t, msg, update_memory=True,
            )

            loss = self.criterion(logits, y)
            loss.backward()

            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip,
                )

            self.optimizer.step()
            self.model.detach_memory()

            total_loss += loss.detach().item() * (end - start)
            total_edges += (end - start)

        if total_edges > 0:
            return total_loss / total_edges
        return 0.0

    # ------------------------------------------------------------------
    # Internal: evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _evaluate_split(
        self, edge_range: tuple[int, int], threshold: float = 0.5,
    ) -> Dict[str, float]:
        """Evaluate on a contiguous range of edges (no memory update)."""
        self.model.eval()
        self.model.reset_memory()

        all_probs = []
        all_labels = []

        start, end = edge_range
        for lo in range(start, end, self.batch_size):
            hi = min(lo + self.batch_size, end)
            if hi <= lo:
                continue

            src, dst, t, msg, y = self._slice(lo, hi)

            # Eval: memory is read but NOT updated by compute_edge_logits
            logits = self.model.compute_edge_logits(
                src, dst, t, msg, update_memory=False,
            )
            all_probs.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(y.cpu().numpy())

            # Manually update memory for chronological consistency
            new_memory, _, n_id = self.model.memory.update_and_embed(
                src, dst, t, msg,
            )

        if not all_probs:
            return {"auc_roc": 0.5, "auc_pr": 0.0}

        y_prob = np.concatenate(all_probs)
        y_true = np.concatenate(all_labels)
        return compute_all_metrics(y_true, y_prob, threshold=threshold)

    @torch.no_grad()
    def _calibrate_threshold(self) -> None:
        """Find F1-optimal threshold on validation edges."""
        self.model.eval()
        self.model.reset_memory()

        all_probs = []
        all_labels = []

        start, end = self.val_edges
        for lo in range(start, end, self.batch_size):
            hi = min(lo + self.batch_size, end)
            if hi <= lo:
                continue

            src, dst, t, msg, y = self._slice(lo, hi)

            logits = self.model.compute_edge_logits(
                src, dst, t, msg, update_memory=False,
            )
            all_probs.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(y.cpu().numpy())

            # Manually update memory for chronological consistency
            new_memory, _, n_id = self.model.memory.update_and_embed(
                src, dst, t, msg,
            )

        if all_probs:
            y_prob = np.concatenate(all_probs)
            y_true = np.concatenate(all_labels)
            best_t, best_m = calibrate_threshold(y_true, y_prob, metric="f1")
            self.calibrated_threshold = best_t
            logger.info(
                "Calibrated threshold (F1-optimal on val): %.4f | %s",
                best_t, format_metrics(best_m, prefix="val_cal_"),
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _slice(self, start: int, end: int):
        """Extract a slice of edges from the TemporalData."""
        return (
            self.data.src[start:end],
            self.data.dst[start:end],
            self.data.t[start:end],
            self.data.msg[start:end],
            self.data.y[start:end],
        )

    def _save_checkpoint(self, path: str, epoch: int) -> None:
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_auc": self.best_val_auc,
            "best_state": self.best_state,
            "history": self.history,
        }, path)
