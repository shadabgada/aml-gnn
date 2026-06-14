"""Training loop for snapshot temporal GNNs: TemporalGCN, EvolveGCN-H.

Differs from the static GNNTrainer in three ways:
  1. Training iterates over a *sequence* of graph snapshots.
  2. The model maintains hidden state across snapshots (GRU/RNN).
  3. Evaluation happens on future snapshots (temporal generalisation),
     mirroring real AML deployment where models are trained on historical
     data and evaluated on future transactions.
"""

from __future__ import annotations

import logging
import os
import time
from copy import deepcopy
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from src.utils.metrics import (
    calibrate_threshold,
    compute_all_metrics,
    format_metrics,
)

# Model types are distinguished by which GRU they use
_TEMPORAL_GCN_TYPES = ("TemporalGCN",)
_EVOLVE_GCN_TYPES = ("EvolveGCNH",)

logger = logging.getLogger(__name__)


class TemporalTrainer:
    """Trainer for temporal GNNs operating on snapshot sequences.

    Training proceeds chronologically through snapshots. The model's
    hidden state persists across snapshots but gradients are truncated
    between snapshots (truncated BPTT with k=1). This is more stable
    and mirrors online deployment.

    Args:
        model: Temporal GNN with forward_single_snapshot(x, edge_index,
            edge_attr, state) → (new_state, logits).
        snapshots: List of PyG Data objects.
        train_cutoff: Number of initial snapshots used for training.
        pos_weight: Weight for the positive class.
        pos_weight_multiplier: Scale the pos_weight (see GNNTrainer).
        lr: Initial learning rate.
        weight_decay: L2 regularisation strength.
        grad_clip: Max gradient norm.
        patience: Early-stopping patience based on val AUC-ROC.
        device: "cpu" or "cuda".
        log_interval: Log every N epochs.
    """

    def __init__(
        self,
        model: nn.Module,
        snapshots: List,
        train_cutoff: int,
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
        checkpoint_dir: str = "results/checkpoints",
        checkpoint_interval: int = 10,
        resume: bool = False,
        bptt_steps: int = 4,
    ):
        self.model = model.to(device)
        self.snapshots = [s.to(device) for s in snapshots]
        self.train_cutoff = train_cutoff
        self.device = device
        self.log_interval = log_interval
        self.grad_clip = grad_clip
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = checkpoint_interval
        self.bptt_steps = bptt_steps
        self._start_epoch = 1

        # Detect model type for dispatch
        model_name = model.__class__.__name__
        self._is_temporal_gcn = model_name in _TEMPORAL_GCN_TYPES
        self._is_evolve_gcn = model_name in _EVOLVE_GCN_TYPES

        # Loss -------------------------------------------------------------
        moderated_pw = pos_weight * pos_weight_multiplier
        logger.info(
            "pos_weight: %.1f x %.3f = %.1f",
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
            self.optimizer, mode="max", factor=lr_factor, patience=lr_patience,
        )

        # Early stopping ---------------------------------------------------
        self.patience = patience
        self.best_val_auc = 0.0
        self.best_state = None
        self.epochs_without_improvement = 0

        # Calibration ------------------------------------------------------
        self.calibrated_threshold: float = 0.5

        # Logging ----------------------------------------------------------
        self.history: Dict[str, list] = {
            "epoch": [], "train_loss": [], "val_auc": [], "lr": [],
        }

        # Snapshot info
        logger.info(
            "Temporal trainer: %d snapshots (train=%d, val+test=%d)",
            len(snapshots), train_cutoff, len(snapshots) - train_cutoff,
        )
        for i, s in enumerate(self.snapshots):
            tag = "train" if i < train_cutoff else "val/test"
            logger.debug(
                "  snapshot %2d [%s]: %d edges, %d nodes",
                i, tag, s.edge_index.shape[1], s.x.shape[0],
            )

    def train(self, num_epochs: int = 200) -> Dict[str, list]:
        """Run the full training loop with checkpointing."""
        model_name = self.model.__class__.__name__
        ckpt_path = os.path.join(self.checkpoint_dir, f"{model_name}_latest.pt")
        best_path = os.path.join(self.checkpoint_dir, f"{model_name}_best.pt")

        # Resume from checkpoint if available ---------------------------------
        if self._start_epoch == 1 and os.path.exists(ckpt_path):
            self._load_checkpoint(ckpt_path)
            logger.info("Resumed from checkpoint — starting at epoch %d", self._start_epoch)

        logger.info("Starting temporal GNN training on %s (%d epochs)",
                     self.device, num_epochs)
        t_start = time.time()

        for epoch in range(self._start_epoch, num_epochs + 1):
            train_loss = self._train_epoch()
            val_metrics = self._evaluate_validation()

            current_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step(val_metrics["auc_roc"])

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

            if val_metrics["auc_roc"] > self.best_val_auc:
                self.best_val_auc = val_metrics["auc_roc"]
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

            # Periodic checkpoint for crash recovery
            if epoch % self.checkpoint_interval == 0:
                self._save_checkpoint(ckpt_path, epoch)

        elapsed = time.time() - t_start
        logger.info("Training finished in %.1f min", elapsed / 60.0)

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)

        self._calibrate_threshold()
        return self.history

    def evaluate_test(self) -> Dict[str, float]:
        """Evaluate on test snapshots (those after train_cutoff)."""
        return self._evaluate_snapshots(
            self.snapshots[self.train_cutoff:], threshold=0.5,
        )

    def evaluate_test_calibrated(self) -> Dict[str, float]:
        return self._evaluate_snapshots(
            self.snapshots[self.train_cutoff:],
            threshold=self.calibrated_threshold,
        )

    def evaluate_per_snapshot(self) -> List[Dict[str, float]]:
        """Evaluate each snapshot individually (for trend analysis)."""
        results = []
        for i, snap in enumerate(self.snapshots):
            m = self._evaluate_snapshots([snap], threshold=0.5)
            m["snapshot_idx"] = i
            results.append(m)
        return results

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _save_checkpoint(self, path: str, epoch: int) -> None:
        """Save training state for crash recovery."""
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

    def _load_checkpoint(self, path: str) -> None:
        """Restore training state from checkpoint."""
        ckpt = torch.load(path, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.best_val_auc = ckpt.get("best_val_auc", 0.0)
        self.best_state = ckpt.get("best_state")
        self.history = ckpt.get("history", {"epoch": [], "train_loss": [], "val_auc": [], "lr": []})
        self._start_epoch = ckpt["epoch"] + 1
        logger.info("Loaded checkpoint from epoch %d (best AUC-ROC=%.4f)",
                     ckpt["epoch"], self.best_val_auc)

    def _train_epoch(self) -> float:
        """Process training snapshots chronologically.

        Backprop is done per-snapshot to keep memory bounded — the
        computation graph for each snapshot is freed before the next one
        loads. Gradients accumulate across snapshots, which is equivalent
        to summing losses (gradient of sum = sum of gradients).

        For TemporalGCN the node state is detached between snapshots
        (truncated BPTT k=1), so per-snapshot backprop is exact.
        For EvolveGCN-H the weight GRU state persists without detach
        across snapshots within a backward group (controlled by
        bptt_steps), allowing gradients to flow through the weight
        evolution over short horizons.
        """
        self.model.train()
        self.optimizer.zero_grad()

        total_loss = 0.0
        total_edges = 0

        if self._is_temporal_gcn:
            state = torch.zeros(
                self.snapshots[0].x.shape[0],
                self.model.hidden_dim,
                device=self.device,
            )
            for i in range(self.train_cutoff):
                snap = self.snapshots[i]
                state_i, logits = self.model.forward_single_snapshot(
                    snap.x, snap.edge_index, snap.edge_attr, state,
                )
                state = state_i.detach()  # truncated BPTT
                if snap.edge_label.shape[0] > 0:
                    loss = self.criterion(logits, snap.edge_label)
                    n_edges = snap.edge_label.shape[0]
                    total_loss += loss.detach().item() * n_edges
                    total_edges += n_edges
                    # Per-snapshot backprop; BCEWithLogitsLoss already
                    # averages per edge, so loss.backward() gives the
                    # correct gradient contribution for this snapshot.
                    loss.backward()
        else:
            # EvolveGCN-H: accumulate gradient over bptt_steps snapshots
            # to allow gradient flow through weight GRU evolution
            if hasattr(self.model, 'init_weight_states'):
                self.model.init_weight_states(self.device)
            train_snaps = self.snapshots[:self.train_cutoff]

            step_loss = 0.0
            step_edges = 0
            for i in range(len(train_snaps)):
                snap = train_snaps[i]
                _, self.model._weight_states, logits = \
                    self.model.forward_single_snapshot(
                        snap.x, snap.edge_index, snap.edge_attr,
                        self.model._weight_states,
                    )
                if snap.edge_label.shape[0] > 0:
                    loss = self.criterion(logits, snap.edge_label)
                    n = snap.edge_label.shape[0]
                    step_loss += loss * n
                    step_edges += n

                # Backprop every bptt_steps snapshots (or at end)
                if (i + 1) % self.bptt_steps == 0 or i == len(train_snaps) - 1:
                    if step_edges > 0:
                        (step_loss / step_edges).backward()
                        total_loss += step_loss.detach().item()
                        total_edges += step_edges
                    step_loss = 0.0
                    step_edges = 0
                    # Detach weight states to truncate BPTT
                    self.model._weight_states = [
                        s.detach() for s in self.model._weight_states
                    ]

        if total_edges > 0:
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip,
                )
            self.optimizer.step()
            return total_loss / total_edges
        return 0.0

    @torch.no_grad()
    def _evaluate_validation(self) -> Dict[str, float]:
        """Evaluate on first validation snapshot for early stopping."""
        if self.train_cutoff >= len(self.snapshots):
            return {"auc_roc": 0.0, "auc_pr": 0.0}
        return self._evaluate_snapshots(
            [self.snapshots[self.train_cutoff]], threshold=0.5,
        )

    @torch.no_grad()
    def _evaluate_snapshots(
        self,
        snapshots: List,
        threshold: float = 0.5,
    ) -> Dict[str, float]:
        """Run forward pass through snapshots and compute metrics."""
        self.model.eval()

        all_probs = []
        all_labels = []

        if self._is_temporal_gcn:
            state = torch.zeros(
                snapshots[0].x.shape[0], self.model.hidden_dim,
                device=self.device,
            )
            for snap in snapshots:
                state, logits = self.model.forward_single_snapshot(
                    snap.x, snap.edge_index, snap.edge_attr, state,
                )
                if snap.edge_label.shape[0] > 0:
                    all_probs.append(torch.sigmoid(logits).cpu().numpy())
                    all_labels.append(snap.edge_label.cpu().numpy())
        else:
            if hasattr(self.model, 'init_weight_states'):
                self.model.init_weight_states(self.device)
            all_logits = self.model.forward_sequence(snapshots)
            for i, logits in enumerate(all_logits):
                labels = snapshots[i].edge_label
                if labels.shape[0] > 0:
                    all_probs.append(torch.sigmoid(logits).cpu().numpy())
                    all_labels.append(labels.cpu().numpy())

        if not all_probs:
            return {"auc_roc": 0.5, "auc_pr": 0.0}

        y_prob = np.concatenate(all_probs)
        y_true = np.concatenate(all_labels)
        return compute_all_metrics(y_true, y_prob, threshold=threshold)

    @torch.no_grad()
    def _calibrate_threshold(self) -> None:
        """Find F1-optimal threshold on validation snapshots."""
        self.model.eval()
        all_probs, all_labels = [], []

        val_snaps = self.snapshots[self.train_cutoff:]

        if self._is_temporal_gcn:
            state = torch.zeros(
                val_snaps[0].x.shape[0], self.model.hidden_dim,
                device=self.device,
            )
            for snap in val_snaps:
                state, logits = self.model.forward_single_snapshot(
                    snap.x, snap.edge_index, snap.edge_attr, state,
                )
                if snap.edge_label.shape[0] > 0:
                    all_probs.append(torch.sigmoid(logits).cpu().numpy())
                    all_labels.append(snap.edge_label.cpu().numpy())
        else:
            if hasattr(self.model, 'init_weight_states'):
                self.model.init_weight_states(self.device)
            all_logits = self.model.forward_sequence(val_snaps)
            for i, logits in enumerate(all_logits):
                labels = val_snaps[i].edge_label
                if labels.shape[0] > 0:
                    all_probs.append(torch.sigmoid(logits).cpu().numpy())
                    all_labels.append(labels.cpu().numpy())

        if all_probs:
            y_prob = np.concatenate(all_probs)
            y_true = np.concatenate(all_labels)
            best_t, best_m = calibrate_threshold(y_true, y_prob, metric="f1")
            self.calibrated_threshold = best_t
            logger.info(
                "Calibrated threshold (F1-optimal on val): %.4f | %s",
                best_t, format_metrics(best_m, prefix="val_cal_"),
            )
