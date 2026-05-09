"""Train and evaluate TGN on the IBM AML dataset.

Continuous-time temporal graph network that processes each transaction
with its individual timestamp. Per-node GRU memory accumulates account
behaviour over time, enabling detection of behavioural drift patterns
that snapshot-based models miss.

Usage:
    python experiments/run_tgn.py --variant HI-Small
    python experiments/run_tgn.py --variant HI-Small --memory_dim 256 --epochs 150
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, ".")

import numpy as np
import torch

from src.data.loader import load_raw_data
from src.data.temporal_data_builder import build_temporal_data
from src.models.tgn_model import TGNModel
from src.training.tgn_trainer import TGNTrainer
from src.utils.config import DataConfig
from src.utils.logger import setup_logging
from src.utils.metrics import format_metrics

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Train TGN for AML edge classification"
    )
    parser.add_argument("--variant", type=str, default="HI-Small")
    parser.add_argument("--device", type=str, default="cpu")

    # Architecture
    parser.add_argument("--memory_dim", type=int, default=128)
    parser.add_argument("--time_dim", type=int, default=16)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)

    # Training
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--pos_weight_mult", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_dir", type=str, default="results/checkpoints")
    parser.add_argument("--checkpoint_interval", type=int, default=10)

    args = parser.parse_args()

    setup_logging(log_dir="results/logs", experiment_name="tgn")

    # Reproducibility --------------------------------------------------------
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(8)

    # Data -------------------------------------------------------------------
    logger.info("Loading %s dataset ...", args.variant)
    cfg = DataConfig(dataset_variant=args.variant)
    accounts, transactions = load_raw_data(cfg)

    graph_data = build_temporal_data(accounts, transactions, cfg)
    logger.info(
        "TemporalData: %d nodes, %d edges, %d edge features, "
        "time range [%.0f, %.0f] days",
        graph_data.num_nodes, graph_data.num_edges,
        graph_data.num_edge_features,
        graph_data.data.t.min().item() / 86400,
        graph_data.data.t.max().item() / 86400,
    )

    # Model ------------------------------------------------------------------
    model = TGNModel(
        num_nodes=graph_data.num_nodes,
        edge_dim=graph_data.num_edge_features,
        memory_dim=args.memory_dim,
        time_dim=args.time_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("TGN: %.0f parameters (memory_dim=%d, time_dim=%d)",
                 n_params, args.memory_dim, args.time_dim)

    # Train ------------------------------------------------------------------
    trainer = TGNTrainer(
        model=model,
        temporal_data=graph_data.data,
        train_end_idx=graph_data.train_end_idx,
        val_end_idx=graph_data.val_end_idx,
        pos_weight=graph_data.pos_weight,
        pos_weight_multiplier=args.pos_weight_mult,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        patience=args.patience,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
    )
    trainer.train(num_epochs=args.epochs)

    # Evaluate ---------------------------------------------------------------
    test_default = trainer.evaluate_test()
    test_cal = trainer.evaluate_test_calibrated()

    logger.info("=" * 65)
    logger.info("  TGN — Test Results")
    logger.info("=" * 65)
    logger.info("  Default threshold (0.50):  %s",
                 format_metrics(test_default, "test_"))
    logger.info("  Calibrated threshold (%.4f): %s",
                 trainer.calibrated_threshold, format_metrics(test_cal, "test_"))

    # Per-time-slice evaluation
    per_slice = trainer.evaluate_per_time_slice(num_slices=12)
    logger.info("\n  Per-time-slice performance (threshold=0.5):")
    logger.info("  %6s  %8s  %8s  %8s  %8s",
                 "Slice", "AUC-ROC", "AUC-PR", "Prec", "Rec")
    for sm in per_slice:
        logger.info(
            "  %6d  %8.4f  %8.4f  %8.4f  %8.4f",
            sm["slice_idx"],
            sm.get("auc_roc", 0),
            sm.get("auc_pr", 0),
            sm.get("precision", 0),
            sm.get("recall", 0),
        )

    logger.info(
        "\nTGN Summary: AUC-ROC=%.4f AUC-PR=%.4f (calibrated threshold=%.4f)",
        test_cal["auc_roc"], test_cal["auc_pr"],
        trainer.calibrated_threshold,
    )


if __name__ == "__main__":
    main()
