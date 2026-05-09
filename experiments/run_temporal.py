"""Train and evaluate temporal GNNs on the IBM AML dataset.

Compares temporal GNNs against static GNNs by training on historical
snapshots and evaluating on future ones — the deployment-realistic
scenario for AML systems.

Usage:
    python experiments/run_temporal.py --model temporal_gcn
    python experiments/run_temporal.py --model evolve_gcn_h
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, ".")

import numpy as np
import torch

from src.data.loader import load_raw_data
from src.data.graph_constructor import build_temporal_snapshots
from src.models.temporal_gnn import TemporalGCN, EvolveGCNH
from src.training.temporal_trainer import TemporalTrainer
from src.utils.config import DataConfig
from src.utils.logger import setup_logging
from src.utils.metrics import format_metrics

CACHE_DIR = "data/processed"

logger = logging.getLogger(__name__)

MODEL_REGISTRY = {
    "temporal_gcn": TemporalGCN,
    "evolve_gcn_h": EvolveGCNH,
}


def main():
    parser = argparse.ArgumentParser(
        description="Train temporal GNNs for AML edge classification"
    )
    parser.add_argument("--model", type=str, default="temporal_gcn",
                        choices=["temporal_gcn", "evolve_gcn_h"])
    parser.add_argument("--variant", type=str, default="HI-Small")

    # Architecture
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--rank", type=int, default=8,
                        help="Low-rank adaptation rank (EvolveGCN-H only)")

    # Temporal
    parser.add_argument("--num_snapshots", type=int, default=12)
    parser.add_argument("--snapshot_strategy", type=str, default="quantile",
                        choices=["quantile", "fixed"])

    # Training
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--pos_weight_mult", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint")
    parser.add_argument("--checkpoint_dir", type=str,
                        default="results/checkpoints")
    parser.add_argument("--checkpoint_interval", type=int, default=10)

    args = parser.parse_args()

    setup_logging(log_dir="results/logs", experiment_name=f"temporal_{args.model}")

    # Reproducibility -----------------------------------------------------
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(8)

    # Data ----------------------------------------------------------------
    cfg = DataConfig(
        dataset_variant=args.variant,
        num_time_snapshots=args.num_snapshots,
        snapshot_strategy=args.snapshot_strategy,
    )

    # Try disk cache first — building 12 snapshots takes ~10 min
    cache_path = os.path.join(
        CACHE_DIR,
        f"{args.variant}_{args.num_snapshots}snap_{args.snapshot_strategy}.pt",
    )
    if os.path.exists(cache_path):
        logger.info("Loading cached snapshots from %s", cache_path)
        temporal = torch.load(cache_path, weights_only=False)
    else:
        logger.info("Loading %s dataset ...", args.variant)
        accounts, transactions = load_raw_data(cfg)
        temporal = build_temporal_snapshots(accounts, transactions, cfg)
        os.makedirs(CACHE_DIR, exist_ok=True)
        logger.info("Caching snapshots to %s", cache_path)
        torch.save(temporal, cache_path)

    logger.info(
        "Temporal graph: %d nodes, %d snapshots (pos_weight=%.1f)",
        temporal.num_nodes, temporal.num_snapshots, temporal.pos_weight,
    )
    for i, s in enumerate(temporal.snapshots):
        logger.debug(
            "  [%d] edges=%d, nodes=%d",
            i, s.edge_index.shape[1], s.x.shape[0],
        )

    # Compute train cutoff from config
    train_cutoff = int(
        temporal.num_snapshots * (1 - cfg.val_ratio - cfg.test_ratio)
    )
    logger.info(
        "Snapshot split: train=[0..%d), val+test=[%d..%d)",
        train_cutoff, train_cutoff, temporal.num_snapshots,
    )

    # Model ---------------------------------------------------------------
    cls = MODEL_REGISTRY[args.model]
    kwargs = dict(
        node_dim=temporal.num_node_features,
        edge_dim=temporal.num_edge_features,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    if args.model == "evolve_gcn_h":
        kwargs["rank"] = args.rank

    model = cls(**kwargs)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("%s: %.0f parameters", args.model.upper(), n_params)

    # Train ---------------------------------------------------------------
    trainer = TemporalTrainer(
        model=model,
        snapshots=temporal.snapshots,
        train_cutoff=train_cutoff,
        pos_weight=temporal.pos_weight,
        pos_weight_multiplier=args.pos_weight_mult,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        patience=args.patience,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
        resume=args.resume,
    )
    trainer.train(num_epochs=args.epochs)

    # Evaluate ------------------------------------------------------------
    test_default = trainer.evaluate_test()
    test_cal = trainer.evaluate_test_calibrated()

    logger.info("=" * 65)
    logger.info("  %s — Test Results", args.model.upper())
    logger.info("=" * 65)
    logger.info("  Default threshold (0.50):  %s", format_metrics(test_default, "test_"))
    logger.info("  Calibrated threshold (%.4f): %s",
                trainer.calibrated_threshold, format_metrics(test_cal, "test_"))

    # Per-snapshot evaluation (trend analysis)
    per_snap = trainer.evaluate_per_snapshot()
    logger.info("\n  Per-snapshot performance (threshold=0.5):")
    logger.info("  %6s  %8s  %8s  %8s  %8s", "Snap", "AUC-ROC", "AUC-PR", "Prec", "Rec")
    for snap_m in per_snap:
        logger.info(
            "  %6d  %8.4f  %8.4f  %8.4f  %8.4f",
            snap_m["snapshot_idx"],
            snap_m.get("auc_roc", 0),
            snap_m.get("auc_pr", 0),
            snap_m.get("precision", 0),
            snap_m.get("recall", 0),
        )

    # Log summary
    logger.info(
        "\n%s Summary: AUC-ROC=%.4f AUC-PR=%.4f (calibrated threshold=%.4f)",
        args.model.upper(),
        test_cal["auc_roc"], test_cal["auc_pr"],
        trainer.calibrated_threshold,
    )


if __name__ == "__main__":
    main()
