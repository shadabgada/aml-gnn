"""Train and evaluate static GNNs on the IBM AML dataset.

Usage:
    python experiments/run_gnn.py --model gcn
    python experiments/run_gnn.py --model gat --heads 4
    python experiments/run_gnn.py --model sage --aggregator mean
    python experiments/run_gnn.py --model all   # run all three
"""

import argparse
import logging
import sys
import time

sys.path.insert(0, ".")

import numpy as np
import torch

from src.data.loader import load_raw_data
from src.data.graph_constructor import build_static_graph
from src.models.gcn import GCNEdgeClassifier
from src.models.gat import GATEdgeClassifier
from src.models.sage import GraphSAGEEdgeClassifier
from src.training.evaluator import log_baseline_results
from src.training.trainer import GNNTrainer
from src.utils.config import DataConfig
from src.utils.logger import setup_logging
from src.utils.metrics import format_metrics

logger = logging.getLogger(__name__)

MODEL_REGISTRY = {
    "gcn": GCNEdgeClassifier,
    "gat": GATEdgeClassifier,
    "sage": GraphSAGEEdgeClassifier,
}


def build_model(args, num_node_feats, num_edge_feats):
    """Instantiate the requested GNN model."""
    cls = MODEL_REGISTRY[args.model]
    kwargs = dict(
        node_dim=num_node_feats,
        edge_dim=num_edge_feats,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    if args.model == "gat":
        kwargs["heads"] = args.heads
    elif args.model == "sage":
        kwargs["aggregator"] = args.aggregator

    return cls(**kwargs)


def run_single(args, setup_log: bool = True) -> dict:
    """Train one GNN and return results dict."""
    if setup_log:
        setup_logging(log_dir="results/logs", experiment_name=f"gnn_{args.model}")

    # Reproducibility -----------------------------------------------------
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(8)

    # Data ----------------------------------------------------------------
    logger.info("Loading %s dataset ...", args.variant)
    cfg = DataConfig(dataset_variant=args.variant)
    accounts, transactions = load_raw_data(cfg)
    static = build_static_graph(accounts, transactions, cfg)
    logger.info(
        "Graph: %d nodes, %d edges | node_dim=%d, edge_dim=%d | pos_weight=%.1f",
        static.num_nodes, static.num_edges,
        static.num_node_features, static.num_edge_features,
        static.pos_weight,
    )

    # Model ---------------------------------------------------------------
    model = build_model(args, static.num_node_features, static.num_edge_features)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("%s: %.0f parameters", args.model.upper(), n_params)

    # Train ---------------------------------------------------------------
    trainer = GNNTrainer(
        model=model,
        data=static.data,
        pos_weight=static.pos_weight,
        pos_weight_multiplier=args.pos_weight_mult,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        patience=args.patience,
        device=args.device,
    )
    trainer.train(num_epochs=args.epochs)

    # Evaluate ------------------------------------------------------------
    test_default = trainer.evaluate_test()
    test_cal = trainer.evaluate_test_calibrated()

    logger.info("=" * 65)
    logger.info("  %s — Test Results", args.model.upper())
    logger.info("=" * 65)
    logger.info("  Default threshold (0.5):   %s", format_metrics(test_default, "test_"))
    logger.info("  Calibrated threshold (%.4f): %s",
                trainer.calibrated_threshold, format_metrics(test_cal, "test_"))

    return {
        "model": args.model,
        "num_params": n_params,
        "threshold": trainer.calibrated_threshold,
        "test_default": test_default,
        "test_calibrated": test_cal,
        "history": trainer.history,
    }


def run_all(args):
    """Run GCN, GAT, and GraphSAGE sequentially and compare."""
    setup_logging(log_dir="results/logs", experiment_name="gnn_all")

    # Model-specific overrides
    model_configs = {
        "gcn":  {},
        "gat":  {"heads": 1},                         # 1 head to avoid OOM
        "sage": {"aggregator": "mean"},                 # normalize=True for stability
    }

    results = {}
    for model_name in ["gcn", "gat", "sage"]:
        logger.info("\n%s", "=" * 70)
        logger.info("  Running %s", model_name.upper())
        logger.info("%s\n", "=" * 70)
        args.model = model_name

        # Apply model-specific settings
        for k, v in model_configs.get(model_name, {}).items():
            setattr(args, k, v)

        try:
            results[model_name] = run_single(args, setup_log=False)
        except Exception as e:
            logger.error("Failed to run %s: %s", model_name, e, exc_info=True)

    # Comparison table ----------------------------------------------------
    logger.info("\n")
    logger.info("=" * 90)
    logger.info("  STATIC GNN COMPARISON (HI-Small)")
    logger.info("=" * 90)
    header = (
        f"{'Model':<12} {'Params':>8} "
        f"{'AUC-ROC':>10} {'AUC-PR':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} "
        f"{'Thresh':>8}"
    )
    logger.info(header)
    logger.info("-" * 90)

    # Also print baseline reference
    logger.info(
        "%-12s %8s %10s %10s %10s %10s %10s %8s",
        "---baselines---", "", "", "", "", "", "", "",
    )
    logger.info(
        "%-12s %8s %10.4f %10.4f %10.4f %10.4f %10.4f %8s",
        "XGBoost", "", 0.9381, 0.1511, 0.0265, 0.8610, 0.0514, "0.50",
    )
    logger.info(
        "%-12s %8s %10.4f %10.4f %10.4f %10.4f %10.4f %8s",
        "LR", "", 0.9378, 0.0376, 0.0135, 0.9295, 0.0267, "0.50",
    )
    logger.info("-" * 90)

    for name, r in results.items():
        m = r["test_calibrated"]
        t = r["threshold"]
        logger.info(
            "%-12s %8.0f %10.4f %10.4f %10.4f %10.4f %10.4f %8.4f",
            name.upper(), r["num_params"],
            m["auc_roc"], m["auc_pr"],
            m["precision"], m["recall"], m["f1"], t,
        )

    logger.info("=" * 90)
    logger.info("  Calibrated thresholds found on validation set (F1-optimal).")
    logger.info("  Baseline references at default threshold=0.50.")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train static GNNs for AML edge classification"
    )
    parser.add_argument("--model", type=str, default="gcn",
                        choices=["gcn", "gat", "sage", "all"])
    parser.add_argument("--variant", type=str, default="HI-Small")

    # Architecture
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--heads", type=int, default=4,
                        help="Attention heads (GAT only)")
    parser.add_argument("--aggregator", type=str, default="mean",
                        choices=["mean", "max", "lstm"],
                        help="Aggregator (GraphSAGE only)")

    # Training
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--pos_weight_mult", type=float, default=0.1,
                        help="Multiplier on auto-computed pos_weight (0.01-0.5)")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    if args.model == "all":
        run_all(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()
