"""Reliability and sensitivity analysis for the two headline models (GCN, TGN).

Supports the reliability claims in Section 3.6 / Section 4.4:
  - Seed stability: retrain across multiple random seeds, report mean +/- std.
  - Sensitivity: vary one principal hyperparameter per model and report the
    effect on test AUC-PR / AUC-ROC.

All runs use the same chronological split as the main experiments. Results are
written incrementally to results/curves/robustness.json so partial results
survive an interrupted run.

Usage:
    python experiments/run_robustness.py --device cuda --epochs_gnn 200 --epochs_tgn 100
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, ".")

import numpy as np
import torch

from src.data.loader import load_raw_data
from src.data.graph_constructor import build_static_graph
from src.data.temporal_data_builder import build_temporal_data
from src.models.gcn import GCNEdgeClassifier
from src.models.tgn_model import TGNModel
from src.training.trainer import GNNTrainer
from src.training.tgn_trainer import TGNTrainer
from src.utils.config import DataConfig
from src.utils.logger import setup_logging

logger = logging.getLogger(__name__)
OUT = "results/curves/robustness.json"
RESULTS: list = []


def _seed(s: int):
    np.random.seed(s)
    torch.manual_seed(s)


def _key(rec: dict):
    """Identity of a run, so a resumed invocation can skip completed work."""
    if rec.get("kind") == "seed":
        return ("seed", rec.get("model"), rec.get("seed"))
    return ("sensitivity", rec.get("model"), rec.get("param"), rec.get("value"))


def _load_existing():
    """Resume support: reload prior results so an interrupted run can be
    relaunched without discarding or overwriting completed runs (important on
    long unattended runs)."""
    if os.path.exists(OUT):
        try:
            with open(OUT) as f:
                RESULTS.extend(json.load(f))
            logger.info("Resume: loaded %d existing run(s) from %s", len(RESULTS), OUT)
        except (json.JSONDecodeError, OSError):
            logger.warning("Could not read %s; starting fresh", OUT)


def _done(key) -> bool:
    return any(_key(r) == key for r in RESULTS)


def _save(rec: dict):
    RESULTS.append(rec)
    os.makedirs("results/curves", exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(RESULTS, f, indent=2)
    logger.info("  -> %s", rec)


def run_gcn(static, seed, device, epochs, dropout=0.3):
    _seed(seed)
    model = GCNEdgeClassifier(node_dim=static.num_node_features,
                              edge_dim=static.num_edge_features, dropout=dropout)
    tr = GNNTrainer(model=model, data=static.data, pos_weight=static.pos_weight,
                    pos_weight_multiplier=0.1, lr=1e-3, grad_clip=1.0,
                    patience=25, device=device)
    tr.train(num_epochs=epochs)
    m = tr.evaluate_test_calibrated()
    return {"auc_roc": m["auc_roc"], "auc_pr": m["auc_pr"], "f1": m["f1"]}


def run_tgn(tgraph, seed, device, epochs, pos_weight_mult=0.01):
    _seed(seed)
    model = TGNModel(num_nodes=tgraph.num_nodes, edge_dim=tgraph.num_edge_features,
                     memory_dim=64, time_dim=8, dropout=0.3)
    tr = TGNTrainer(model=model, temporal_data=tgraph.data,
                    train_end_idx=tgraph.train_end_idx, val_end_idx=tgraph.val_end_idx,
                    pos_weight=tgraph.pos_weight, pos_weight_multiplier=pos_weight_mult,
                    lr=3e-3, grad_clip=0.0, patience=25, device=device)
    tr.train(num_epochs=epochs)
    cold = tr.evaluate_test_calibrated()
    warm = tr.evaluate_test_warm(threshold=0.5)
    return {"auc_roc": cold["auc_roc"], "auc_pr_cold": cold["auc_pr"],
            "auc_pr_warm": warm["auc_pr"], "f1": cold["f1"]}


def summarise(tag, runs):
    pr = np.array([r["auc_pr"] if "auc_pr" in r else r["auc_pr_cold"] for r in runs])
    roc = np.array([r["auc_roc"] for r in runs])
    logger.info("%s: AUC-ROC %.4f +/- %.4f | AUC-PR %.4f +/- %.4f (n=%d)",
                tag, roc.mean(), roc.std(), pr.mean(), pr.std(), len(runs))


def main():
    ap = argparse.ArgumentParser(description="Seed + sensitivity robustness runs")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--variant", type=str, default="HI-Small")
    ap.add_argument("--epochs_gnn", type=int, default=200)
    ap.add_argument("--epochs_tgn", type=int, default=100)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 7])
    ap.add_argument("--model", type=str, default="both", choices=["both", "gcn", "tgn"],
                    help="Run only GCN or only TGN. Lets GCN use GPU (venv_gpu) and "
                         "TGN use CPU (venv_new) in separate invocations, mirroring the "
                         "headline setup. Results merge into the same file.")
    args = ap.parse_args()

    setup_logging(log_dir="results/logs", experiment_name="robustness")
    torch.set_num_threads(8)
    _load_existing()

    do_gcn = args.model in ("both", "gcn")
    do_tgn = args.model in ("both", "tgn")

    cfg = DataConfig(dataset_variant=args.variant)
    accounts, txn = load_raw_data(cfg)
    static = build_static_graph(accounts, txn, cfg) if do_gcn else None
    tgraph = build_temporal_data(accounts, txn, cfg) if do_tgn else None

    # --- Seed stability -------------------------------------------------
    if do_gcn:
        for s in args.seeds:
            if _done(("seed", "GCN", s)):
                logger.info("skip (already done): GCN seed %d", s); continue
            logger.info("=== GCN seed %d ===", s)
            r = run_gcn(static, s, args.device, args.epochs_gnn)
            r.update(kind="seed", model="GCN", seed=s)
            _save(r)
        summarise("GCN seed-stability",
                  [r for r in RESULTS if r.get("kind") == "seed" and r.get("model") == "GCN"])

    if do_tgn:
        for s in args.seeds:
            if _done(("seed", "TGN", s)):
                logger.info("skip (already done): TGN seed %d", s); continue
            logger.info("=== TGN seed %d ===", s)
            r = run_tgn(tgraph, s, args.device, args.epochs_tgn)
            r.update(kind="seed", model="TGN", seed=s)
            _save(r)
        summarise("TGN seed-stability",
                  [r for r in RESULTS if r.get("kind") == "seed" and r.get("model") == "TGN"])

    # --- Sensitivity (principal hyperparameter, seed fixed at 42) -------
    if do_gcn:
        for dp in [0.2, 0.5]:
            if _done(("sensitivity", "GCN", "dropout", dp)):
                logger.info("skip (already done): GCN dropout=%.1f", dp); continue
            logger.info("=== GCN sensitivity dropout=%.1f ===", dp)
            r = run_gcn(static, 42, args.device, args.epochs_gnn, dropout=dp)
            r.update(kind="sensitivity", model="GCN", param="dropout", value=dp)
            _save(r)

    if do_tgn:
        for pw in [0.005, 0.02]:
            if _done(("sensitivity", "TGN", "pos_weight_mult", pw)):
                logger.info("skip (already done): TGN pos_weight_mult=%.3f", pw); continue
            logger.info("=== TGN sensitivity pos_weight_mult=%.3f ===", pw)
            r = run_tgn(tgraph, 42, args.device, args.epochs_tgn, pos_weight_mult=pw)
            r.update(kind="sensitivity", model="TGN", param="pos_weight_mult", value=pw)
            _save(r)

    logger.info("Robustness analysis complete -> %s", OUT)


if __name__ == "__main__":
    main()
