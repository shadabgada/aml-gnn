"""Empirical scalability study for the AML GNN pipeline (workstream F).

Builds the static graph on progressively larger chronological prefixes of the
IBM AML dataset and, for each, measures how training time per epoch, inference
time, peak memory, and throughput scale with the number of transactions (edges).
This demonstrates scalability empirically rather than asserting it from the
architecture, addressing the requirement in Section 4.8 / Section 5.5.

A small, fixed number of epochs is timed per subset (convergence is not the
point here, cost is), and the mean per-epoch time is reported.

Usage:
    python experiments/run_scalability.py --device cpu --model gcn --epochs 3

Quick smoke test:
    python experiments/run_scalability.py --device cpu --model gcn --epochs 1 --fractions 0.1 0.2
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time

sys.path.insert(0, ".")

import numpy as np
import torch

from src.data.loader import load_raw_data
from src.data.graph_constructor import build_static_graph
from src.models.gcn import GCNEdgeClassifier
from src.models.sage import GraphSAGEEdgeClassifier
from src.training.trainer import GNNTrainer
from src.utils.config import DataConfig
from src.utils.logger import setup_logging

logger = logging.getLogger(__name__)
OUT_JSON = "results/curves/scalability.json"

MODELS = {"gcn": GCNEdgeClassifier, "sage": GraphSAGEEdgeClassifier}


def peak_mem_mb():
    """Best-effort peak resident memory in MB (portable)."""
    try:
        import resource
        m = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return m / 1e6 if sys.platform == "darwin" else m / 1024.0  # macOS bytes, Linux KB
    except Exception:
        try:
            import psutil
            return psutil.Process(os.getpid()).memory_info().rss / 1e6
        except Exception:
            return None


def main():
    ap = argparse.ArgumentParser(description="AML GNN scalability study")
    ap.add_argument("--variant", type=str, default="HI-Small")
    ap.add_argument("--model", type=str, default="gcn", choices=list(MODELS))
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--epochs", type=int, default=3, help="Timed epochs per subset")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fractions", type=float, nargs="+",
                    default=[0.2, 0.4, 0.6, 0.8, 1.0])
    args = ap.parse_args()

    setup_logging(log_dir="results/logs", experiment_name="scalability")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(8)

    cfg = DataConfig(dataset_variant=args.variant)
    accounts, transactions = load_raw_data(cfg)
    transactions = transactions.sort_values("timestamp").reset_index(drop=True)
    total = len(transactions)
    logger.info("Loaded %d transactions; scaling over fractions %s", total, args.fractions)

    results = []
    for frac in args.fractions:
        n = int(total * frac)
        sub = transactions.iloc[:n].copy()
        logger.info("=== fraction %.2f (%d transactions) ===", frac, n)

        static = build_static_graph(accounts, sub, cfg)
        model = MODELS[args.model](node_dim=static.num_node_features,
                                   edge_dim=static.num_edge_features)
        trainer = GNNTrainer(model=model, data=static.data, pos_weight=static.pos_weight,
                             pos_weight_multiplier=0.1, lr=1e-3, grad_clip=1.0,
                             patience=25, device=args.device, log_interval=10_000)
        train_edges = int(static.data.train_mask.sum().item())

        # Time training epochs -------------------------------------------
        epoch_times = []
        for _ in range(args.epochs):
            t0 = time.perf_counter()
            trainer._train_epoch()
            epoch_times.append(time.perf_counter() - t0)
        epoch_time = float(np.mean(epoch_times))

        # Time inference (full-graph encode + decode over the test edges) -
        t0 = time.perf_counter()
        trainer._evaluate(static.data.test_mask)
        infer_time = time.perf_counter() - t0

        mem = peak_mem_mb()
        thr = train_edges / epoch_time if epoch_time > 0 else 0.0
        rec = {
            "fraction": frac, "transactions": n,
            "nodes": int(static.num_nodes), "edges": int(static.num_edges),
            "train_edges": train_edges,
            "epoch_time_s": round(epoch_time, 3),
            "inference_time_s": round(infer_time, 3),
            "peak_mem_mb": round(mem, 1) if mem else None,
            "throughput_edges_per_s": round(thr, 0),
        }
        results.append(rec)
        logger.info("  edges=%d | epoch=%.2fs | infer=%.2fs | mem=%s MB | throughput=%.0f edges/s",
                    rec["edges"], epoch_time, infer_time, rec["peak_mem_mb"], thr)

        del static, model, trainer
        gc.collect()

    os.makedirs("results/curves", exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump({"model": args.model, "timed_epochs": args.epochs, "runs": results}, f, indent=2)
    logger.info("Saved %s", OUT_JSON)

    # --- Figure -----------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        e = [r["edges"] / 1e6 for r in results]
        fig, ax = plt.subplots(1, 3, figsize=(12, 3.6), dpi=130)
        ax[0].plot(e, [r["epoch_time_s"] for r in results], "o-", color="#1f4e79")
        ax[0].set_xlabel("edges (millions)"); ax[0].set_ylabel("training time / epoch (s)")
        ax[0].set_title("Training time")
        if all(r["peak_mem_mb"] for r in results):
            ax[1].plot(e, [r["peak_mem_mb"] for r in results], "o-", color="#c55a11")
        ax[1].set_xlabel("edges (millions)"); ax[1].set_ylabel("peak memory (MB)")
        ax[1].set_title("Peak memory")
        ax[2].plot(e, [r["throughput_edges_per_s"] / 1e6 for r in results], "o-", color="#548235")
        ax[2].set_xlabel("edges (millions)"); ax[2].set_ylabel("throughput (M edges/s)")
        ax[2].set_title("Training throughput")
        fig.suptitle(f"Pipeline scalability with dataset size ({args.model.upper()})")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig("results/curves/scalability.png", bbox_inches="tight")
        logger.info("Saved results/curves/scalability.png")
    except Exception as ex:  # noqa: BLE001
        logger.warning("Figure generation skipped: %s", ex)


if __name__ == "__main__":
    main()
