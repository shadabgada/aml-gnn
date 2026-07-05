"""Permutation feature importance for the GCN detector (post-hoc explainability demo).

Trains a GCN under the standard chronological protocol, then measures, for each
input feature, the drop in test AUC-PR when that feature's values are randomly
permuted. A larger drop means the model relies more on that feature. This is a
model-agnostic post-hoc attribution method that gives a documentable rationale
for a flag, instantiating the interpretability recommendation in Section 5.3.5.

Edge features enter the model only through the edge classifier (decode), so their
permutation reuses cached node embeddings and is cheap. Node features enter
through message passing (encode), so their permutation re-encodes the full graph.

Usage (full run, e.g. on a laptop):
    python experiments/run_permutation_importance.py --epochs 100 --device cpu --repeats 5

Quick smoke test (verifies the pipeline, not the numbers):
    python experiments/run_permutation_importance.py --epochs 2 --repeats 1
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
from src.models.gcn import GCNEdgeClassifier
from src.training.trainer import GNNTrainer
from src.utils.config import DataConfig
from src.utils.logger import setup_logging
from src.utils.metrics import compute_all_metrics

logger = logging.getLogger(__name__)
OUT_JSON = "results/curves/permutation_importance.json"


def main():
    ap = argparse.ArgumentParser(description="GCN permutation feature importance")
    ap.add_argument("--variant", type=str, default="HI-Small")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--repeats", type=int, default=5,
                    help="Permutation repeats per feature (averaged; higher = less noise)")
    args = ap.parse_args()

    setup_logging(log_dir="results/logs", experiment_name="permutation_importance")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(8)

    # --- Data ------------------------------------------------------------
    logger.info("Loading %s dataset ...", args.variant)
    cfg = DataConfig(dataset_variant=args.variant)
    accounts, txn = load_raw_data(cfg)
    static = build_static_graph(accounts, txn, cfg)
    logger.info("Graph: %d nodes, %d edges | node_dim=%d edge_dim=%d",
                static.num_nodes, static.num_edges,
                static.num_node_features, static.num_edge_features)

    # --- Train GCN -------------------------------------------------------
    model = GCNEdgeClassifier(node_dim=static.num_node_features,
                              edge_dim=static.num_edge_features)
    logger.info("GCN: %d parameters", sum(p.numel() for p in model.parameters()))
    trainer = GNNTrainer(model=model, data=static.data, pos_weight=static.pos_weight,
                         pos_weight_multiplier=0.1, lr=1e-3, grad_clip=1.0,
                         patience=25, device=args.device)
    trainer.train(num_epochs=args.epochs)

    # --- Permutation importance -----------------------------------------
    m = trainer.model.eval()
    data = trainer.data
    test_idx = data.test_mask.nonzero(as_tuple=False).squeeze(1)
    n_test = int(test_idx.numel())
    y_true = data.edge_label[test_idx].cpu().numpy()
    rng = np.random.default_rng(args.seed)

    @torch.no_grad()
    def test_auc_pr(h):
        logits = m.decode(h, data.edge_index, data.edge_attr, idx=test_idx)
        yp = torch.sigmoid(logits).cpu().numpy()
        return compute_all_metrics(y_true, yp, threshold=0.5)["auc_pr"]

    with torch.no_grad():
        h_base = m.encode(data.x, data.edge_index)  # reused for all edge-feature perms
    baseline = test_auc_pr(h_base)
    logger.info("Baseline test AUC-PR: %.4f", baseline)

    results = {"baseline_auc_pr": baseline, "repeats": args.repeats, "edge": [], "node": []}

    # Edge features: permute test-edge values of column j; encode is unaffected.
    logger.info("--- Edge features (%d) ---", static.num_edge_features)
    for j, name in enumerate(static.edge_feature_names):
        drops = []
        col = data.edge_attr[test_idx, j].clone()
        for _ in range(args.repeats):
            perm = torch.as_tensor(rng.permutation(n_test), device=data.edge_attr.device)
            data.edge_attr[test_idx, j] = col[perm]
            drops.append(baseline - test_auc_pr(h_base))
        data.edge_attr[test_idx, j] = col  # restore
        mean, std = float(np.mean(drops)), float(np.std(drops))
        results["edge"].append({"feature": name, "importance": mean, "std": std})
        logger.info("  edge %-22s AUC-PR drop = %.4f +/- %.4f", name, mean, std)

    # Node features: permute column j across all nodes; re-encode the graph.
    logger.info("--- Node features (%d) ---", static.num_node_features)
    n_nodes = data.x.shape[0]
    for j, name in enumerate(static.node_feature_names):
        drops = []
        col = data.x[:, j].clone()
        for _ in range(args.repeats):
            perm = torch.as_tensor(rng.permutation(n_nodes), device=data.x.device)
            data.x[:, j] = col[perm]
            with torch.no_grad():
                h = m.encode(data.x, data.edge_index)
            drops.append(baseline - test_auc_pr(h))
        data.x[:, j] = col  # restore
        mean, std = float(np.mean(drops)), float(np.std(drops))
        results["node"].append({"feature": name, "importance": mean, "std": std})
        logger.info("  node %-22s AUC-PR drop = %.4f +/- %.4f", name, mean, std)

    results["edge"].sort(key=lambda d: -d["importance"])
    results["node"].sort(key=lambda d: -d["importance"])

    os.makedirs("results/curves", exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Top-5 edge features: %s", [d["feature"] for d in results["edge"][:5]])
    logger.info("Top-5 node features: %s", [d["feature"] for d in results["node"][:5]])
    logger.info("Saved %s", OUT_JSON)

    # --- Figure (top features by importance) -----------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        top = (results["edge"] + results["node"])
        top.sort(key=lambda d: -d["importance"])
        top = top[:15]
        names = [d["feature"] for d in top][::-1]
        vals = [d["importance"] for d in top][::-1]
        errs = [d["std"] for d in top][::-1]
        fig, ax = plt.subplots(figsize=(8, 5), dpi=130)
        ax.barh(names, vals, xerr=errs, color="#1f4e79")
        ax.set_xlabel("Test AUC-PR drop when feature is permuted")
        ax.set_title("GCN permutation feature importance (top 15)")
        fig.tight_layout()
        fig.savefig("results/curves/permutation_importance.png", bbox_inches="tight")
        logger.info("Saved results/curves/permutation_importance.png")
    except Exception as e:  # noqa: BLE001
        logger.warning("Figure generation skipped: %s", e)


if __name__ == "__main__":
    main()
