"""Ablation: does GNN message passing add signal beyond hand-crafted features?

The examiner noted that the node features already encode relational summaries
(in/out-degree, amounts, counterparty counts), so a GNN's advantage might come
from those features rather than from message passing. This ablation isolates the
contribution of message passing by training the SAME non-graph classifier
(XGBoost, identical config to the Tier-1 baseline) on three feature sets, all on
the SAME chronological split every model uses:

  (a) edge features only           (28)  -> reproduces the Tier-1 baseline
  (b) src+dst node features only   (24)  -> hand-crafted relational features, no graph
  (c) edge + node features         (52)  -> everything a GCN sees, minus message passing

The GCN (which adds message passing on top of (c)) is compared against these in
Chapter 4. If GCN > (c), message passing adds value beyond the engineered
features; if GCN ~ (c), the graph structure adds nothing the features did not
already capture.

Usage:
    python experiments/run_ablation.py --variant HI-Small --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np

from src.data.loader import load_raw_data
from src.data.graph_constructor import build_static_graph
from src.models.baselines import XGBoostBaseline
from src.utils.config import DataConfig
from src.utils.logger import setup_logging
from src.utils.metrics import calibrate_threshold, compute_all_metrics

logger = logging.getLogger(__name__)


def run_set(name, X, y, tr, va, te, pos_weight):
    """Train XGBoost on one feature set and evaluate on the chronological test set."""
    model = XGBoostBaseline(pos_weight=pos_weight)
    model.fit(X[tr], y[tr], X_val=X[va], y_val=y[va])
    p_val = model.predict_proba(X[va])
    p_te = model.predict_proba(X[te])
    t, _ = calibrate_threshold(y[va], p_val, metric="f1")
    m_def = compute_all_metrics(y[te], p_te, threshold=0.5)
    m_cal = compute_all_metrics(y[te], p_te, threshold=t)
    logger.info(
        "%-12s (%2d feats): AUC-ROC=%.4f AUC-PR=%.4f | cal thr=%.3f "
        "F1=%.4f P=%.4f R=%.4f",
        name, X.shape[1], m_def["auc_roc"], m_def["auc_pr"], t,
        m_cal["f1"], m_cal["precision"], m_cal["recall"],
    )
    return {
        "feature_set": name, "n_features": int(X.shape[1]),
        "auc_roc": m_def["auc_roc"], "auc_pr": m_def["auc_pr"],
        "threshold": t, "f1_cal": m_cal["f1"],
        "precision_cal": m_cal["precision"], "recall_cal": m_cal["recall"],
    }


def main():
    ap = argparse.ArgumentParser(description="GNN vs hand-crafted-features ablation")
    ap.add_argument("--variant", type=str, default="HI-Small")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    setup_logging(log_dir="results/logs", experiment_name="ablation")
    np.random.seed(args.seed)

    cfg = DataConfig(dataset_variant=args.variant)
    accounts, txn = load_raw_data(cfg)
    g = build_static_graph(accounts, txn, cfg)
    d = g.data

    ei = d.edge_index.numpy()
    edge = d.edge_attr.numpy()
    x = d.x.numpy()
    y = d.edge_label.numpy()
    tr, va, te = d.train_mask.numpy(), d.val_mask.numpy(), d.test_mask.numpy()

    src, dst = ei[0], ei[1]
    node = np.concatenate([x[src], x[dst]], axis=1)      # 24
    both = np.concatenate([edge, node], axis=1)          # 52

    logger.info("Chronological split: train=%d val=%d test=%d | pos_weight=%.1f",
                tr.sum(), va.sum(), te.sum(), g.pos_weight)

    results = []
    for name, X in [("edge_only", edge), ("node_only", node), ("edge+node", both)]:
        results.append(run_set(name, X, y, tr, va, te, g.pos_weight))

    Path("results/eda").mkdir(parents=True, exist_ok=True)
    Path("results/eda/ablation.json").write_text(json.dumps(results, indent=2))
    logger.info("Saved results/eda/ablation.json")

    print("\n=== ABLATION SUMMARY (chronological test set) ===")
    print(f"{'feature set':<12} {'feats':>5} {'AUC-ROC':>8} {'AUC-PR':>8} {'F1(cal)':>8}")
    for r in results:
        print(f"{r['feature_set']:<12} {r['n_features']:>5} "
              f"{r['auc_roc']:>8.4f} {r['auc_pr']:>8.4f} {r['f1_cal']:>8.4f}")
    print("Compare 'edge+node' (no message passing) against the GCN in Chapter 4.")


if __name__ == "__main__":
    main()
