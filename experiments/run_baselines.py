"""Run all three baseline classifiers (LR, RF, XGBoost) on the IBM AML dataset.

Baselines operate on flat edge features — no graph structure, no neighbor
information. This establishes the performance floor against which GNNs
are compared (addressing SQ3).

Usage:
    python experiments/run_baselines.py --variant HI-Small
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
from src.models.baselines import create_baselines, XGBoostBaseline
from src.training.evaluator import evaluate_baseline, log_baseline_results
from src.utils.config import DataConfig
from src.utils.logger import setup_logging
from src.utils.metrics import calibrate_threshold, compute_all_metrics, format_metrics


def main():
    parser = argparse.ArgumentParser(description="Run AML baseline classifiers")
    parser.add_argument("--variant", type=str, default="HI-Small",
                        choices=["HI-Small", "LI-Small", "HI-Medium", "LI-Medium"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_logging(log_dir="results/logs", experiment_name="baselines")

    # Set seeds -----------------------------------------------------------
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load data -----------------------------------------------------------
    logging.info("Loading dataset: %s", args.variant)
    cfg = DataConfig(dataset_variant=args.variant)
    accounts, transactions = load_raw_data(cfg)

    # Build static graph --------------------------------------------------
    static = build_static_graph(accounts, transactions, cfg)

    # Extract flat feature vectors + labels -------------------------------
    X = static.data.edge_attr.numpy()
    y = static.data.edge_label.numpy()
    train_mask = static.data.train_mask.numpy()
    val_mask = static.data.val_mask.numpy()
    test_mask = static.data.test_mask.numpy()

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    logging.info(
        "Train: %d edges (%d laundering), Val: %d (%d), Test: %d (%d)",
        len(y_train), int(y_train.sum()),
        len(y_val), int(y_val.sum()),
        len(y_test), int(y_test.sum()),
    )

    # Create and evaluate all baselines -----------------------------------
    baselines = create_baselines(pos_weight=static.pos_weight)

    for model in baselines:
        logging.info("Training %s ...", model.name)
        t0 = time.time()

        # Pass validation data to XGBoost for early stopping
        if isinstance(model, XGBoostBaseline):
            model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        else:
            model.fit(X_train, y_train)
        train_time = time.time() - t0

        # Predictions on all splits
        y_prob_train = model.predict_proba(X_train)
        y_prob_val = model.predict_proba(X_val)
        y_prob_test = model.predict_proba(X_test)

        # Calibrate threshold on validation set (F1-optimal)
        best_t, cal_metrics = calibrate_threshold(y_val, y_prob_val, metric="f1")
        logging.info(
            "  Calibrated threshold (F1-optimal on val): %.4f — %s",
            best_t, format_metrics(cal_metrics, prefix="val_cal_"),
        )

        # Evaluate at default threshold 0.5
        results_default = {
            "train": compute_all_metrics(y_train, y_prob_train, threshold=0.5),
            "val": compute_all_metrics(y_val, y_prob_val, threshold=0.5),
            "test": compute_all_metrics(y_test, y_prob_test, threshold=0.5),
        }
        results_default["train"]["_train_time_s"] = train_time

        # Evaluate at calibrated threshold
        results_cal = {
            "test": compute_all_metrics(y_test, y_prob_test, threshold=best_t),
            "threshold": best_t,
        }

        log_baseline_results(model.name, results_default)
        logging.info(
            "  Calibrated (t=%.4f): test_auc_roc=%.4f test_auc_pr=%.4f test_precision=%.4f test_recall=%.4f test_f1=%.4f",
            best_t,
            results_cal["test"]["auc_roc"],
            results_cal["test"]["auc_pr"],
            results_cal["test"]["precision"],
            results_cal["test"]["recall"],
            results_cal["test"]["f1"],
        )


if __name__ == "__main__":
    main()
