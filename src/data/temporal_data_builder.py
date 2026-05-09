"""Build PyG TemporalData from IBM AML transactions for TGN training.

Converts raw transactions into a single TemporalData object where each
edge carries its individual timestamp. Unlike the snapshot-based approach
(build_temporal_snapshots), this preserves fine-grained temporal ordering
so the TGN memory module can track per-node interaction histories.

Design:
  1. Global node index from accounts + transactions (same as static/temporal).
  2. Edge features fitted on training-time transactions only (no leakage).
  3. Chronological edge ordering with train/val/test split by time.
  4. Timestamps converted to relative seconds from training start.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import TemporalData

from src.data.feature_engineering import engineer_edge_features
from src.data.graph_constructor import _build_node_index, _parse_timestamp
from src.utils.config import DataConfig

logger = logging.getLogger(__name__)


@dataclass
class TemporalGraphData:
    """Container for TGN-ready temporal graph data."""

    data: TemporalData
    num_nodes: int
    num_edges: int
    num_edge_features: int
    edge_feature_names: List[str]
    pos_weight: float
    train_end_idx: int  # last training edge index (exclusive)
    val_end_idx: int     # last val edge index (exclusive)


def build_temporal_data(
    accounts: pd.DataFrame,
    transactions: pd.DataFrame,
    cfg: DataConfig,
) -> TemporalGraphData:
    """Convert transactions into a chronological TemporalData object.

    Args:
        accounts: Normalised accounts DataFrame.
        transactions: Normalised transactions DataFrame with 'timestamp'.
        cfg: DataConfig specifying val/test ratios.

    Returns:
        TemporalGraphData wrapping a PyG TemporalData with:
          - data.src:    (E,) source node indices
          - data.dst:    (E,) destination node indices
          - data.t:      (E,) relative timestamps (seconds from train start)
          - data.msg:    (E, F_e) edge features
          - data.y:      (E,) binary laundering labels
    """
    logger.info("=== Building TEMPORAL DATA (TGN format) ===")

    # --- 1. Global node index -----------------------------------------------
    account_ids, account_to_idx = _build_node_index(accounts, transactions)
    num_nodes = len(account_ids)

    # --- 2. Parse timestamps and sort chronologically -----------------------
    ts = _parse_timestamp(transactions["timestamp"])
    chrono_order = np.argsort(ts)
    ts_sorted = ts[chrono_order]
    txns_sorted = transactions.iloc[chrono_order].reset_index(drop=True)

    # Relative time from training start (seconds)
    t_min = ts_sorted.min()
    t_rel = (ts_sorted - t_min).astype(np.float32)

    # --- 3. Chronological train/val/test split ------------------------------
    n_total = len(txns_sorted)
    n_train = int(n_total * (1.0 - cfg.val_ratio - cfg.test_ratio))
    n_val = int(n_total * cfg.val_ratio)
    train_end_idx = n_train
    val_end_idx = n_train + n_val

    logger.info(
        "Chronological split: train=[0..%d), val=[%d..%d), test=[%d..%d] (%d total)",
        n_train, n_train, val_end_idx, val_end_idx, n_total, n_total,
    )

    # --- 4. Fit edge feature encoders on training portion ONLY --------------
    train_txns = txns_sorted.iloc[:n_train]
    edge_feats_all, edge_scaler, pay_enc, cur_enc, edge_feat_names = \
        engineer_edge_features(
            train_txns, fit=True, scaler=None,
            payment_encoder=None, currency_encoder=None,
            encode_cyclic_time=cfg.encode_cyclic_time,
        )
    logger.info("Edge feature encoders fitted on %d training transactions", n_train)

    # --- 5. Compute edge features for ALL transactions ----------------------
    if n_total > len(train_txns):
        edge_feats_all, _, _, _, _ = engineer_edge_features(
            txns_sorted, fit=False, scaler=edge_scaler,
            payment_encoder=pay_enc, currency_encoder=cur_enc,
            encode_cyclic_time=cfg.encode_cyclic_time,
        )

    # --- 6. Edge indices ----------------------------------------------------
    src = txns_sorted["from_account"].map(account_to_idx).values.astype(np.int64)
    dst = txns_sorted["to_account"].map(account_to_idx).values.astype(np.int64)

    # --- 7. Labels ----------------------------------------------------------
    y = txns_sorted["is_laundering"].values.astype(np.float32)

    # --- 8. Pos weight from training edges ----------------------------------
    train_labels = y[:n_train]
    n_pos = int(train_labels.sum())
    n_neg = n_train - n_pos
    pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    logger.info("Computed pos_weight: %.2f (neg=%d, pos=%d)", pos_weight, n_neg, n_pos)

    # --- 9. Build TemporalData ----------------------------------------------
    data = TemporalData(
        src=torch.tensor(src, dtype=torch.long),
        dst=torch.tensor(dst, dtype=torch.long),
        t=torch.tensor(t_rel, dtype=torch.float32),
        msg=torch.tensor(edge_feats_all, dtype=torch.float32),
        y=torch.tensor(y, dtype=torch.float32),
    )

    logger.info(
        "TemporalData: %d nodes, %d edges, edge_dim=%d, time range [%.0f, %.0f]s",
        num_nodes, n_total, len(edge_feat_names), t_rel[0], t_rel[-1],
    )

    return TemporalGraphData(
        data=data,
        num_nodes=num_nodes,
        num_edges=n_total,
        num_edge_features=len(edge_feat_names),
        edge_feature_names=edge_feat_names,
        pos_weight=pos_weight,
        train_end_idx=train_end_idx,
        val_end_idx=val_end_idx,
    )
