r"""Build PyTorch Geometric Data objects from IBM AML transaction data.

Two graph representations are constructed:

  StaticGraph
      A single Data object where every transaction is an edge. Used by
      GCN, GAT, and GraphSAGE. Time-based train/val/test edge masks.

  TemporalSnapshots
      A list of Data objects, each covering a fixed time window. All
      snapshots share the same node set (stable IDs). Used by EvolveGCN.

Design decisions (addressing SQ1):
  1. Accounts → nodes; transactions → directed edges.
  2. Stable node IDs are maintained across all time windows via a
     global account-to-index mapping built once from accounts.csv.
  3. Edge features are timestamp-derived, enabling both static models
     (which consume them as edge attr) and temporal models (which use
     timestamp for snapshot partitioning).
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch_geometric.data import Data

from src.data.feature_engineering import engineer_edge_features, engineer_node_features
from src.utils.config import DataConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------


@dataclass
class StaticGraph:
    """A single-graph representation for static GNNs."""

    data: Data  # PyG Data object with (x, edge_index, edge_attr, edge_label, masks)
    num_nodes: int
    num_edges: int
    node_feature_names: List[str]
    edge_feature_names: List[str]
    num_node_features: int
    num_edge_features: int
    pos_weight: float  # for BCEWithLogitsLoss


@dataclass
class TemporalSnapshots:
    """A sequence of time-ordered graph snapshots for temporal GNNs."""

    snapshots: List[Data]  # List[T] PyG Data objects, one per time window
    num_nodes: int
    num_snapshots: int
    node_feature_names: List[str]
    edge_feature_names: List[str]
    num_node_features: int
    num_edge_features: int
    pos_weight: float
    window_boundaries: List[float]  # timestamp boundaries between windows


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def build_static_graph(
    accounts: pd.DataFrame,
    transactions: pd.DataFrame,
    cfg: DataConfig,
) -> StaticGraph:
    """Build a single PyG Data object from the full transaction set.

    Edges are split chronologically into train / val / test using
    time-based boundaries (no shuffling), preventing data leakage.

    Returns:
        StaticGraph wrapping a PyG Data with:
          - data.x:            (N, F_n) node features
          - data.edge_index:   (2, E) directed edges
          - data.edge_attr:    (E, F_e) edge features
          - data.edge_label:   (E,) binary labels
          - data.train_mask:   (E,) boolean
          - data.val_mask:     (E,) boolean
          - data.test_mask:    (E,) boolean
    """
    logger.info("=== Building STATIC graph ===")

    # --- 1. Global node index (stable across all splits) ------------------
    account_ids, account_to_idx = _build_node_index(accounts, transactions)
    num_nodes = len(account_ids)

    # --- 2. Time-based train/val/test split ------------------------------
    train_mask, val_mask, test_mask = _time_based_split(transactions, cfg)

    # --- 3. Node features (computed from accounts + training txn stats) --
    train_txns = transactions[train_mask]
    node_feats, node_scaler, node_feat_names = engineer_node_features(
        accounts, train_txns, fit=True, scaler=None,
    )
    # Pad to full node set (accounts not in training get zero features) ---
    node_feats = _align_node_features(node_feats, account_to_idx, accounts)

    # --- 4. Edge features ------------------------------------------------
    edge_feats, edge_scaler, pay_enc, cur_enc, edge_feat_names = engineer_edge_features(
        transactions, fit=True, scaler=None, payment_encoder=None, currency_encoder=None,
        encode_cyclic_time=cfg.encode_cyclic_time,
    )

    # --- 5. Edge indices -------------------------------------------------
    src = transactions["from_account"].map(account_to_idx).values.astype(np.int64)
    dst = transactions["to_account"].map(account_to_idx).values.astype(np.int64)
    edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)

    # --- 6. Labels & masks -----------------------------------------------
    edge_label = torch.tensor(transactions["is_laundering"].values.astype(np.float32))
    train_mask = torch.tensor(train_mask.values, dtype=torch.bool)
    val_mask = torch.tensor(val_mask.values, dtype=torch.bool)
    test_mask = torch.tensor(test_mask.values, dtype=torch.bool)

    # --- 7. Pos weight for class imbalance -------------------------------
    pos_weight = _compute_pos_weight(edge_label[train_mask])

    data = Data(
        x=torch.tensor(node_feats, dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=torch.tensor(edge_feats, dtype=torch.float32),
        edge_label=edge_label,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )

    logger.info(
        "Static graph: %d nodes, %d edges, node_dim=%d, edge_dim=%d",
        num_nodes, edge_index.shape[1], node_feats.shape[1], edge_feats.shape[1],
    )
    logger.info(
        "Train/Val/Test edges: %d / %d / %d",
        train_mask.sum().item(), val_mask.sum().item(), test_mask.sum().item(),
    )

    return StaticGraph(
        data=data,
        num_nodes=num_nodes,
        num_edges=edge_index.shape[1],
        node_feature_names=node_feat_names,
        edge_feature_names=edge_feat_names,
        num_node_features=node_feats.shape[1],
        num_edge_features=edge_feats.shape[1],
        pos_weight=pos_weight,
    )


def build_temporal_snapshots(
    accounts: pd.DataFrame,
    transactions: pd.DataFrame,
    cfg: DataConfig,
) -> TemporalSnapshots:
    """Partition transactions into T time-ordered snapshots.

    Each snapshot is a PyG Data with edges from a fixed-width time window.
    All snapshots share the same node set (stable IDs). Node features are
    recomputed per window using transactions up to (and including) that
    window to avoid lookahead.

    Snapshots are partitioned into train/val/test blocks based on time
    window index (not individual edges). The first ``train_windows``
    snapshots form the training sequence; the remainder are validation
    and test respectively.

    Returns:
        TemporalSnapshots with:
          - snapshots: list of Data objects (ordered by time)
          - Each Data has: x, edge_index, edge_attr, edge_label, mask
    """
    logger.info("=== Building TEMPORAL graph (%d snapshots) ===", cfg.num_time_snapshots)

    # --- 1. Global node index --------------------------------------------
    account_ids, account_to_idx = _build_node_index(accounts, transactions)
    num_nodes = len(account_ids)

    # --- 2. Determine time boundaries ------------------------------------
    if "timestamp" not in transactions.columns:
        raise ValueError(
            "transactions must have a 'timestamp' column for temporal snapshot construction."
        )

    ts = _parse_timestamp(transactions["timestamp"])
    t_min, t_max = ts.min(), ts.max()

    if cfg.snapshot_strategy == "quantile":
        # Equal number of edges per window (handles skewed density)
        quantiles = np.linspace(0, 1, cfg.num_time_snapshots + 1)
        boundaries = np.quantile(ts, quantiles)
        # Avoid duplicate boundaries
        boundaries = np.unique(boundaries)
        if len(boundaries) < 3:
            raise ValueError("Too few unique timestamps for temporal snapshots.")
        logger.info(
            "Quantile-based windows: %d boundaries across [%.0f, %.0f]",
            len(boundaries) - 1, t_min, t_max,
        )
    else:
        boundaries = np.linspace(t_min, t_max, cfg.num_time_snapshots + 1)
        logger.info("Fixed-width windows: %.0f – %.0f  (%d windows)", t_min, t_max, cfg.num_time_snapshots)

    # --- 3. Fit feature encoders on TRAINING portion only ----------------
    num_windows = len(boundaries) - 1
    train_cutoff = int(num_windows * (1 - cfg.val_ratio - cfg.test_ratio))
    train_boundary = boundaries[train_cutoff]

    train_txn_mask = ts <= train_boundary
    train_txns = transactions[train_txn_mask]

    # Fit node featuriser on training transactions ------------------------
    _, node_scaler, node_feat_names = engineer_node_features(
        accounts, train_txns, fit=True,
    )
    # Fit edge featuriser -------------------------------------------------
    edge_scaler, pay_enc, cur_enc = _fit_edge_encoders(train_txns, cfg)
    # Get edge feature names (run once to discover dims)
    _, _, _, _, edge_feat_names = engineer_edge_features(
        train_txns.iloc[:100], fit=False,
        scaler=edge_scaler,
        payment_encoder=pay_enc, currency_encoder=cur_enc,
        encode_cyclic_time=cfg.encode_cyclic_time,
    )

    # --- 4. Build snapshots ----------------------------------------------
    snapshots: List[Data] = []

    for t_idx in range(num_windows):
        lo, hi = boundaries[t_idx], boundaries[t_idx + 1]
        in_window = (ts >= lo) & (ts < hi)
        w_txns = transactions[in_window].copy()

        # Node features from all transactions up to this window (no lookahead)
        past_txns = transactions[ts <= hi]
        node_feats, _, _ = engineer_node_features(
            accounts, past_txns,
            fit=False, scaler=node_scaler,
        )
        node_feats = _align_node_features(node_feats, account_to_idx, accounts)

        # Edge features for this window
        if len(w_txns) > 0:
            edge_feats, _, _, _, _ = engineer_edge_features(
                w_txns, fit=False, scaler=edge_scaler,
                payment_encoder=pay_enc, currency_encoder=cur_enc,
                encode_cyclic_time=cfg.encode_cyclic_time,
            )
        else:
            edge_feats = np.empty((0, len(edge_feat_names)), dtype=np.float32)

        # Edge indices
        if len(w_txns) > 0:
            src = w_txns["from_account"].map(account_to_idx).values.astype(np.int64)
            dst = w_txns["to_account"].map(account_to_idx).values.astype(np.int64)
            edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)
            edge_label = torch.tensor(w_txns["is_laundering"].values.astype(np.float32))
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_label = torch.empty((0,), dtype=torch.float32)

        # Train/val/test mask for this snapshot (window-level assignment)
        if t_idx < train_cutoff:
            mask = torch.full((len(w_txns),), True, dtype=torch.bool)
        else:
            mask = torch.full((len(w_txns),), False, dtype=torch.bool)

        data = Data(
            x=torch.tensor(node_feats, dtype=torch.float32),
            edge_index=edge_index,
            edge_attr=torch.tensor(edge_feats, dtype=torch.float32),
            edge_label=edge_label,
            train_mask=mask,
        )

        snapshots.append(data)
        logger.debug(
            "  Window %2d/%d: %7d edges, time [%.0f, %.0f)",
            t_idx + 1, num_windows, len(w_txns), lo, hi,
        )

    # --- 5. Pos weight from training windows -----------------------------
    all_train_labels = torch.cat([s.edge_label[s.train_mask] for s in snapshots])
    pos_weight = _compute_pos_weight(all_train_labels)

    logger.info(
        "Temporal graph: %d nodes, %d snapshots (train=%d, val+test=%d)",
        num_nodes, num_windows, train_cutoff,
        num_windows - train_cutoff,
    )

    return TemporalSnapshots(
        snapshots=snapshots,
        num_nodes=num_nodes,
        num_snapshots=num_windows,
        node_feature_names=node_feat_names,
        edge_feature_names=edge_feat_names,
        num_node_features=node_feats.shape[1],
        num_edge_features=len(edge_feat_names),
        pos_weight=pos_weight,
        window_boundaries=boundaries.tolist(),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_node_index(
    accounts: pd.DataFrame,
    transactions: pd.DataFrame,
) -> Tuple[List, Dict]:
    """Create stable global mapping from account_id → 0-based integer index.

    All accounts from accounts.csv are included, plus any accounts that
    appear only in transactions.csv (orphans — added during validation).
    """
    account_ids = sorted(accounts["account_id"].unique().tolist())
    txn_ids = set(transactions["from_account"]) | set(transactions["to_account"])
    missing = sorted(txn_ids - set(account_ids))
    all_ids = account_ids + missing
    return all_ids, {aid: i for i, aid in enumerate(all_ids)}


def _time_based_split(
    transactions: pd.DataFrame,
    cfg: DataConfig,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Assign each transaction to train/val/test by chronological order.

    Transactions are sorted by timestamp; the first (1 - val - test)%
    are assigned to train, the next `val_ratio` to val, and the remainder
    to test. This mirrors real-world AML detection where we train on
    past data and evaluate on future data.
    """
    ts = _parse_timestamp(transactions["timestamp"])
    order = np.argsort(ts)
    n = len(order)

    n_train = int(n * (1.0 - cfg.val_ratio - cfg.test_ratio))
    n_val = int(n * cfg.val_ratio)

    train_mask = pd.Series(False, index=transactions.index)
    val_mask = pd.Series(False, index=transactions.index)
    test_mask = pd.Series(False, index=transactions.index)

    train_mask.iloc[order[:n_train]] = True
    val_mask.iloc[order[n_train:n_train + n_val]] = True
    test_mask.iloc[order[n_train + n_val:]] = True

    logger.info(
        "Time-based split: train=%d, val=%d, test=%d (chronological order)",
        train_mask.sum(), val_mask.sum(), test_mask.sum(),
    )
    return train_mask, val_mask, test_mask


def _parse_timestamp(series: pd.Series) -> np.ndarray:
    """Robustly parse a timestamp column to numeric Unix epoch seconds.

    Handles both string formats ("2022/09/01 00:20") and raw numeric Unix timestamps.
    """
    # Try string parsing first (IBM AML default format)
    ts = pd.to_datetime(series, errors="coerce")
    if ts.notna().all():
        # .timestamp() gives float64 Unix seconds directly — avoids int64 precision issues
        return ts.map(pd.Timestamp.timestamp).values.astype(np.float64)
    # Fallback: treat as already numeric (seconds or milliseconds)
    raw = pd.to_numeric(series).values.astype(np.float64)
    if raw.max() > 1e12:  # likely milliseconds
        raw = raw / 1000.0
    return raw


def _align_node_features(
    node_feats: np.ndarray,
    account_to_idx: Dict,
    accounts: pd.DataFrame,
) -> np.ndarray:
    """Ensure node_feats covers every node index 0..N-1 in order."""
    n_full = len(account_to_idx)
    n_feat = node_feats.shape[1]

    # Build mapping from account_id → row position in node_feats
    account_ids_sorted = sorted(accounts["account_id"].unique())
    feat_idx_map = {aid: i for i, aid in enumerate(account_ids_sorted)}

    full = np.zeros((n_full, n_feat), dtype=np.float32)
    for aid, idx in account_to_idx.items():
        if aid in feat_idx_map:
            full[idx] = node_feats[feat_idx_map[aid]]
        # else: stays zero (orphan accounts with no features)

    return full


def _compute_pos_weight(labels: torch.Tensor) -> float:
    """pos_weight = (#negatives / #positives) for BCEWithLogitsLoss."""
    n_pos = labels.sum().item()
    n_neg = len(labels) - n_pos
    if n_pos == 0:
        logger.warning("No positive labels in training set — pos_weight set to 1.0")
        return 1.0
    pw = n_neg / n_pos
    logger.info("Computed pos_weight: %.2f (neg=%d, pos=%d)", pw, n_neg, int(n_pos))
    return pw


def _fit_edge_encoders(
    transactions: pd.DataFrame,
    cfg: DataConfig,
) -> Tuple[StandardScaler, LabelEncoder, LabelEncoder]:
    """Fit edge feature encoders once so they can be reused across snapshots."""
    _, scaler, pay_enc, cur_enc, _ = engineer_edge_features(
        transactions,
        fit=True,
        scaler=None,
        payment_encoder=None,
        currency_encoder=None,
        encode_cyclic_time=cfg.encode_cyclic_time,
    )
    return scaler, pay_enc, cur_enc
