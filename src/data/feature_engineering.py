"""Feature engineering for nodes (accounts) and edges (transactions).

Node features = account-level attributes + aggregated transaction-history stats.
Edge features = amount (log1p) + cyclic time + one-hot categoricals.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def engineer_node_features(
    accounts: pd.DataFrame,
    transactions: pd.DataFrame,
    fit: bool = True,
    scaler: Optional[StandardScaler] = None,
) -> Tuple[np.ndarray, StandardScaler, List[str]]:
    """Build per-account (node) feature matrix.

    Args:
        accounts: Normalised accounts DataFrame with "account_id".
        transactions: Normalised transactions DataFrame.
        fit: If True, fit a new scaler; otherwise use *scaler*.
        scaler: Pre-fitted StandardScaler (required when fit=False).

    Returns:
        (node_features, scaler, feature_names) — features indexed by
        sorted account_id order.
    """
    df = accounts.set_index("account_id").sort_index()

    # Drop pure-ID columns (not features) ---------------------------------
    for c in ["account_number", "entity_id"]:
        if c in df.columns:
            df = df.drop(columns=[c])

    # Extract entity type from entity_name ---------------------------------
    if "entity_name" in df.columns:
        df["entity_type"] = df["entity_name"].apply(_extract_entity_type)
        df = df.drop(columns=["entity_name"])

    # Label-encode categorical columns ------------------------------------
    cat_cols = _categorical_columns(df)
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    # Aggregate transaction statistics per account ------------------------
    txn_stats = _compute_transaction_stats(transactions)

    # Join — accounts with no transactions get zero features ---------------
    df = df.join(txn_stats, how="left").fillna(0.0)

    feature_names = list(df.columns)
    X = df.values.astype(np.float32)

    if fit:
        scaler = StandardScaler()
        X = scaler.fit_transform(X).astype(np.float32)
    else:
        if scaler is None:
            raise ValueError("Must provide pre-fitted scaler when fit=False.")
        X = scaler.transform(X).astype(np.float32)

    logger.info("Node features: %d accounts x %d features", X.shape[0], X.shape[1])
    return X, scaler, feature_names


def engineer_edge_features(
    transactions: pd.DataFrame,
    fit: bool = True,
    scaler: Optional[StandardScaler] = None,
    payment_encoder: Optional[LabelEncoder] = None,
    currency_encoder: Optional[LabelEncoder] = None,
    encode_cyclic_time: bool = True,
) -> Tuple[np.ndarray, StandardScaler, Optional[LabelEncoder], Optional[LabelEncoder], List[str]]:
    """Build per-transaction (edge) feature matrix.

    Returns:
        (edge_features, scaler, payment_encoder, currency_encoder, feature_names)
    """
    parts: List[np.ndarray] = []
    names: List[str] = []

    # --- Amount (log1p) ---------------------------------------------------
    if "amount" in transactions.columns:
        amount = np.clip(transactions["amount"].values.astype(np.float64), 0, None)
        parts.append(np.log1p(amount).reshape(-1, 1).astype(np.float32))
        names.append("amount_log1p")

    # --- Cyclic time encoding ---------------------------------------------
    if encode_cyclic_time and "timestamp" in transactions.columns:
        ts = _parse_timestamps(transactions["timestamp"])
        hour = ts.dt.hour.fillna(0).values.astype(np.float32)
        dow = ts.dt.dayofweek.fillna(0).values.astype(np.float32)

        parts.append(np.sin(2 * np.pi * hour / 24.0).reshape(-1, 1))
        names.append("hour_sin")
        parts.append(np.cos(2 * np.pi * hour / 24.0).reshape(-1, 1))
        names.append("hour_cos")
        parts.append(np.sin(2 * np.pi * dow / 7.0).reshape(-1, 1))
        names.append("dow_sin")
        parts.append(np.cos(2 * np.pi * dow / 7.0).reshape(-1, 1))
        names.append("dow_cos")

    # --- Payment format (one-hot) -----------------------------------------
    if "payment_format" in transactions.columns:
        raw = transactions["payment_format"].astype(str).values
        if fit:
            payment_encoder = LabelEncoder()
            encoded = payment_encoder.fit_transform(raw)
        else:
            if payment_encoder is None:
                raise ValueError("Must provide payment_encoder when fit=False.")
            encoded = _safe_transform(payment_encoder, raw, fallback=0)
        n = len(payment_encoder.classes_) if payment_encoder else 0
        parts.append(np.eye(n, dtype=np.float32)[encoded])
        names.extend([f"pmt_{c}" for c in payment_encoder.classes_])

    # --- Currency (one-hot) -----------------------------------------------
    if "currency" in transactions.columns:
        raw = transactions["currency"].astype(str).values
        if fit:
            currency_encoder = LabelEncoder()
            encoded = currency_encoder.fit_transform(raw)
        else:
            if currency_encoder is None:
                raise ValueError("Must provide currency_encoder when fit=False.")
            encoded = _safe_transform(currency_encoder, raw, fallback=0)
        n = len(currency_encoder.classes_) if currency_encoder else 0
        parts.append(np.eye(n, dtype=np.float32)[encoded])
        names.extend([f"cur_{c}" for c in currency_encoder.classes_])

    # --- Amount paid (log1p) — if different from amount received ----------
    if "amount_paid" in transactions.columns:
        paid = np.clip(transactions["amount_paid"].values.astype(np.float64), 0, None)
        parts.append(np.log1p(paid).reshape(-1, 1).astype(np.float32))
        names.append("amount_paid_log1p")

    X = np.hstack(parts).astype(np.float32)

    # --- Scale only the numeric (non-one-hot) columns ---------------------
    n_numeric = _count_numeric(names, encode_cyclic_time,
                               "amount" in transactions.columns,
                               "amount_paid" in transactions.columns)

    if fit:
        scaler = StandardScaler()
        X[:, :n_numeric] = scaler.fit_transform(X[:, :n_numeric]).astype(np.float32)
    else:
        if scaler is None:
            raise ValueError("Must provide scaler when fit=False.")
        X[:, :n_numeric] = scaler.transform(X[:, :n_numeric]).astype(np.float32)

    logger.info("Edge features: %d transactions x %d features", X.shape[0], X.shape[1])
    return X, scaler, payment_encoder, currency_encoder, names


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_timestamps(series: pd.Series) -> pd.Series:
    """Robustly parse a timestamp series to datetime64."""
    ts = pd.to_datetime(series, errors="coerce")
    if ts.notna().all():
        return ts
    return pd.to_datetime(series, unit="s", errors="coerce")


def _extract_entity_type(name: str) -> str:
    """Extract entity type from descriptive name like 'Corporation #33520'."""
    if not isinstance(name, str):
        return "Unknown"
    # Remove trailing ID numbers
    parts = name.split("#")
    return parts[0].strip() if parts else name


def _compute_transaction_stats(txns: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transaction-history statistics per account.

    Returns DataFrame indexed by account_id with:
      degree_out, degree_in, degree_total,
      total_amount_out, total_amount_in, avg_amount_out, avg_amount_in,
      num_counterparties_out, num_counterparties_in.
    """
    stats = {}

    out_grp = txns.groupby("from_account")
    stats["degree_out"] = out_grp.size()
    if "amount" in txns.columns:
        stats["total_amount_out"] = out_grp["amount"].sum()
        stats["avg_amount_out"] = out_grp["amount"].mean()
    stats["num_counterparties_out"] = out_grp["to_account"].nunique()
    out_df = pd.DataFrame(stats)
    out_df.index.name = "account_id"

    in_grp = txns.groupby("to_account")
    in_stats = {}
    in_stats["degree_in"] = in_grp.size()
    if "amount" in txns.columns:
        in_stats["total_amount_in"] = in_grp["amount"].sum()
        in_stats["avg_amount_in"] = in_grp["amount"].mean()
    in_stats["num_counterparties_in"] = in_grp["from_account"].nunique()
    in_df = pd.DataFrame(in_stats)
    in_df.index.name = "account_id"

    result = out_df.join(in_df, how="outer").fillna(0)
    result["degree_total"] = result["degree_out"] + result["degree_in"]
    if "total_amount_out" in result.columns:
        result["total_amount_total"] = (
            result["total_amount_out"] + result["total_amount_in"]
        )
    return result.astype(np.float32)


def _categorical_columns(df: pd.DataFrame) -> List[str]:
    cat = []
    for c in df.columns:
        if df[c].dtype in (object, "string") or df[c].dtype.name == "category":
            cat.append(c)
        elif df[c].nunique() <= 20 and df[c].dtype.kind in ("i", "f"):
            cat.append(c)
    return cat


def _safe_transform(encoder: LabelEncoder, values: np.ndarray, fallback: int = 0) -> np.ndarray:
    """Transform with existing LabelEncoder, mapping unseen classes to fallback."""
    classes = set(encoder.classes_)
    result = np.full(len(values), fallback, dtype=np.int64)
    for i, v in enumerate(values):
        if v in classes:
            result[i] = int(encoder.transform([v])[0])
    return result


def _count_numeric(names: List[str], has_cyclic: bool, has_amount: bool, has_paid: bool) -> int:
    n = 0
    if has_amount:
        n += 1
    if has_cyclic:
        n += 4
    if has_paid:
        n += 1
    return n
