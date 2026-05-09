"""Load and validate raw IBM AML CSV files.

The IBM AML dataset (Altman et al., 2023) provides flat CSV files per variant:

    data/raw/
        <variant>_accounts.csv      e.g. HI-Small_accounts.csv
        <variant>_Trans.csv         e.g. HI-Small_Trans.csv

Schema (HI/LI variants):

    Accounts:  Bank Name | Bank ID | Account Number | Entity ID | Entity Name
    Trans:     Timestamp | From Bank | Account | To Bank | Account |
               Amount Received | Receiving Currency | Amount Paid |
               Payment Currency | Payment Format | Is Laundering

Account identity is composite: "{Bank ID}_{Account Number}".
"""

import logging
from pathlib import Path
from typing import Tuple

import pandas as pd

from src.utils.config import DataConfig

logger = logging.getLogger(__name__)


def load_raw_data(cfg: DataConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load accounts and transactions DataFrames from raw CSV files.

    Args:
        cfg: DataConfig specifying dataset variant and paths.

    Returns:
        (accounts_df, transactions_df) — normalised with canonical column names.

    Raises:
        FileNotFoundError: If the variant CSV files are missing.
    """
    raw_dir = cfg.resolve_raw_dir()

    # Locate CSV files (flat in raw_dir, not in a subdirectory) -----------
    variant = cfg.dataset_variant  # e.g. "HI-Small"
    accounts_path = _find_file(raw_dir, variant, "account")
    transactions_path = _find_file(raw_dir, variant, "Trans")

    if not accounts_path:
        raise FileNotFoundError(
            f"No accounts CSV found for variant '{variant}' in {raw_dir}. "
            f"Expected a file containing both '{variant}' and 'account' in its name."
        )
    if not transactions_path:
        raise FileNotFoundError(
            f"No transactions CSV found for variant '{variant}' in {raw_dir}. "
            f"Expected a file containing both '{variant}' and 'Trans' in its name."
        )

    accounts = pd.read_csv(accounts_path)
    transactions = pd.read_csv(transactions_path)

    logger.info("Loaded accounts: %s (%d rows)", accounts_path.name, len(accounts))
    logger.info("Loaded transactions: %s (%d rows)", transactions_path.name, len(transactions))

    # Normalise to canonical schema ---------------------------------------
    accounts = _normalise_accounts(accounts)
    transactions = _normalise_transactions(transactions)
    _validate(accounts, transactions, variant)

    return accounts, transactions


def _find_file(directory: Path, variant: str, keyword: str) -> Path | None:
    """Find a CSV whose name contains both *variant* and *keyword*."""
    for p in directory.glob("*.csv"):
        name = p.name
        if variant in name and keyword.lower() in name.lower():
            return p
    return None


# ---------------------------------------------------------------------------
# Schema normalisation — IBM AML → canonical columns
# ---------------------------------------------------------------------------


def _normalise_accounts(df: pd.DataFrame) -> pd.DataFrame:
    """Convert IBM AML accounts schema to canonical form.

    Input:  Bank Name, Bank ID, Account Number, Entity ID, Entity Name
    Output: account_id (= "BANKID_ACCOUNTNUM"), bank_id, account_number,
            entity_id, entity_name, bank_name
    """
    # Some variants may use different casing
    col_map = {}
    for c in df.columns:
        c_lower = c.strip().lower().replace(" ", "_")
        col_map[c] = c_lower

    df = df.rename(columns=col_map)

    if "bank_id" in df.columns and "account_number" in df.columns:
        df["account_id"] = (
            df["bank_id"].astype(str) + "_" + df["account_number"].astype(str)
        )
    elif "bank_id" in df.columns and "account" in df.columns:
        df["account_id"] = (
            df["bank_id"].astype(str) + "_" + df["account"].astype(str)
        )
    else:
        raise ValueError(
            f"Cannot build composite account_id. Columns: {list(df.columns)}"
        )

    # Keep bank_name and entity_name as categorical features for later
    canonical = ["account_id", "bank_id", "account_number", "entity_id", "entity_name"]
    for c in df.columns:
        if c not in canonical and c not in {"bank_name"}:
            canonical.append(c)

    present = [c for c in canonical if c in df.columns]
    return df[present]


def _normalise_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Convert IBM AML transaction schema to canonical form.

    Input has: Timestamp, From Bank, Account (from), To Bank, Account (to),
               Amount Received, Receiving Currency, Amount Paid,
               Payment Currency, Payment Format, Is Laundering

    Pandas renames the second 'Account' column to 'Account.1'.
    We build composite from_account / to_account IDs.
    """
    # Normalise column names: lowercase, underscore
    col_map = {}
    for c in df.columns:
        stripped = c.strip()
        # Handle pandas deduplication: "Account.1" → "account_to"
        lower = stripped.lower().replace(" ", "_")
        if lower == "account.1":
            lower = "account_to"
        col_map[c] = lower
    df = df.rename(columns=col_map)

    # Build composite account IDs -----------------------------------------
    if "from_bank" in df.columns and "account" in df.columns:
        df["from_account"] = (
            df["from_bank"].astype(str) + "_" + df["account"].astype(str)
        )
        df = df.drop(columns=["from_bank", "account"])
    else:
        raise ValueError(f"Missing from_bank/account columns. Got: {list(df.columns)}")

    if "to_bank" in df.columns and "account_to" in df.columns:
        df["to_account"] = (
            df["to_bank"].astype(str) + "_" + df["account_to"].astype(str)
        )
        df = df.drop(columns=["to_bank", "account_to"])
    else:
        raise ValueError(f"Missing to_bank/account_to columns. Got: {list(df.columns)}")

    # Rename remaining columns to canonical names -------------------------
    rename = {
        "timestamp": "timestamp",
        "amount_received": "amount",
        "receiving_currency": "currency",
        "amount_paid": "amount_paid",
        "payment_currency": "payment_currency",
        "payment_format": "payment_format",
        "is_laundering": "is_laundering",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    # Ensure label is integer ---------------------------------------------
    if "is_laundering" in df.columns:
        df["is_laundering"] = df["is_laundering"].astype(int)

    # Reorder for readability ---------------------------------------------
    order = [
        "timestamp", "from_account", "to_account", "amount",
        "currency", "amount_paid", "payment_currency",
        "payment_format", "is_laundering",
    ]
    present = [c for c in order if c in df.columns]
    extra = [c for c in df.columns if c not in present]
    return df[present + extra]


def _validate(accounts: pd.DataFrame, transactions: pd.DataFrame, variant: str) -> None:
    """Check data integrity after normalisation."""
    for col in ["account_id", "from_account", "to_account", "is_laundering"]:
        if col not in transactions.columns and col != "account_id":
            raise ValueError(f"Required column '{col}' missing. Got: {list(transactions.columns)}")
    if "account_id" not in accounts.columns:
        raise ValueError(f"Required column 'account_id' missing. Got: {list(accounts.columns)}")
    if "is_laundering" not in transactions.columns:
        raise ValueError(f"Label column missing. Got: {list(transactions.columns)}")

    # Orphan accounts (in transactions but not in accounts file) ----------
    acct_ids = set(accounts["account_id"])
    txn_senders = set(transactions["from_account"])
    txn_receivers = set(transactions["to_account"])
    txn_ids = txn_senders | txn_receivers
    orphans = txn_ids - acct_ids

    if orphans:
        logger.warning(
            "%d accounts in transactions are missing from accounts.csv — "
            "adding them with NaN features.", len(orphans),
        )
        missing = pd.DataFrame({"account_id": sorted(orphans)})
        accounts = pd.concat([accounts, missing], ignore_index=True)

    # Class imbalance -----------------------------------------------------
    n_total = len(transactions)
    n_laundering = int(transactions["is_laundering"].sum())
    ratio = n_laundering / n_total if n_total > 0 else 0.0
    logger.info(
        "Class distribution [%s]: %d/%d laundering (%.4f%%), pos/neg ratio = %.6f",
        variant, n_laundering, n_total, 100.0 * ratio, ratio,
    )
