"""Exploratory Data Analysis for the IBM AML HI-Small dataset.

Rebuilt for the resubmission to empirically characterise the dataset BEFORE any
model is applied, addressing the requirement to establish that the data exhibits
the graph structure and temporal dynamics the study relies on.

Covers, using ONLY the transactions and accounts files (no external labels):
  1. Class balance.
  2. Graph structure: degree distributions, counterparty spread, hubs,
     connected components (scipy), sampled clustering coefficient.
  3. Structure vs laundering: where laundering accounts sit structurally.
  4. Temporal distribution: transaction volume and laundering prevalence over time.
  5. Feature distributions: laundering vs legitimate (amount, payment, currency, time).
  6. Typology signatures detectable in the laundering subgraph: fan-out, fan-in,
     layering/chains (pass-through accounts), and structuring (small-amount splitting).

Outputs figures to results/eda/*.png and a machine-readable summary to
results/eda/eda_stats.json for use in the report tables.

Deterministic: fixed seed (42) for all sampling.

Usage:
    python experiments/run_eda.py --variant HI-Small
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, ".")

from src.data.loader import load_raw_data
from src.data.graph_constructor import _parse_timestamp
from src.utils.config import DataConfig

SEED = 42
OUT = Path("results/eda")
OUT.mkdir(parents=True, exist_ok=True)

COLOR_LEGIT = "#4C72B0"
COLOR_LAUNDER = "#C44E52"
DPI = 150

# Thresholds used to *count* typology instances (reported explicitly, not tuned).
FANOUT_MIN = 3          # a laundering source reaching >= this many distinct dsts
FANIN_MIN = 3           # a laundering dst reached from >= this many distinct srcs
STRUCTURING_AMOUNT = 10_000.0   # "small" amount ceiling for structuring
STRUCTURING_MIN_TX = 3          # >= this many small laundering sends to distinct dsts

stats: dict = {}


def savefig(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}")


def main() -> None:
    ap = argparse.ArgumentParser(description="EDA for IBM AML")
    ap.add_argument("--variant", type=str, default="HI-Small")
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)

    # ---- Load -------------------------------------------------------------
    cfg = DataConfig(dataset_variant=args.variant)
    accounts, txn = load_raw_data(cfg)
    n_txn = len(txn)
    n_pos = int(txn["is_laundering"].sum())
    n_acc = len(accounts)
    prevalence = n_pos / n_txn

    # Robust timestamp parse (same logic as the pipeline) -> datetime.
    epoch = _parse_timestamp(txn["timestamp"])          # float64 unix seconds
    ts = pd.to_datetime(epoch, unit="s")
    span_days = (epoch.max() - epoch.min()) / 86400.0

    stats["dataset"] = {
        "accounts": n_acc, "transactions": n_txn, "laundering": n_pos,
        "prevalence": prevalence, "span_days": round(float(span_days), 2),
    }
    print(f"Accounts={n_acc:,} | Transactions={n_txn:,} | "
          f"Laundering={n_pos:,} ({prevalence:.4%}) | Span={span_days:.1f} days")

    is_l = txn["is_laundering"].to_numpy().astype(bool)

    # ---- 1. Class balance -------------------------------------------------
    fig, ax = plt.subplots(figsize=(5, 3.5), dpi=DPI)
    counts = [n_txn - n_pos, n_pos]
    ax.bar(["Legitimate", "Laundering"], counts, color=[COLOR_LEGIT, COLOR_LAUNDER])
    ax.set_yscale("log")
    ax.set_ylabel("Transactions (log scale)")
    ax.set_title(f"Class imbalance (laundering = {prevalence:.3%})")
    for i, v in enumerate(counts):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=8)
    savefig(fig, "01_class_balance.png")

    # ---- Degrees / counterparties (vectorised) ----------------------------
    out_deg = txn.groupby("from_account").size()
    in_deg = txn.groupby("to_account").size()
    out_cp = txn.groupby("from_account")["to_account"].nunique()
    in_cp = txn.groupby("to_account")["from_account"].nunique()
    total_deg = out_deg.add(in_deg, fill_value=0)
    total_cp = out_cp.add(in_cp, fill_value=0)

    stats["degree"] = {
        "median_total_degree": float(total_deg.median()),
        "mean_total_degree": float(total_deg.mean()),
        "max_total_degree": int(total_deg.max()),
        "p99_total_degree": float(total_deg.quantile(0.99)),
        "accounts_1_or_2_counterparties_pct":
            float((total_cp <= 2).mean() * 100),
        "accounts_ge_10_counterparties_pct":
            float((total_cp >= 10).mean() * 100),
    }

    # ---- 2. Graph structure: degree + counterparty spread -----------------
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), dpi=DPI)
    for series, lab, c in [(out_deg, "out-degree", COLOR_LEGIT),
                           (in_deg, "in-degree", COLOR_LAUNDER)]:
        vals, cnts = np.unique(series.to_numpy(), return_counts=True)
        axes[0].loglog(vals, cnts, ".", markersize=3, label=lab, color=c)
    axes[0].set_xlabel("degree"); axes[0].set_ylabel("count of accounts")
    axes[0].set_title("Degree distribution (log-log)"); axes[0].legend(fontsize=8)

    cp_vals = total_cp.to_numpy()
    axes[1].hist(np.clip(cp_vals, 0, 50), bins=50, color=COLOR_LEGIT)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("distinct counterparties (clipped at 50)")
    axes[1].set_ylabel("accounts (log)")
    axes[1].set_title("Counterparty spread per account")
    savefig(fig, "02_degree_counterparties.png")

    # ---- Connected components (scipy) + sampled clustering ----------------
    codes_from, uniques = pd.factorize(
        pd.concat([txn["from_account"], txn["to_account"]], ignore_index=True))
    N = len(uniques)
    half = len(txn)
    u = codes_from[:half]
    v = codes_from[half:]
    A = sp.coo_matrix(
        (np.ones(2 * half, dtype=np.int8),
         (np.concatenate([u, v]), np.concatenate([v, u]))),
        shape=(N, N)).tocsr()
    A.sum_duplicates()
    A.data[:] = 1  # binarise (collapse multi-edges)

    n_comp, labels = connected_components(A, directed=False)
    comp_sizes = np.bincount(labels)
    giant = int(comp_sizes.max())

    # Standard average local clustering coefficient, estimated over a uniform
    # random sample of ALL nodes (degree < 2 contributes 0, per convention).
    # For hub nodes clustering is estimated from a random subset of neighbours
    # to bound cost (an unbiased estimator of the node's local clustering).
    indptr, indices = A.indptr, A.indices
    samp = rng.choice(N, size=int(min(5000, N)), replace=False)
    cc = []
    for node in samp:
        nbrs = indices[indptr[node]:indptr[node + 1]]
        k = len(nbrs)
        if k < 2:
            cc.append(0.0)
            continue
        if k > 200:
            nbrs = rng.choice(nbrs, size=200, replace=False)
            k = 200
        links = 0
        for a in nbrs:
            a_nbrs = indices[indptr[a]:indptr[a + 1]]
            links += int(np.isin(a_nbrs, nbrs).sum())
        cc.append(links / (k * (k - 1)))
    avg_clustering = float(np.mean(cc))

    stats["structure"] = {
        "graph_nodes": int(N),
        "connected_components": int(n_comp),
        "giant_component_nodes": giant,
        "giant_component_pct": round(100 * giant / N, 2),
        "avg_clustering_sampled": round(avg_clustering, 4),
        "clustering_sample_size": int(len(samp)),
    }
    print(f"  components={n_comp:,} | giant={giant:,} "
          f"({100*giant/N:.1f}%) | clustering~{avg_clustering:.4f}")

    fig, ax = plt.subplots(figsize=(5, 3.5), dpi=DPI)
    sizes_sorted = np.sort(comp_sizes)[::-1]
    ax.loglog(np.arange(1, len(sizes_sorted) + 1), sizes_sorted, ".",
              markersize=3, color=COLOR_LEGIT)
    ax.set_xlabel("component rank"); ax.set_ylabel("component size")
    ax.set_title(f"Connected components (giant = {100*giant/N:.1f}% of nodes)")
    savefig(fig, "03_components.png")

    # ---- 3. Structure vs laundering ---------------------------------------
    l_txn = txn[is_l]
    l_accts = pd.Index(pd.unique(
        pd.concat([l_txn["from_account"], l_txn["to_account"]], ignore_index=True)))
    is_l_acct = total_deg.index.isin(l_accts)
    deg_l = total_deg[is_l_acct].to_numpy()
    deg_n = total_deg[~is_l_acct].to_numpy()
    cp_l = total_cp[is_l_acct].to_numpy()
    cp_n = total_cp[~is_l_acct].to_numpy()

    stats["structure_vs_laundering"] = {
        "n_laundering_accounts": int(len(l_accts)),
        "median_degree_laundering": float(np.median(deg_l)),
        "median_degree_normal": float(np.median(deg_n)),
        "median_counterparties_laundering": float(np.median(cp_l)),
        "median_counterparties_normal": float(np.median(cp_n)),
    }

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), dpi=DPI)
    axes[0].boxplot([np.log10(deg_n + 1), np.log10(deg_l + 1)],
                    tick_labels=["Normal", "Laundering"], showfliers=False)
    axes[0].set_ylabel("log10(total degree + 1)")
    axes[0].set_title("Degree: laundering vs normal accounts")
    axes[1].boxplot([np.log10(cp_n + 1), np.log10(cp_l + 1)],
                    tick_labels=["Normal", "Laundering"], showfliers=False)
    axes[1].set_ylabel("log10(counterparties + 1)")
    axes[1].set_title("Counterparties: laundering vs normal")
    savefig(fig, "04_structure_vs_laundering.png")

    # ---- 4. Temporal distribution -----------------------------------------
    day = ((epoch - epoch.min()) // 86400).astype(int)
    df_t = pd.DataFrame({"day": day, "is_l": is_l})
    vol = df_t.groupby("day").size()
    rate = df_t.groupby("day")["is_l"].mean() * 100

    stats["temporal"] = {
        "laundering_rate_first_day_pct": float(rate.iloc[0]),
        "laundering_rate_last_day_pct": float(rate.iloc[-1]),
        "peak_day_rate_pct": float(rate.max()),
    }

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), dpi=DPI)
    # Left: volume per day on a log scale, so the low-volume tail (days 10+,
    # hundreds of transactions) remains visible next to the millions in days 0-9.
    axes[0].bar(vol.index, vol.values, color=COLOR_LEGIT)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("day"); axes[0].set_ylabel("transactions/day (log)")
    axes[0].set_title("Transaction volume over time")
    # Right: laundering rate per day (line) with daily volume overlaid on a
    # secondary log axis (grey bars), making it explicit that the high late-period
    # rates are computed over a negligible number of transactions.
    ax2 = axes[1].twinx()
    ax2.bar(vol.index, vol.values, color="0.82", zorder=0)
    ax2.set_yscale("log")
    ax2.set_ylabel("transactions/day (log)", color="0.55")
    ax2.tick_params(axis="y", labelcolor="0.55")
    axes[1].set_zorder(ax2.get_zorder() + 1)   # keep the rate line in front
    axes[1].patch.set_visible(False)           # let the volume bars show through
    axes[1].plot(rate.index, rate.values, marker="o", color=COLOR_LAUNDER, zorder=3)
    axes[1].axhline(prevalence * 100, ls="--", c="black", lw=0.8,
                    label=f"overall {prevalence*100:.2f}%")
    axes[1].set_xlabel("day"); axes[1].set_ylabel("laundering rate (%)")
    axes[1].set_title("Laundering rate vs daily volume"); axes[1].legend(fontsize=8, loc="center left")
    savefig(fig, "05_temporal.png")

    # ---- 5. Feature distributions: laundering vs legitimate ----------------
    amt = txn["amount"].to_numpy(dtype=np.float64)
    amt_l = np.clip(amt[is_l], 1, None)
    amt_n = np.clip(amt[~is_l], 1, None)
    stats["amount"] = {
        "median_laundering": float(np.median(amt_l)),
        "median_legit": float(np.median(amt_n)),
    }

    pmt = pd.crosstab(txn["payment_format"], is_l, normalize="index")
    if True in pmt.columns:
        pmt_rate = (pmt[True] * 100).sort_values(ascending=False)
    else:
        pmt_rate = pd.Series(dtype=float)
    stats["payment_format_laundering_rate_pct"] = {
        str(k): round(float(v), 4) for k, v in pmt_rate.items()}

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), dpi=DPI)
    bins = np.logspace(0, 7, 60)
    axes[0].hist(amt_n, bins=bins, density=True, alpha=0.6,
                 color=COLOR_LEGIT, label="Legitimate")
    axes[0].hist(amt_l, bins=bins, density=True, alpha=0.7,
                 color=COLOR_LAUNDER, label="Laundering")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("amount (log)"); axes[0].set_ylabel("density")
    axes[0].set_title("Amount by class"); axes[0].legend(fontsize=8)
    if not pmt_rate.empty:
        axes[1].barh(pmt_rate.index[::-1], pmt_rate.values[::-1], color=COLOR_LAUNDER)
        axes[1].axvline(prevalence * 100, ls="--", c="black", lw=0.8)
        axes[1].set_xlabel("laundering rate (%)")
        axes[1].set_title("Laundering rate by payment format")
    savefig(fig, "06_feature_distributions.png")

    # ---- 6. Typology signatures in the laundering subgraph ----------------
    l_out_cp = l_txn.groupby("from_account")["to_account"].nunique()
    l_in_cp = l_txn.groupby("to_account")["from_account"].nunique()
    l_senders = set(l_txn["from_account"])
    l_receivers = set(l_txn["to_account"])
    passthrough = l_senders & l_receivers   # receive laundering AND send it on -> layering

    small = l_txn[l_txn["amount"] <= STRUCTURING_AMOUNT]
    small_cp = small.groupby("from_account")["to_account"].nunique()

    stats["typologies"] = {
        "fanout_accounts": int((l_out_cp >= FANOUT_MIN).sum()),
        "fanout_threshold": FANOUT_MIN,
        "max_fanout": int(l_out_cp.max()) if len(l_out_cp) else 0,
        "fanin_accounts": int((l_in_cp >= FANIN_MIN).sum()),
        "fanin_threshold": FANIN_MIN,
        "max_fanin": int(l_in_cp.max()) if len(l_in_cp) else 0,
        "layering_passthrough_accounts": int(len(passthrough)),
        "structuring_accounts": int((small_cp >= STRUCTURING_MIN_TX).sum()),
        "structuring_amount_ceiling": STRUCTURING_AMOUNT,
    }
    print("  typologies:", stats["typologies"])

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), dpi=DPI)
    axes[0].hist(np.clip(l_out_cp.to_numpy(), 0, 30), bins=30,
                 color=COLOR_LAUNDER, alpha=0.8, label="fan-out")
    axes[0].hist(np.clip(l_in_cp.to_numpy(), 0, 30), bins=30,
                 color=COLOR_LEGIT, alpha=0.6, label="fan-in")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("distinct laundering counterparties")
    axes[0].set_ylabel("accounts (log)")
    axes[0].set_title("Fan-out / fan-in in laundering subgraph")
    axes[0].legend(fontsize=8)
    ty = stats["typologies"]
    labels = ["fan-out", "fan-in", "layering\n(pass-through)", "structuring"]
    vals = [ty["fanout_accounts"], ty["fanin_accounts"],
            ty["layering_passthrough_accounts"], ty["structuring_accounts"]]
    axes[1].bar(labels, vals, color=COLOR_LAUNDER)
    axes[1].set_ylabel("accounts")
    axes[1].set_title("Detected typology signatures")
    for i, vv in enumerate(vals):
        axes[1].text(i, vv, f"{vv:,}", ha="center", va="bottom", fontsize=8)
    savefig(fig, "07_typologies.png")

    # ---- Persist stats ----------------------------------------------------
    (OUT / "eda_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"\nStats written to {OUT/'eda_stats.json'}")
    print(f"Figures in {OUT.resolve()}")


if __name__ == "__main__":
    main()
