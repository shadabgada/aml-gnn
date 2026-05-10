"""Exploratory Data Analysis for IBM AML HI-Small dataset.

Produces publication-ready figures characterizing the dataset: class imbalance,
temporal distribution, degree distributions, transaction amounts, payment patterns,
and graph structure. All figures saved to temp/eda/ for inspection.
"""

import os
import sys
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, ".")

from src.data.loader import load_raw_data
from src.utils.config import DataConfig

OUT = Path("temp/eda")
OUT.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="paper", font_scale=1.1, palette="Set2")
COLOR_LEGIT = "#4C72B0"
COLOR_LAUNDER = "#C44E52"
FIGSIZE_WIDE = (7, 3.5)
FIGSIZE_SQ = (5, 4)
DPI = 150


def main():
    # ---- Load --------------------------------------------------------------
    cfg = DataConfig(dataset_variant="HI-Small")
    accounts, txn = load_raw_data(cfg)

    n_txn = len(txn)
    n_laundering = txn["is_laundering"].sum()
    n_accounts = len(accounts)
    prevalence = n_laundering / n_txn
    ts_dt_all = pd.to_datetime(txn["timestamp"], errors="coerce")
    ts_min = ts_dt_all.min()
    ts_max = ts_dt_all.max()
    ts_days = (ts_max - ts_min).total_seconds() / 86400 if ts_min is not pd.NaT else 0

    print(f"Accounts: {n_accounts:,}")
    print(f"Transactions: {n_txn:,}")
    print(f"Laundering: {n_laundering:,} ({100*prevalence:.4f}%)")
    print(f"Timestamp range: {ts_min} to {ts_max} ({ts_days:.1f} days)")

    # ---- 1. Class imbalance ------------------------------------------------
    fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=FIGSIZE_WIDE, dpi=DPI)

    counts = [n_txn - n_laundering, n_laundering]
    labels = ["Legitimate", "Laundering"]
    colors = [COLOR_LEGIT, COLOR_LAUNDER]
    bars = ax1a.bar(labels, counts, color=colors, edgecolor="white", linewidth=0.5)
    ax1a.set_ylabel("Transactions")
    ax1a.set_title("Class Distribution")
    for bar, val in zip(bars, counts):
        ax1a.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + n_txn * 0.01,
                  f"{val:,}\n({100*val/n_txn:.2f}%)", ha="center", fontsize=8)

    ax1b.pie(counts, labels=labels, colors=colors, autopct="%1.3f%%",
             startangle=90, explode=(0, 0.1), textprops={"fontsize": 9})
    ax1b.set_title("Class Proportion")

    fig1.suptitle("Figure 1: Class imbalance in IBM AML HI-Small", y=1.02, fontweight="bold")
    fig1.tight_layout()
    fig1.savefig(OUT / "01_class_imbalance.png", bbox_inches="tight")
    plt.close(fig1)
    print("Saved 01_class_imbalance.png")

    # ---- 2. Temporal distribution ------------------------------------------
    txn_ts = txn.copy()
    txn_ts["timestamp_dt"] = pd.to_datetime(txn_ts["timestamp"], unit="s", errors="coerce")
    txn_ts["hour"] = txn_ts["timestamp_dt"].dt.hour
    txn_ts["day"] = txn_ts["timestamp_dt"].dt.day
    txn_ts["dayofweek"] = txn_ts["timestamp_dt"].dt.dayofweek

    fig2, axes2 = plt.subplots(2, 2, figsize=(8, 6), dpi=DPI)

    # 2a: Hourly volume by class
    for label_val, col, name in [(0, COLOR_LEGIT, "Legitimate"), (1, COLOR_LAUNDER, "Laundering")]:
        subset = txn_ts[txn_ts["is_laundering"] == label_val]
        hourly = subset.groupby("hour").size()
        hourly = hourly.reindex(range(24), fill_value=0)
        axes2[0, 0].plot(hourly.index, hourly.values, color=col, label=name, linewidth=1.2)
    axes2[0, 0].set_xlabel("Hour of Day")
    axes2[0, 0].set_ylabel("Transactions")
    axes2[0, 0].set_title("Hourly Transaction Volume")
    axes2[0, 0].legend(fontsize=7)

    # 2b: Daily volume
    daily = txn_ts.groupby(["day", "is_laundering"]).size().unstack(fill_value=0)
    axes2[0, 1].bar(daily.index, daily.get(0, 0), color=COLOR_LEGIT, label="Legitimate", width=0.8)
    axes2[0, 1].bar(daily.index, daily.get(1, 0), color=COLOR_LAUNDER, label="Laundering",
                    width=0.8, bottom=daily.get(0, 0))
    axes2[0, 1].set_xlabel("Day (within dataset)")
    axes2[0, 1].set_ylabel("Transactions")
    axes2[0, 1].set_title("Daily Volume (stacked)")
    axes2[0, 1].legend(fontsize=7)

    # 2c: Laundering rate by day
    daily_rate = (daily.get(1, 0) / daily.sum(axis=1) * 100)
    axes2[1, 0].bar(daily_rate.index, daily_rate.values, color=COLOR_LAUNDER, width=0.8)
    axes2[1, 0].axhline(y=prevalence * 100, color="black", linestyle="--", linewidth=0.8,
                        label=f"Overall ({prevalence*100:.2f}%)")
    axes2[1, 0].set_xlabel("Day")
    axes2[1, 0].set_ylabel("Laundering Rate (%)")
    axes2[1, 0].set_title("Laundering Rate by Day")
    axes2[1, 0].legend(fontsize=7)

    # 2d: Day of week
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    for label_val, col, name in [(0, COLOR_LEGIT, "Legitimate"), (1, COLOR_LAUNDER, "Laundering")]:
        subset = txn_ts[txn_ts["is_laundering"] == label_val]
        dow_counts = subset.groupby("dayofweek").size().reindex(range(7), fill_value=0)
        axes2[1, 1].bar(np.arange(7) - 0.15 + (0.3 if label_val else 0), dow_counts.values,
                        width=0.3, color=col, label=name)
    axes2[1, 1].set_xticks(range(7))
    axes2[1, 1].set_xticklabels(dow_names, fontsize=8)
    axes2[1, 1].set_ylabel("Transactions")
    axes2[1, 1].set_title("Volume by Day of Week")
    axes2[1, 1].legend(fontsize=7)

    fig2.suptitle("Figure 2: Temporal distribution of transactions", y=1.02, fontweight="bold")
    fig2.tight_layout()
    fig2.savefig(OUT / "02_temporal_distribution.png", bbox_inches="tight")
    plt.close(fig2)
    print("Saved 02_temporal_distribution.png")

    # ---- 3. Transaction amounts --------------------------------------------
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=FIGSIZE_WIDE, dpi=DPI)

    amt_legit = txn.loc[txn["is_laundering"] == 0, "amount"].clip(upper=1e6)
    amt_launder = txn.loc[txn["is_laundering"] == 1, "amount"].clip(upper=1e6)

    bins = np.logspace(0, 6, 80)
    ax3a.hist(amt_legit, bins=bins, color=COLOR_LEGIT, alpha=0.6, label="Legitimate", density=True)
    ax3a.hist(amt_launder, bins=bins, color=COLOR_LAUNDER, alpha=0.7, label="Laundering", density=True)
    ax3a.set_xscale("log")
    ax3a.set_xlabel("Amount (log scale)")
    ax3a.set_ylabel("Density")
    ax3a.set_title("Amount Distribution (log-log)")
    ax3a.legend(fontsize=8)

    ax3b.boxplot([amt_legit, amt_launder], tick_labels=["Legitimate", "Laundering"],
                 patch_artist=True, boxprops={"facecolor": COLOR_LEGIT, "alpha": 0.5},
                 medianprops={"color": "black"})
    ax3b.set_ylabel("Amount")
    ax3b.set_title("Amount by Class")

    amt_legit_med = amt_legit.median()
    amt_launder_med = amt_launder.median()
    print(f"Median amount: legitimate={amt_legit_med:,.0f}, laundering={amt_launder_med:,.0f}")

    fig3.suptitle("Figure 3: Transaction amount distributions", y=1.02, fontweight="bold")
    fig3.tight_layout()
    fig3.savefig(OUT / "03_amount_distribution.png", bbox_inches="tight")
    plt.close(fig3)
    print("Saved 03_amount_distribution.png")

    # ---- 4. Degree distributions -------------------------------------------
    out_deg = txn.groupby("from_account").size()
    in_deg = txn.groupby("to_account").size()

    laundering_out = txn[txn["is_laundering"] == 1].groupby("from_account").size()
    laundering_in = txn[txn["is_laundering"] == 1].groupby("to_account").size()
    laundering_accts = set(laundering_out.index) | set(laundering_in.index)

    fig4, axes4 = plt.subplots(2, 2, figsize=(8, 6), dpi=DPI)

    for ax, deg_series, title in [
        (axes4[0, 0], out_deg, "Out-degree Distribution"),
        (axes4[0, 1], in_deg, "In-degree Distribution"),
    ]:
        values = deg_series.values
        ax.hist(np.log10(values + 1), bins=60, color=COLOR_LEGIT, alpha=0.7, density=True)
        ax.set_xlabel("log10(degree + 1)")
        ax.set_ylabel("Density")
        ax.set_title(title)

    total_deg = out_deg.add(in_deg, fill_value=0)
    all_accts = set(out_deg.index) | set(in_deg.index)
    legit_deg = [total_deg.get(a, 0) for a in all_accts if a not in laundering_accts]
    laund_deg = [total_deg.get(a, 0) for a in all_accts if a in laundering_accts]

    axes4[1, 0].hist([np.log10(np.array(legit_deg) + 1), np.log10(np.array(laund_deg) + 1)],
                     bins=50, color=[COLOR_LEGIT, COLOR_LAUNDER], alpha=0.6,
                     label=["Legitimate", "Laundering"], density=True)
    axes4[1, 0].set_xlabel("log10(total degree + 1)")
    axes4[1, 0].set_ylabel("Density")
    axes4[1, 0].set_title("Total Degree by Class")
    axes4[1, 0].legend(fontsize=7)

    axes4[1, 1].boxplot([legit_deg, laund_deg], tick_labels=["Legitimate", "Laundering"],
                        patch_artist=True,
                        boxprops={"facecolor": COLOR_LEGIT, "alpha": 0.5},
                        medianprops={"color": "black"})
    axes4[1, 1].set_ylabel("Total Degree")
    axes4[1, 1].set_title("Degree by Class (boxplot)")

    print(f"Median total degree: legit={np.median(legit_deg):.1f}, laundering={np.median(laund_deg):.1f}")

    fig4.suptitle("Figure 4: Account degree distributions", y=1.02, fontweight="bold")
    fig4.tight_layout()
    fig4.savefig(OUT / "04_degree_distributions.png", bbox_inches="tight")
    plt.close(fig4)
    print("Saved 04_degree_distributions.png")

    # ---- 5. Payment format preference --------------------------------------
    fig5, (ax5a, ax5b) = plt.subplots(1, 2, figsize=FIGSIZE_WIDE, dpi=DPI)

    pmt_legit = txn[txn["is_laundering"] == 0]["payment_format"].value_counts()
    pmt_launder = txn[txn["is_laundering"] == 1]["payment_format"].value_counts()

    pmt_all = pd.DataFrame({"Legitimate": pmt_legit, "Laundering": pmt_launder}).fillna(0)
    pmt_pct = pmt_all.div(pmt_all.sum(axis=0), axis=1) * 100
    pmt_pct = pmt_pct.sort_values("Laundering", ascending=False)
    pmt_pct.plot(kind="barh", ax=ax5a, color=[COLOR_LEGIT, COLOR_LAUNDER], width=0.8)
    ax5a.set_xlabel("% of Transactions")
    ax5a.set_title("Payment Format Usage by Class")
    ax5a.legend(fontsize=7)

    pmt_launder_rate = pmt_all.sum(axis=1)
    pmt_launder_rate = (pmt_all["Laundering"] / pmt_launder_rate * 100).sort_values(ascending=False)
    ax5b.barh(pmt_launder_rate.index, pmt_launder_rate.values, color=COLOR_LAUNDER)
    ax5b.axvline(x=prevalence * 100, color="black", linestyle="--", linewidth=0.8)
    ax5b.set_xlabel("Laundering Rate (%)")
    ax5b.set_title("Laundering Rate by Payment Format")

    print("Payment format laundering rates:")
    for fmt, rate in pmt_launder_rate.head(5).items():
        print(f"  {fmt}: {rate:.3f}%")

    fig5.suptitle("Figure 5: Payment format analysis", y=1.02, fontweight="bold")
    fig5.tight_layout()
    fig5.savefig(OUT / "05_payment_format.png", bbox_inches="tight")
    plt.close(fig5)
    print("Saved 05_payment_format.png")

    # ---- 6. Graph structure: component analysis ----------------------------
    from collections import defaultdict

    edges = list(zip(txn["from_account"], txn["to_account"]))
    node_to_idx = {}
    idx_to_node = []
    for u, v in edges:
        for node in (u, v):
            if node not in node_to_idx:
                node_to_idx[node] = len(idx_to_node)
                idx_to_node.append(node)
    n_nodes = len(idx_to_node)
    print(f"Graph nodes (accounts with >=1 transaction): {n_nodes:,}")

    adj = defaultdict(set)
    for u, v in edges:
        ui, vi = node_to_idx[u], node_to_idx[v]
        adj[ui].add(vi)
        adj[vi].add(ui)

    visited = set()
    comp_sizes = []
    for node in range(n_nodes):
        if node in visited:
            continue
        stack = [node]
        visited.add(node)
        size = 0
        while stack:
            cur = stack.pop()
            size += 1
            for nb in adj[cur]:
                if nb not in visited:
                    visited.add(nb)
                    stack.append(nb)
        comp_sizes.append(size)

    comp_sizes.sort(reverse=True)
    print(f"Connected components: {len(comp_sizes)}")
    print(f"Giant component: {comp_sizes[0]:,} nodes ({100*comp_sizes[0]/n_nodes:.1f}%)")

    fig6, (ax6a, ax6b) = plt.subplots(1, 2, figsize=FIGSIZE_WIDE, dpi=DPI)

    ax6a.loglog(range(1, len(comp_sizes) + 1), comp_sizes, marker=".", markersize=2,
                linewidth=0.8, color=COLOR_LEGIT)
    ax6a.set_xlabel("Component Rank")
    ax6a.set_ylabel("Component Size")
    ax6a.set_title("Component Size Distribution")

    cumsum = np.cumsum(comp_sizes) / n_nodes * 100
    ax6b.plot(range(1, len(cumsum) + 1), cumsum, color=COLOR_LEGIT, linewidth=1.2)
    ax6b.axhline(y=100, color="black", linestyle="--", linewidth=0.5)
    ax6b.set_xlabel("Number of Components")
    ax6b.set_ylabel("Cumulative Coverage (%)")
    ax6b.set_title("Cumulative Node Coverage by Components")

    fig6.suptitle("Figure 6: Graph connectivity structure", y=1.02, fontweight="bold")
    fig6.tight_layout()
    fig6.savefig(OUT / "06_graph_connectivity.png", bbox_inches="tight")
    plt.close(fig6)
    print("Saved 06_graph_connectivity.png")

    # ---- 7. Laundering edge patterns: fan-in / fan-out ---------------------
    fig7, (ax7a, ax7b) = plt.subplots(1, 2, figsize=FIGSIZE_WIDE, dpi=DPI)

    laundering_txn = txn[txn["is_laundering"] == 1]
    src_laundering = laundering_txn.groupby("from_account").size()
    dst_laundering = laundering_txn.groupby("to_account").size()

    ax7a.hist(src_laundering.values, bins=50, color=COLOR_LAUNDER, alpha=0.7, density=True)
    ax7a.set_xlabel("Laundering Transactions Sent")
    ax7a.set_ylabel("Density")
    ax7a.set_title("Laundering Fan-out per Account")

    ax7b.hist(dst_laundering.values, bins=50, color=COLOR_LAUNDER, alpha=0.7, density=True)
    ax7b.set_xlabel("Laundering Transactions Received")
    ax7b.set_ylabel("Density")
    ax7b.set_title("Laundering Fan-in per Account")

    print(f"Max laundering fan-out: {src_laundering.max()}")
    print(f"Max laundering fan-in: {dst_laundering.max()}")

    fig7.suptitle("Figure 7: Laundering transaction patterns per account", y=1.02, fontweight="bold")
    fig7.tight_layout()
    fig7.savefig(OUT / "07_laundering_patterns.png", bbox_inches="tight")
    plt.close(fig7)
    print("Saved 07_laundering_patterns.png")

    # ---- 8. Entity type analysis -------------------------------------------
    if "entity_name" in accounts.columns:
        accounts["entity_type"] = accounts["entity_name"].apply(
            lambda x: str(x).split("#")[0].strip() if isinstance(x, str) and "#" in x else (
                str(x)[:30] if isinstance(x, str) else "Unknown"
            )
        )

        acct_to_type = accounts.set_index("account_id")["entity_type"].to_dict()
        txn["src_type"] = txn["from_account"].map(acct_to_type).fillna("Orphan")
        txn["dst_type"] = txn["to_account"].map(acct_to_type).fillna("Orphan")

        entity_counts = accounts["entity_type"].value_counts()
        print(f"Entity types: {len(entity_counts)} unique")
        print(entity_counts.head(10).to_string())

        fig8, (ax8a, ax8b) = plt.subplots(1, 2, figsize=FIGSIZE_WIDE, dpi=DPI)

        top_entities = entity_counts.head(8).index
        entity_launder_rates = {}
        for etype in top_entities:
            etype_accts = set(accounts[accounts["entity_type"] == etype]["account_id"])
            etype_txn = txn[(txn["from_account"].isin(etype_accts)) |
                            (txn["to_account"].isin(etype_accts))]
            if len(etype_txn) > 0:
                entity_launder_rates[etype] = etype_txn["is_laundering"].mean() * 100

        et_df = pd.Series(entity_launder_rates).sort_values(ascending=False)
        ax8a.barh(et_df.index, et_df.values, color=COLOR_LAUNDER)
        ax8a.axvline(x=prevalence * 100, color="black", linestyle="--", linewidth=0.8)
        ax8a.set_xlabel("Laundering Rate (%)")
        ax8a.set_title("Laundering Rate by Entity Type (top 8)")

        top_counts = entity_counts.head(8)
        ax8b.barh(top_counts.index, top_counts.values, color=COLOR_LEGIT)
        ax8b.set_xlabel("Number of Accounts")
        ax8b.set_title("Account Count by Entity Type (top 8)")

        fig8.suptitle("Figure 8: Entity type analysis", y=1.02, fontweight="bold")
        fig8.tight_layout()
        fig8.savefig(OUT / "08_entity_types.png", bbox_inches="tight")
        plt.close(fig8)
        print("Saved 08_entity_types.png")

    # ---- Summary statistics -------------------------------------------------
    print("\n" + "=" * 60)
    print("EDA SUMMARY")
    print("=" * 60)
    print(f"Accounts: {n_accounts:,} (nodes)")
    print(f"Transactions: {n_txn:,} (edges)")
    print(f"Laundering prevalence: {prevalence:.4%}")
    print(f"Timestamp span: {ts_days:.1f} days")
    print(f"Connected components: {len(comp_sizes)}")
    print(f"Giant component coverage: {comp_sizes[0]/n_nodes:.1%}")
    print(f"Median degree (all): {np.median(total_deg.values):.1f}")
    print(f"Median degree (laundering): {np.median(laund_deg):.1f}")
    print(f"Median amount: {amt_legit_med:,.0f} (legit) vs {amt_launder_med:,.0f} (launder)")
    print(f"Figures saved to: {OUT.resolve()}")


if __name__ == "__main__":
    main()
