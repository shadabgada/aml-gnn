# AML GNN — Money Laundering Detection with Graph Neural Networks

Master's thesis by Shadab Gada (HU Utrecht), supervised by Kees van Montfort, PhD.

## Project overview

Comparative evaluation of static and temporal Graph Neural Networks against conventional ML baselines for detecting money laundering in financial transaction networks. Uses the IBM AML HI-Small dataset (~518K accounts, ~5M transactions, 0.1% laundering rate).

**Key result:** TGN (continuous-time temporal GNN) achieves AUC-ROC=0.9684, AUC-PR=0.3195, F1=0.3527 on chronological split — matching static GCN's AUC-ROC on a harder evaluation and beating it on AUC-PR by 70%.

## Quick start (fully reproducible)

All scripts use fixed random seed (42). Data splits are deterministic (chronological sort + index-based). Re-running with the same arguments produces identical results.

```bash
pip install -r requirements.txt

# Baselines (conventional ML — no graph structure)
python experiments/run_baselines.py --variant HI-Small --seed 42

# Static GNNs (random 70/15/15 split)
python experiments/run_gnn.py --variant HI-Small --model gcn --seed 42 --epochs 100
python experiments/run_gnn.py --variant HI-Small --model gat --seed 42 --epochs 100
python experiments/run_gnn.py --variant HI-Small --model sage --seed 42 --epochs 100

# Snapshot temporal GNNs (12 time windows)
python experiments/run_temporal.py --variant HI-Small --model temporal_gcn --seed 42 --epochs 100
python experiments/run_temporal.py --variant HI-Small --model evolve_gcn_h --seed 42 --epochs 100

# TGN (continuous-time — BEST model)
python experiments/run_tgn.py --variant HI-Small --epochs 100 --lr 0.003 --pos_weight_mult 0.01 --grad_clip 0 --seed 42
```

**Critical:** TGN requires `--grad_clip 0` (no gradient clipping). With `pos_weight=12.4`, clipping destroys the minority-class learning signal.

## Architecture

```
transactions.csv ──→ TemporalData (chronological edges) ──→ TGNModel ──→ edge logits
accounts.csv ──┘                                          ├─ EMAMemory (per-node)
                                                          ├─ TimeEncoder (sinusoidal)
                                                          ├─ NodeProjection MLP
                                                          └─ EdgeClassifier MLP
```

## Dataset

IBM AML HI-Small (`data/raw/`):
- `HI-Small_Trans.csv`: 5,078,345 transactions (src, dst, timestamp, amount, payment format, currency, is_laundering)
- `HI-Small_accounts.csv`: 518,581 accounts
- 0.1019% laundering (5,177 positive edges)
- Timestamps span ~18 days in seconds

## Results summary (HI-Small test set)

### Baselines (no graph, threshold 0.5)
| Model | AUC-ROC | AUC-PR | F1 |
|--------|---------|--------|-----|
| XGBoost | 0.9381 | 0.1511 | 0.0514 |
| Logistic Regression | 0.9378 | 0.0376 | 0.0267 |

### Static GNNs (random 70/15/15 split, calibrated threshold)
| Model | Params | AUC-ROC | AUC-PR | F1 |
|--------|--------|---------|--------|-----|
| **GCN** | 63K | **0.9705** | **0.1882** | **0.2513** |
| GAT | 64K | 0.9581 | 0.0958 | 0.0979 |
| GraphSAGE | 81K | 0.9459 | 0.0420 | 0.0946 |

### Temporal GNNs (chronological split — train on past, test on future)
| Model | Params | AUC-ROC | AUC-PR | F1 | Eval Type |
|--------|--------|---------|--------|-----|-----------|
| TemporalGCN | 162K | 0.9570 | 0.0637 | 0.1343 | Snapshot (12 windows) |
| EvolveGCN-H | 578K | 0.8972 | 0.0275 | 0.0631 | Snapshot (12 windows) |
| **TGN** | 119K | **0.9684** | **0.3195** | **0.3527** | Continuous-time (per-edge) |

EvolveGCN-H first attempt (rank=8) had 33M params → OOM. Reduced rank to fit memory crippled expressive capacity. See docs/RESULTS.md for full analysis.

**TGN matches GCN's AUC-ROC on a harder evaluation and beats it on AUC-PR by 70%.** Continuous-time temporal modeling with per-node memory is the best approach for AML detection.

## Three temporal models — architectural distinction

| | TemporalGCN | EvolveGCN-H | TGN |
|---|---|---|---|
| **What evolves** | Per-node hidden states | GCN weight matrices | Per-node EMA memory |
| **Mechanism** | GRU on node state | GRU on low-rank weights | EMA: m_new = β·m_old + (1-β)·msg |
| **Updates per run** | 12 (once per snapshot) | 12 (once per snapshot) | ~5M (every transaction) |
| **Per-node history?** | Coarse behavior summary | None (weights only) | Fine-grained interaction history |
| **Key weakness** | Snapshot granularity loss | Weight instability + param explosion | Data leakage (fixed) |

**Per-slice performance explanation:** TGN's model weights are frozen at test time, but per-node memory keeps accumulating. Test transactions are processed chronologically — each prediction updates the node's memory. Earliest test slice only has training-period memory. By the latest slice, memory contains training + all preceding test interactions. AUC-PR climbs 0.05 → 0.45 because the model observes each account's full behavioral arc over time. A static model would show a flat line — the rising curve is direct evidence that per-node memory works.

## Key design decisions

1. **No gradient clipping** for TGN — `pos_weight=12.4` creates large minority-class gradients; clipping destroys the learning signal
2. **No data leakage** — TGN always predicts with OLD memory (before current batch). msg_proj is trained through the edge classifier input, not through memory
3. **Chronological split** for temporal models — train on past, validate/test on future. This is harder than random split but deployment-realistic
4. **EMA memory** (beta=0.85) instead of GRU — avoids PyG TGNMemory dtype bugs and provides stable training

## Novelty & research contribution

1. **First application of TGN to AML edge classification on a public benchmark.** The IBM AML dataset paper (Altman et al., 2023, NeurIPS) only tested static GNNs. The TGN paper (Rossi et al., 2020) evaluated on Wikipedia/Reddit/Twitter. No published work combines TGN's continuous-time temporal modeling with per-node memory for AML detection.

2. **First systematic 3-tier comparison on a unified AML benchmark.** No published study compares conventional ML → static GNNs → continuous-time temporal GNNs on the same dataset with the same evaluation protocol.

3. **Empirical demonstration that temporal granularity matters.** Snapshot-based temporal models (12 windows) lose fine-grained transaction ordering. Continuous-time TGN achieves 5x higher AUC-PR (0.320 vs 0.064) than snapshot models. This is a novel finding with implications for future AML GNN architecture design.

4. **Deployment-realistic chronological evaluation.** Published AML GNN studies predominantly use random splits (mixing past and future edges). This research evaluates on strict future-edge prediction, providing more honest real-world performance estimates.

5. **Snapshot-based temporal GNNs proven insufficient for AML.** Both TemporalGCN (GRU on node states, 0.957) and EvolveGCN-H (GRU on weights, 0.897) underperform static GCN (0.971) due to the 12-snapshot bottleneck — only 12 state transitions cannot capture transaction-level laundering patterns. EvolveGCN-H additionally suffers from weight-space evolution instability and parameter explosion (33M at rank=8). This negative finding directly motivates continuous-time modeling.

## Known issues & fixes applied

1. **TGNMemory dtype mismatch** (PyG 2.7) — `last_update` buffer is Long, timestamps are Float → replaced with custom EMAMemory
2. **Data leakage in TGN training** — predictions used updated memory during training but old memory during eval → fixed: always use old memory
3. **Gradient clipping destroys learning** — `grad_clip=1.0` with `pos_weight=12.4` clips almost all positive-class gradients → disabled clipping
4. **XGBoost early stopping** — was not configured → fixed
5. **LR/RF double-weighting** — class_weight + sample_weight applied twice → fixed

## File structure

```
src/
├── data/
│   ├── loader.py              — Load raw CSV data
│   ├── feature_engineering.py — Edge/node feature construction
│   ├── graph_constructor.py   — Build PyG Data objects (static)
│   └── temporal_data_builder.py — Build PyG TemporalData (TGN)
├── models/
│   ├── static_gnn.py          — GCN, GAT, GraphSAGE models
│   ├── temporal_gnn.py        — TemporalGCN, EvolveGCN-H models
│   └── tgn_model.py           — TGN with custom EMAMemory
├── training/
│   ├── trainer.py             — Static GNN training loop
│   ├── temporal_trainer.py    — Snapshot-based temporal training
│   └── tgn_trainer.py         — TGN chronological training
└── utils/
    ├── config.py              — Dataset configuration
    ├── metrics.py             — AUC-ROC, AUC-PR, F1, calibration
    └── logger.py              — Logging setup

experiments/
├── run_baselines.py           — LR, RF, XGBoost baselines
├── run_gnn.py                 — Static GNN runner
├── run_temporal.py            — Snapshot temporal runner
├── run_tgn.py                 — TGN runner
├── diagnose_memory.py         — Diagnostic: test EMA beta values
└── debug_trainer.py           — Diagnostic: compare training loops

results/
├── checkpoints/               — Model checkpoints
└── logs/                      — Training logs

docs/
├── RESULTS.md                 — Complete leaderboard with all models
└── THESIS_NARRATIVE.md        — Full research journey, novelty, and conclusions
```
