# HI-Small Test Results — Complete Leaderboard (2026-05-10)

## Baselines (no graph structure, threshold = 0.50, random 70/15/15 split)

| Model | AUC-ROC | AUC-PR | Precision | Recall | F1 | Train Time |
|--------|---------|--------|-----------|--------|-----|------------|
| Logistic Regression | 0.9378 | 0.0376 | 0.0135 | 0.9295 | 0.0267 | ~2 min |
| Random Forest | 0.8603 | 0.0619 | 0.0035 | 0.9148 | 0.0070 | ~10 min |
| **XGBoost** | **0.9381** | **0.1511** | 0.0265 | 0.8610 | 0.0514 | ~3 min |

## Static GNNs (random 70/15/15 split, calibrated thresholds)

| Model | Params | AUC-ROC | AUC-PR | Precision | Recall | F1 | Thresh | Train Time |
|--------|--------|---------|--------|-----------|--------|-----|--------|------------|
| **GCN** | 63K | **0.9705** | 0.1882 | 0.1846 | 0.3933 | 0.2513 | 0.7029 | 102 min |
| GAT (1 head) | 64K | 0.9581 | 0.0958 | 0.0539 | 0.5317 | 0.0979 | 0.5544 | 154 min |
| GraphSAGE | 81K | 0.9459 | 0.0420 | 0.0563 | 0.2953 | 0.0946 | 0.4852 | 55 min |

## Temporal GNNs (chronological split — train on past, test on future)

| Model | Params | AUC-ROC | AUC-PR | Precision | Recall | F1 | Thresh | Train Time |
|--------|--------|---------|--------|-----------|--------|-----|--------|------------|
| TemporalGCN | 162K | 0.9570 | 0.0637 | 0.1177 | 0.1563 | 0.1343 | 0.7326 | 65 min |
| EvolveGCN-H | 578K | 0.8972 | 0.0275 | 0.0465 | 0.0982 | 0.0631 | 0.7029 | 50 min |
| EvolveGCN-H (rank=8) | 33M | — | — | — | — | — | — | OOM — too large |
| **TGN (EMA memory)** | 119K | **0.9684** | **0.3195** | **0.4257** | **0.3011** | **0.3527** | 0.4159 | 114 min |

**EvolveGCN-H note:** First attempt with rank=8 produced 33M parameters (GRU hidden state = rank × (in_dim + out_dim) scales quadratically) — OOM on CPU. Reducing rank to fit memory (578K params) allowed training but crippled expressive capacity. Weight-space evolution (GRU on GCN weight matrices) proved inherently less stable than state-space evolution (GRU on node states, as in TemporalGCN). Combined with the 12-snapshot bottleneck, EvolveGCN-H (0.897 AUC-ROC) underperforms both TemporalGCN (0.957) and TGN (0.968). This negative result supports the thesis that snapshot-based temporal GNNs are architecturally insufficient for AML detection.

## TGN per-time-slice performance (12 equal slices of test set, threshold=0.5)

| Slice | AUC-ROC | AUC-PR | Precision | Recall |
|-------|---------|--------|-----------|--------|
| 0 (earliest) | 0.9205 | 0.0502 | 0.0000 | 0.0000 |
| 3 | 0.9280 | 0.0853 | 0.6000 | 0.0405 |
| 6 | 0.9714 | 0.0712 | 0.0000 | 0.0000 |
| 9 | 0.9591 | 0.0769 | 0.1294 | 0.1028 |
| 10 | 0.9563 | 0.1875 | 0.1553 | 0.3617 |
| **11 (latest)** | **0.9732** | **0.4518** | **0.5749** | **0.3300** |

AUC-PR improves 9x from earliest to latest time slice (0.05 → 0.45) — the model gets better as it accumulates interaction history.

**Why does performance improve across slices?** The model weights are frozen at test time — but per-node memory keeps accumulating. Test transactions are processed chronologically. For each one, TGN reads the node's current memory, predicts, then updates memory with that interaction. A node in slice 0 only has memory from the training period. By slice 11, the same node's memory contains training history + all test interactions up to that point. An account that starts layering in slice 5 can't be caught in slice 0 (the behavior hasn't happened yet), but by slice 11 the model has observed its full behavioral arc. A static GCN would show a flat line across slices — the rising curve is direct evidence that per-node memory accumulates useful signal over time.

## Key findings

1. **TGN is the best overall model.** Matches GCN on AUC-ROC (0.968 vs 0.971) on a harder evaluation protocol, and beats it decisively on AUC-PR (0.320 vs 0.188, +70%).
2. **Continuous-time beats snapshot-based temporal.** TGN > TemporalGCN by 5x on AUC-PR (0.320 vs 0.064). Per-transaction temporal granularity captures patterns that 12-snapshot models miss.
3. **Graph structure adds measurable value.** The best non-graph model (XGBoost, AUC-PR=0.151) is significantly below TGN (AUC-PR=0.320).
4. **Temporal generalization confirmed.** TGN's performance improves as it accumulates more history (per-slice AUC-PR: 0.05 → 0.45).
