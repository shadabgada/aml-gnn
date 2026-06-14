# Chapter 4: Results, Analyses and Tool Performance

This chapter presents the empirical results of the three-tier comparative evaluation described in Chapter 3. Results are reported for each model tier in sequence, followed by a cross-model comparison that synthesises the findings across tiers, and an assessment of tool scalability and generalizability. All results are from the IBM AML HI-Small dataset.

**4.1 Baseline Results: Conventional Machine Learning (Tier 1)**

Table 4.1 presents the performance of the three conventional supervised classifiers. These models operate on flat edge feature vectors without access to graph structure or temporal information, establishing the performance floor against which GNN-based models are compared.

**Table 4.1: Conventional ML baseline results (random 70/15/15 split, threshold 0.50).**

| Model | AUC-ROC | AUC-PR | Precision | Recall | F1 |
|-------|---------|--------|-----------|--------|-----|
| XGBoost | 0.9381 | 0.1511 | 0.0265 | 0.8610 | 0.0514 |
| Random Forest | 0.8603 | 0.0619 | 0.0035 | 0.9148 | 0.0070 |
| Logistic Regression | 0.9378 | 0.0376 | 0.0135 | 0.9295 | 0.0267 |

XGBoost is the strongest conventional classifier, achieving AUC-ROC 0.9381 and AUC-PR 0.1511. Logistic Regression matches XGBoost on AUC-ROC (0.9378) but achieves substantially lower AUC-PR (0.0376), indicating that its strong ranking performance does not translate to effective identification of the minority class. At the default 0.5 threshold, Logistic Regression achieves very high recall (0.9295) but near-zero precision (0.0135): it flags nearly all laundering transactions, but at the cost of an overwhelming false positive rate that would be operationally unworkable.

Random Forest underperforms both alternatives across all metrics (AUC-ROC 0.8603, F1 0.0070). The depth limit of 10, applied to prevent overfitting to the majority class, appears to have been too restrictive for the complex decision boundary required under extreme class imbalance. Without the depth constraint, preliminary experiments showed severe overfitting; with it, the model lacks sufficient capacity.

The key insight from the baseline tier is that even the best conventional classifier (XGBoost) achieves precision of only 0.0265 at the default threshold. For every 1,000 transactions flagged as suspicious, approximately 974 are false positives. This reflects the fundamental limitation identified in Section 2.2: without access to relational information, individual transaction features carry limited signal for distinguishing laundering from legitimate activity.

**4.2 Static GNN Results: Graph Structure Without Time (Tier 2)**

Table 4.2 presents the performance of the three static GNN architectures. These models incorporate graph structure through message passing but treat all transactions as simultaneously present, without temporal ordering.

**Table 4.2: Static GNN results (random 70/15/15 split, calibrated thresholds).**

| Model | Params | AUC-ROC | AUC-PR | Precision | Recall | F1 | Thresh |
|-------|--------|---------|--------|-----------|--------|-----|--------|
| GCN | 63K | 0.9705 | 0.1882 | 0.1846 | 0.3933 | 0.2513 | 0.7029 |
| GAT (1 head) | 64K | 0.9581 | 0.0958 | 0.0539 | 0.5317 | 0.0979 | 0.5544 |
| GraphSAGE | 81K | 0.9459 | 0.0420 | 0.0563 | 0.2953 | 0.0946 | 0.4852 |

GCN is the strongest static GNN, achieving AUC-ROC 0.9705 and AUC-PR 0.1882 with only 63,489 parameters. At its calibrated threshold of 0.7029, GCN detects 39.3% of laundering transactions at 18.5% precision. Compared to the best baseline (XGBoost, AUC-PR 0.1511), GCN adds 0.0371 AUC-PR, confirming that graph structural information contributes measurable detection value beyond what flat features provide.

GAT underperforms GCN (AUC-ROC 0.9581, AUC-PR 0.0958) despite its theoretically more expressive attention mechanism. The single-head configuration, necessitated by CPU memory constraints (4 heads caused OOM on the 5-million-edge graph), likely limits the model's capacity to learn multiple relational patterns in parallel. The original GAT paper (Veličković et al., 2018) reported that multi-head attention was important for stable training and performance; the single-head result here is consistent with that finding. This is a hardware constraint rather than an architectural limitation of GAT, and is discussed as a limitation in Section 5.4.

GraphSAGE achieves the lowest static GNN performance (AUC-ROC 0.9459, AUC-PR 0.0420). Mean aggregation with neighbourhood sampling, while computationally efficient, appears to lose discriminative signal. In a graph where laundering accounts are structurally distinctive (high out-degree, unusual counterparty patterns), averaging neighbour features may dilute the very signal the model needs to detect. LSTM or max aggregation might preserve more of this signal, at increased computational cost.

Comparing these results to the original IBM AML dataset paper (Altman et al., 2023), the GCN performance reported here (AUC-ROC 0.9705) is broadly consistent with their findings, though direct numeric comparison is complicated by differences in data split strategy and evaluation protocol.

**4.3 Temporal GNN Results: Graph Structure With Time (Tier 3)**

This section presents results for the three temporal GNN architectures. Unlike the static GNNs in Section 4.2, which were evaluated on a random 70/15/15 split, the temporal models were evaluated on a chronological split: trained on the earliest 70% of transactions, validated on the next 15%, and tested on the latest 15%. This protocol is deployment-realistic (Section 3.3.3) but also inherently harder: the model must detect laundering in a future time period using patterns learned from the past.

**4.3.1 Snapshot-Based Temporal Models**

Table 4.3 presents results for the two snapshot-based temporal architectures.

**Table 4.3: Snapshot temporal GNN results (chronological split, calibrated thresholds).**

| Model | Params | AUC-ROC | AUC-PR | Precision | Recall | F1 | Thresh |
|-------|--------|---------|--------|-----------|--------|-----|--------|
| TemporalGCN | 162K | 0.9570 | 0.0637 | 0.1177 | 0.1563 | 0.1343 | 0.7326 |
| EvolveGCN-H | 578K | 0.8972 | 0.0275 | 0.0465 | 0.0982 | 0.0631 | 0.7029 |

TemporalGCN achieves AUC-ROC 0.9570 with 161,793 parameters. Despite incorporating temporal information through GRU-evolved node states across 12 snapshots, it underperforms the static GCN (AUC-ROC 0.9705, AUC-PR 0.1882). The difference is partly attributable to the harder chronological evaluation protocol, but the magnitude of the gap (0.1245 AUC-PR) suggests that the 12-snapshot temporal resolution is too coarse to capture laundering patterns. Structuring and layering schemes that unfold across individual transactions within a single snapshot window are invisible to the model.

EvolveGCN-H is the weakest GNN across all three tiers (AUC-ROC 0.8972, AUC-PR 0.0275). Two implementations were attempted. With the rank parameter set to 8, the model had 33 million parameters, exceeding available CPU memory. Reducing the rank to 2 produced a 578,369-parameter model that could be trained, but at substantially reduced expressive capacity. The parameter explosion is inherent to the EvolveGCN-H design: the GRU hidden state dimension scales as rank multiplied by (input dimension plus output dimension), growing quadratically with GCN layer dimensions. Even at rank 2, EvolveGCN-H has more parameters than GCN and GAT combined (578K vs 63K + 64K) yet achieves the worst performance, indicating that weight-space evolution is not merely expensive but architecturally unstable for this task.

**4.3.2 Continuous-Time TGN**

Table 4.4 presents results for the continuous-time TGN.

**Table 4.4: TGN results (chronological split, calibrated thresholds).**

| Model | Params | AUC-ROC | AUC-PR | Precision | Recall | F1 | Thresh |
|-------|--------|---------|--------|-----------|--------|-----|--------|
| TGN | 119K | 0.9684 | 0.3195 | 0.4257 | 0.3011 | 0.3527 | 0.4159 |

TGN achieves AUC-ROC 0.9684 and AUC-PR 0.3195 with 119,000 parameters. This is the best overall result across all three tiers. At its calibrated threshold of 0.4159, TGN detects 30.1% of laundering transactions at 42.6% precision: for every 100 alerts generated, approximately 43 are genuine laundering cases.

Two comparisons are essential for interpreting these results. First, TGN's AUC-ROC (0.9684) matches GCN's (0.9705) despite being evaluated on a harder protocol (chronological vs random split). Chronological evaluation prevents the model from seeing future transactions during training, making the task more representative of real deployment. That TGN matches GCN's AUC-ROC under these stricter conditions is a strong result. Second, TGN's AUC-PR (0.3195) is 5.0 times higher than TemporalGCN's (0.0637) under the same chronological evaluation protocol. Since both models are evaluated identically, this gap can be attributed to architectural differences: continuous-time processing with individual timestamps versus coarse snapshot bucketing.

The development of the TGN implementation involved resolving four methodological issues, as documented in Section 3.4.4: a PyG TGNMemory dtype incompatibility, a GRU gradient disconnection, a data leakage bug in which training used updated memory while evaluation used old memory, and the interaction between gradient clipping and class-weighted loss. Each issue was identified, resolved, and documented as a methodological finding. The final configuration, with EMA memory (beta=0.85), disabled gradient clipping, and predictions always computed from old memory state, produced the results reported here.

**4.3.3 TGN Temporal Generalisation: Per-Slice Analysis**

Table 4.5 presents TGN performance across individual time slices of the chronologically ordered test set. The test set (15% of all transactions, approximately 760,000 edges) was divided into 12 equal slices by edge count after chronological sorting, matching the 12-window configuration used for the snapshot-based temporal models to enable a like-for-like temporal resolution comparison. Metrics were computed independently for each slice. This analysis tests whether model performance improves as per-node memory accumulates interaction history.

**Table 4.5: TGN per-slice performance (threshold 0.50, selected slices).**

| Slice | AUC-ROC | AUC-PR |
|-------|---------|--------|
| 0 (earliest test) | 0.9205 | 0.0502 |
| 3 | 0.9280 | 0.0853 |
| 6 | 0.9714 | 0.0712 |
| 9 | 0.9591 | 0.0769 |
| 10 | 0.9563 | 0.1875 |
| 11 (latest test) | 0.9732 | 0.4518 |

AUC-PR improves from 0.0502 in the earliest test slice to 0.4518 in the latest, a factor of 9.0. AUC-ROC improves from 0.9205 to 0.9732. The upward trend is not monotonic across all slices (slice 6 achieves higher AUC-ROC than slice 9, for example), reflecting natural variation in laundering prevalence and difficulty across time windows. The overall trajectory, however, is unambiguously positive.

This improvement is not a training effect. Model weights are frozen at test time. What changes across slices is the content of per-node memory: each transaction processed by the model updates the source and destination accounts' memory states via the EMA mechanism. An account active in slice 0 only has memory accumulated during the training period. By slice 11, the same account's memory encodes information from the training period plus all preceding test-period transactions.

The practical interpretation is that TGN gets better the longer it runs. An account that begins exhibiting laundering behaviour mid-way through the test period cannot be detected in earlier slices (the behaviour has not occurred yet), but by later slices the model has observed the account's full behavioural arc and can recognise the laundering pattern. A static GCN would show a flat performance line across slices, since it has no memory and classifies each edge independently. The rising curve is direct evidence that per-node memory accumulates behaviourally useful signal over time.

**4.4 Cross-Model Comparison**

Table 4.6 presents all nine models in a unified leaderboard, ordered by AUC-PR. The evaluation protocol column is essential for fair comparison: models evaluated on random splits are being tested on an easier task than those evaluated on chronological splits.

**Table 4.6: Complete model leaderboard, ordered by AUC-PR.**

| Tier | Model | Params | AUC-ROC | AUC-PR | F1 | Eval Split |
|------|-------|--------|---------|--------|-----|------------|
| Temporal | TGN | 119K | 0.9684 | 0.3195 | 0.3527 | Chronological |
| Static | GCN | 63K | 0.9705 | 0.1882 | 0.2513 | Random |
| Conv | XGBoost | N/A | 0.9381 | 0.1511 | 0.0514 | Random |
| Static | GAT | 64K | 0.9581 | 0.0958 | 0.0979 | Random |
| Temporal | TemporalGCN | 162K | 0.9570 | 0.0637 | 0.1343 | Chronological |
| Conv | RF | N/A | 0.8603 | 0.0619 | 0.0070 | Random |
| Static | GraphSAGE | 81K | 0.9459 | 0.0420 | 0.0946 | Random |
| Conv | LR | N/A | 0.9378 | 0.0376 | 0.0267 | Random |
| Temporal | EvolveGCN-H | 578K | 0.8972 | 0.0275 | 0.0631 | Chronological |

Several patterns emerge from the cross-model comparison. First, there is a clear three-tier progression in detection quality. The best conventional model (XGBoost, AUC-PR 0.151) establishes a competitive baseline. The best static GNN (GCN, AUC-PR 0.188) adds modest but measurable value. The best temporal GNN (TGN, AUC-PR 0.320) adds substantial value, more than doubling XGBoost's AUC-PR. The progression is consistent: graph structure helps, and fine-grained temporal modelling helps decisively.

Second, temporal modelling is not automatically beneficial. Both snapshot-based temporal models underperform the static GCN on AUC-ROC, and TemporalGCN's AUC-PR (0.064) is lower than XGBoost's (0.151). Temporal information must be modelled at the right granularity to be useful. Coarse snapshot bucketing discards the transaction-level temporal signal that continuous-time modelling preserves.

Third, the evaluation protocol matters for interpreting results. The static GNNs were evaluated on random splits, which mix past and future transactions across training and test sets. As discussed in Section 3.3.3, this inflates performance relative to deployment conditions. The temporal GNNs were evaluated on chronological splits, which are deployment-realistic but harder. TGN's AUC-PR of 0.320 on chronological evaluation is therefore a more honest estimate of real-world performance than GCN's 0.188 on random evaluation. If GCN were evaluated chronologically, its performance would likely be lower, widening the gap between static and continuous-time temporal approaches.

Fourth, parameter count and performance are uncorrelated. EvolveGCN-H has the most parameters (578K) and the worst performance. TGN has fewer parameters (119K) than TemporalGCN (162K) yet achieves 5 times its AUC-PR. Efficient architecture design, not parameter count, determines detection quality.

**4.5 Tool Performance Summary**

**Reliability and reproducibility.** All experiments used a fixed random seed (42) across NumPy, PyTorch, and Python's random module. Data splits are deterministic: chronological sort followed by index-based partitioning. Training procedures do not involve stochastic data augmentation. Under these conditions, re-running any experiment with the same arguments produces identical results. The complete source code, including data loading, feature engineering, graph construction, model implementations, training loops, and evaluation procedures, is available in the project repository. Reproduction commands for each experiment are documented in Appendix B.

**Computational efficiency.** All models were trained on a single CPU (Intel Core i7, 8 threads). Total computational investment across all nine models was approximately 9 CPU-hours. The longest training runs are dominated by graph convolution operations on the full 5-million-edge graph. GraphSAGE's neighbourhood sampling and TGN's batched edge processing are inherently more scalable to larger graphs than full-batch GCN and GAT.

**Scalability.** All models were trained on the HI-Small variant (518,581 nodes, 5,078,345 edges). The IBM AML dataset also provides a Medium variant with tens of millions of transactions. While the architectural findings reported here are expected to generalise, since all four variants share an identical data-generating process (Altman et al., 2023), empirical verification on larger variants was not performed due to CPU time constraints and is noted as future work in Section 5.5. The static GNNs use full-batch training, which would require more memory for larger graphs. GraphSAGE's neighbourhood sampling and TGN's batched edge processing are inherently more scalable to larger graphs than full-batch GCN and GAT.

**Parameter efficiency.** TGN demonstrates the best parameter efficiency among GNNs: 119,000 parameters achieving 0.9684 AUC-ROC, compared to GCN's 63,489 parameters achieving 0.9705 (on the easier random split). EvolveGCN-H illustrates the opposite: 578,369 parameters (nearly five times TGN) achieving only 0.8972 AUC-ROC. Parameter efficiency is relevant to deployment scenarios where model size affects inference latency and memory requirements.

**Generalizability.** The per-slice analysis in Section 4.3.3 provides evidence of temporal generalisation: TGN's performance improves as it accumulates interaction history in per-node memory, demonstrating that the model learns transferable behavioural patterns rather than memorising specific transactions. Cross-dataset generalisation to other IBM AML variants was not evaluated and is noted as a limitation in Section 5.4. The model's performance on the held-out test set, which contains transactions from a later time period than the training data, provides within-dataset evidence of generalisation to unseen temporal patterns.
