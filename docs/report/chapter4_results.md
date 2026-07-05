# Chapter 4: Results, Analyses and Tool Performance

This chapter presents the empirical results of the study. It opens with an exploratory characterisation of the dataset (Section 4.1) that establishes, before any model is applied, that the data carries the relational and temporal structure the research relies on. It then reports the three-tier comparative evaluation in sequence: conventional baselines (Section 4.2), static GNNs (Section 4.3), an ablation that isolates the contribution of graph structure (Section 4.4), and temporal GNNs (Section 4.5). A cross-model comparison (Section 4.6), a reliability and sensitivity analysis of the two headline models (Section 4.7), and a tool-performance summary (Section 4.8) close the chapter. Unless stated otherwise, all models are trained and evaluated on the IBM AML HI-Small dataset under a single, uniform chronological split (Section 3.3.3): the earliest 70% of transactions for training, the next 15% for validation, and the latest 15% for testing. All results are deterministic under a fixed seed (42).

**4.1 Exploratory Data Analysis and Dataset Characterisation**

Before any model is applied, the dataset is characterised empirically to establish that it exhibits the relational and temporal structure this study relies on. The analyses use the transactions and accounts files; the supporting figures are provided in Appendix G (Figures G.1 to G.7), and the statistics are produced deterministically by the EDA script (seed 42). Table 4.1 collects the headline figures. The dataset is severely imbalanced, laundering comprising just 0.102% of transactions (Figure G.1).

**Table 4.1: Dataset characterisation summary (HI-Small).**

| Property | Value |
|----------|-------|
| Accounts / transactions | 518,581 / 5,078,345 |
| Laundering transactions (prevalence) | 5,177 (0.102%) |
| Time span | 17.7 days |
| Median / mean / max account degree | 6 / 19.7 / 169,756 |
| Accounts with <=2 counterparties | 39.6% |
| Giant connected component | 72.2% of active accounts |
| Avg. local clustering (sampled, n=5,000) | 0.58 |
| Median degree: laundering vs other | 22 vs 6 |
| Median counterparties: laundering vs other | 6 vs 3 |
| Median amount: laundering vs legitimate | $8,667 vs $1,408 |
| Laundering rate, ACH channel | 0.75% (7x base rate) |

**Graph structure.** The transaction graph is far from a uniform mesh of independent edges. Account degree is heavily right-skewed: the median account participates in 6 transactions, the mean in 20, and the most active account in 169,756. Roughly 40% of accounts transact with only one or two distinct counterparties, while 4% transact with ten or more, a long-tailed, hub-and-spoke topology characteristic of real financial networks (Figure G.2). The graph is also well connected: of 515,088 accounts with at least one transaction, 72.2% lie in a single giant connected component, and the estimated average local clustering coefficient (computed on a random sample of nodes) is approximately 0.58, indicating pronounced local density (Figure G.3). This connectivity is a prerequisite for message passing to propagate information usefully; a fragmented graph would starve a GNN of neighbourhood context.

**Structure versus laundering.** Critically for the premise of this study, laundering-involved accounts occupy structurally distinctive positions (Figure G.4). Their median total degree is 22, against 6 for other accounts, and they transact with a median of 6 distinct counterparties against 3. Laundering activity is not distributed uniformly across the graph but concentrated in higher-degree, more connected accounts. This is direct empirical evidence that account-level graph position carries signal for laundering detection, and it motivates architectures that exploit that position rather than treating transactions in isolation.

**Temporal distribution.** Transaction volume and laundering prevalence are not stationary over the eighteen-day window (Figure G.5). The overwhelming majority of activity falls in the first ten days, during which the laundering rate stays between roughly 0.03% and 0.21%; volume then collapses to a few hundred transactions per day, and within this small tail (around 1,100 transactions in total) the laundering rate rises sharply to about 60%. Because the chronological split partitions by transaction count, this laundering-dense tail is concentrated in the latest portion of the test set. This has two consequences exploited elsewhere: it makes the chronological split a distribution-shifted, and therefore harder, evaluation (Section 3.3.3), and it means any rise in precision-recall performance across later test slices must be read against the concurrent rise in prevalence rather than attributed solely to a model effect (Section 4.5.3).

**Feature distributions.** Several individual features separate the classes even before relational modelling (Figure G.6). Laundering transactions are materially larger (median amount roughly $8,667 against $1,408 for legitimate transactions), and payment channel is informative: ACH transactions carry a laundering rate of 0.75%, more than seven times the overall rate, while other channels sit at or below the base rate. This explains why flat-feature baselines achieve non-trivial ranking performance (Section 4.2), and equally why they are insufficient: individual-transaction features capture part of the signal, but not the relational and temporal structure documented above.

**Typology signatures.** Finally, the laundering typologies described in Chapter 2 are empirically detectable in the transaction graph (Figure G.7). Within the laundering-labelled subgraph, 130 accounts exhibit a fan-out signature (laundering funds sent to three or more distinct destinations, up to 236), 111 exhibit fan-in, 1,003 accounts act as pass-through intermediaries that both receive and forward laundering funds (the structural hallmark of layering), and 84 accounts show structuring behaviour (three or more small transfers, each at or below $10,000, to distinct destinations). These confirm that the FATF typologies motivating this study are present and detectable in the data, not merely asserted from the literature.

**4.2 Baseline Results: Conventional Machine Learning (Tier 1)**

Table 4.2 presents the test set performance of the three conventional supervised classifiers. These models operate on flat edge feature vectors without access to graph structure or temporal information, establishing the performance floor against which GNN-based models are compared. Training and validation set results are provided in Appendix F.

**Table 4.2: Conventional ML baseline results on the test set (chronological split, default threshold 0.50).**

| Model | AUC-ROC | AUC-PR | Precision | Recall | F1 |
|-------|---------|--------|-----------|--------|-----|
| XGBoost | 0.9393 | 0.1460 | 0.0245 | 0.8706 | 0.0476 |
| Random Forest | 0.9409 | 0.1249 | 0.0376 | 0.7687 | 0.0717 |
| Logistic Regression | 0.9378 | 0.0378 | 0.0135 | 0.9295 | 0.0267 |

XGBoost is the strongest conventional classifier on AUC-PR (0.1460), the metric of primary interest under extreme class imbalance, followed closely by Random Forest (0.1249). Logistic Regression matches both on AUC-ROC (0.9378) but achieves far lower AUC-PR (0.0378), indicating that its strong ranking performance does not translate into effective identification of the minority class. At the default 0.5 threshold, Logistic Regression achieves very high recall (0.9295) at near-zero precision (0.0135): it flags almost all laundering transactions, but at a false-positive rate that would be operationally unworkable. Thresholds calibrated for F1 on the validation set improve the operating point of all three (for example, XGBoost reaches F1 0.161 at threshold 0.94, and Random Forest F1 0.186 at threshold 0.90), but do not change the ranking (Appendix F).

The two tree ensembles are close because both can exploit the non-linear interactions in the 28-dimensional edge feature vector (amount, payment format, currency, and temporal encodings), whereas the linear model cannot. The key insight from the baseline tier is nonetheless the ceiling it exposes: even the best conventional classifier reaches only AUC-PR 0.146, and at the default threshold XGBoost flags roughly 40 false positives for every genuine alert. This reflects the limitation identified in Section 2.2: without access to relational information, individual transaction features carry only partial signal for distinguishing laundering from legitimate activity.

**4.3 Static GNN Results: Graph Structure Without Time (Tier 2)**

Table 4.3 presents the test set performance of the three static GNN architectures. These models incorporate graph structure through message passing but treat all transactions as simultaneously present, without temporal ordering. They are evaluated on the same chronological split as every other model, so their numbers are directly comparable to the baselines above and the temporal models below.

**Table 4.3: Static GNN results on the test set (chronological split, calibrated thresholds).**

| Model | Params | AUC-ROC | AUC-PR | Precision | Recall | F1 | Thresh |
|-------|--------|---------|--------|-----------|--------|-----|--------|
| GCN | 63K | 0.9708 | 0.2056 | 0.1214 | 0.5234 | 0.1971 | 0.6732 |
| GAT (1 head) | 64K | 0.9575 | 0.0912 | 0.0487 | 0.5766 | 0.0898 | 0.5247 |
| GraphSAGE | 81K | 0.9452 | 0.0412 | 0.0541 | 0.2972 | 0.0915 | 0.4951 |

GCN is the strongest static GNN, achieving AUC-ROC 0.9708 and AUC-PR 0.2056 with only 63,489 parameters. At its calibrated threshold of 0.673, GCN detects 52.3% of laundering transactions at 12.1% precision. Compared to the best baseline (XGBoost, AUC-PR 0.1460), GCN adds 0.0596 AUC-PR, a 41% relative improvement, confirming that graph structural information contributes measurable detection value beyond what flat features provide. Section 4.4 isolates this contribution more directly through an ablation.

GraphSAGE achieves the lowest static GNN performance (AUC-ROC 0.9452, AUC-PR 0.0412), below the XGBoost baseline on AUC-PR. Mean aggregation with neighbourhood sampling, while computationally efficient, appears to lose discriminative signal. In a graph where laundering accounts are structurally distinctive (high degree, unusual counterparty patterns, Section 4.1), averaging neighbour features may dilute the very signal the model needs to detect. Max or LSTM aggregation might preserve more of this signal at increased computational cost.

GAT reaches AUC-ROC 0.9575 but an AUC-PR of only 0.0912, below both GCN and the XGBoost baseline (0.1460), though above GraphSAGE. It is evaluated with single-head attention: multi-head attention over the full five-million-edge graph is prohibitively memory-intensive, its cost scaling with the number of edges multiplied by the number of heads, and the single-head form has correspondingly limited capacity to learn multiple relational patterns in parallel, which the original GAT formulation identifies as important (Velickovic et al., 2018). That a memory-bounded single-head GAT underperforms the simpler spectral convolution of GCN is consistent with the view that, at this graph scale, the cost of dense attention is not repaid by a commensurate gain in detection quality. The memory behaviour of attention on large graphs is discussed further in Section 5.4.

Comparing GCN to the original IBM AML dataset paper (Altman et al., 2023), the AUC-ROC reported here (0.9708) is broadly consistent with their findings, though direct numeric comparison is complicated by differences in feature construction and evaluation protocol.

**4.4 Isolating the Contribution of Graph Structure**

A central premise of this thesis is that graph structure adds detection signal beyond what hand-crafted features provide. Section 4.3 shows that the best static GNN outperforms the best flat-feature baseline, but that comparison confounds two differences: the GNN sees the graph, and it also uses a different model class. To isolate the value of graph structure specifically, an ablation was run in which a single model class (gradient-boosted trees, the strongest baseline) is given three progressively richer flat feature sets, and compared against the GCN that uses the same information through message passing. All four settings use the identical chronological split.

**Table 4.4: Graph-versus-features ablation (chronological split). Node features are the 12-dimensional per-account features; edge features are the 28-dimensional per-transaction features; message passing is the GCN of Section 4.3.**

| Setting | Features | AUC-ROC | AUC-PR |
|---------|----------|---------|--------|
| XGBoost, node only | 24 | 0.6887 | 0.0187 |
| XGBoost, edge only | 28 | 0.9393 | 0.1460 |
| XGBoost, edge + node | 52 | 0.9505 | 0.1144 |
| GCN (message passing) | 12 node + 28 edge | 0.9708 | 0.2056 |

Two results stand out. First, the hand-crafted node features (in-degree, out-degree, counterparty counts, and similar account-level summaries) are almost useless to a flat classifier on their own: node-only AUC-PR is 0.019, barely above the 0.001 prevalence floor. Concatenating them onto the edge features does not help and slightly hurts (AUC-PR falls from 0.146 to 0.114), indicating that, as flat inputs, the relational summaries add noise rather than signal for a tree ensemble. Second, the GCN, which consumes the same node and edge features but propagates them along the graph through message passing, reaches AUC-PR 0.206, comfortably above every flat configuration. The value of the graph is therefore not in the relational features as static numbers; it is in the message-passing operation that lets each account's representation absorb information from its neighbourhood. This is the most direct evidence in the study that the graph structure itself, not merely the availability of relational attributes, is what improves detection.

**4.5 Temporal GNN Results: Graph Structure With Time (Tier 3)**

This section presents results for the three temporal GNN architectures. All are evaluated on the same chronological split as the static models: the task is to detect laundering in a future time period using patterns learned from the past.

**4.5.1 Snapshot-Based Temporal Models**

Table 4.5 presents the test set results for the two snapshot-based temporal architectures, which discretise the eighteen-day window into 12 snapshots and evolve either node states (TemporalGCN) or GCN weights (EvolveGCN-H) across them.

**Table 4.5: Snapshot temporal GNN results on the test set (chronological split, calibrated thresholds).**

| Model | Params | AUC-ROC | AUC-PR | Precision | Recall | F1 | Thresh |
|-------|--------|---------|--------|-----------|--------|-----|--------|
| TemporalGCN | 163K | 0.9514 | 0.0604 | 0.1041 | 0.1724 | 0.1298 | 0.8415 |
| EvolveGCN-H | 2.2M | 0.9064 | 0.0504 | 0.1319 | 0.0856 | 0.1038 | 0.8415 |

TemporalGCN achieves AUC-ROC 0.9514 with 162,561 parameters. Despite incorporating temporal information through GRU-evolved node states across 12 snapshots, it underperforms the static GCN (AUC-ROC 0.9708, AUC-PR 0.2056) on the same split. Since the evaluation protocol is now identical, the gap cannot be attributed to an easier test set for the static model; it points instead to the snapshot resolution itself. Structuring and layering schemes that unfold across individual transactions within a single snapshot window are invisible to a model that only observes 12 aggregated states.

EvolveGCN-H is the weakest GNN across all three tiers (AUC-ROC 0.9064, AUC-PR 0.0504). It is also by far the largest model in the study at 2,213,673 parameters, roughly thirty-five times the size of the GCN, yet it delivers the worst GNN performance. This combination of parameter explosion and weak accuracy is the signature of the architecture's design: evolving the GCN weight matrices themselves, rather than node states, produces a large and unstable parameter space whose optimisation does not converge to a competitive solution on this task. The result is a clear negative finding: weight-space evolution is not merely expensive here but architecturally ill-suited to transaction-level AML detection.

**4.5.2 Continuous-Time TGN**

Table 4.6 presents the test set results for the continuous-time TGN, which processes each transaction at its exact timestamp and maintains a per-node memory updated by an exponential moving average (Section 3.4).

**Table 4.6: TGN results on the test set (chronological split), 85,905 parameters. The first two rows use the standard cold-start memory, in which per-node memory begins empty at the start of the test period and updates as transactions are processed; they share the same AUC values and differ only in decision threshold and therefore operating point. The third row does not start from an empty memory: it first replays the training and validation transactions to populate each node's memory, then carries and updates that memory continuously through the test period (Section 4.5.3).**

| Setting | AUC-ROC | AUC-PR | Precision | Recall | F1 |
|---------|---------|--------|-----------|--------|-----|
| Cold-start, threshold 0.50 | 0.9698 | 0.3213 | 0.8436 | 0.1659 | 0.2773 |
| Cold-start, calibrated threshold (0.16) | 0.9698 | 0.3213 | 0.2228 | 0.4215 | 0.2915 |
| Warm-started memory, threshold 0.50 | 0.9601 | 0.2708 | 0.7276 | 0.1454 | 0.2424 |

TGN achieves AUC-ROC 0.9698 and AUC-PR 0.3213 with only 85,905 parameters, the best overall result across all three tiers. AUC-ROC and AUC-PR are threshold-independent, so the two cold-start rows report identical ranking quality and differ only in where the decision threshold is placed: at the default 0.50 threshold precision is 0.844 but recall only 0.166, whereas the validation-calibrated threshold trades precision for recall (0.223 precision at 0.422 recall). The warm-started row uses a different memory regime and is discussed in Section 4.5.3.

Two comparisons are essential. First, TGN's AUC-ROC (0.9698) is statistically indistinguishable from GCN's (0.9708) on the same chronological split, so continuous-time modelling matches the best static model on ranking quality. Second, and decisively, TGN's AUC-PR (0.3213) is more than five times TemporalGCN's (0.0604) and 56% above GCN's (0.2056), under an identical evaluation protocol. Because all models are evaluated the same way, this gap is attributable to architecture: continuous-time processing with per-node memory versus coarse snapshot bucketing (TemporalGCN) or timeless message passing (GCN). Section 4.7 shows this advantage is stable across random seeds.

The TGN configuration follows the design established in Section 3.4: an exponential-moving-average memory (beta 0.85) in place of a gated recurrent memory, no gradient clipping under the class-weighted loss, and predictions computed from each node's pre-update memory state so that a transaction is never scored using information from itself. Section 3.4.4 sets out the rationale for these choices.

**4.5.3 TGN Temporal Generalisation: Per-Slice Analysis**

A natural question is whether TGN's per-node memory measurably improves detection as it accumulates interaction history over the test period. Two effects bear on this and must be separated: the amount of history held in memory, and the changing prevalence of laundering across the test window (Section 4.1).

The test set (the latest 15% of transactions, approximately 761,000 edges) was divided into 12 equal slices by edge count after chronological sorting, matching the 12-window configuration of the snapshot models. Performance was measured under two memory regimes. Under memory reset per slice, per-node memory is cleared at the start of every slice, so no interaction history carries across slice boundaries. Under memory carried continuously, per-node memory is warm-started from the training and validation periods and maintained across the entire test set, so that by later slices it encodes all preceding history. Both regimes are leakage-free: every prediction uses the memory state from before the current transaction. Table 4.7 reports AUC-PR for both regimes alongside the laundering prevalence of each slice.

**Table 4.7: TGN per-slice AUC-PR under memory reset per slice versus memory carried continuously, with laundering prevalence (selected slices).**

| Slice | Prevalence | AUC-PR (reset per slice) | AUC-PR (carried continuously) |
|-------|-----------|--------------------------|-------------------------------|
| 0 (earliest) | 0.068% | 0.023 | 0.095 |
| 2 | 0.077% | 0.047 | 0.073 |
| 4 | 0.091% | 0.048 | 0.215 |
| 6 | 0.080% | 0.055 | 0.118 |
| 8 | 0.060% | 0.015 | 0.018 |
| 9 | 0.169% | 0.036 | 0.103 |
| 10 | 0.296% | 0.198 | 0.248 |
| 11 (latest) | 1.252% | 0.506 | 0.497 |

Three observations follow. First, the reset-per-slice curve rises steeply across slices (AUC-PR 0.023 at slice 0 to 0.506 at slice 11), yet under this regime memory holds no cross-slice history, so the rise is not a memory effect. It tracks the concurrent rise in laundering prevalence, which climbs from 0.068% to 1.252% over the same slices; precision-recall performance is mechanically easier to achieve where positives are denser. The upward trend across slices is therefore driven substantially by prevalence rather than by accumulated memory.

Second, the controlled comparison, the two regimes at the same slice and therefore the same prevalence, is where the effect of memory shows. At the low-prevalence early slices, carrying memory continuously is markedly better: AUC-PR 0.095 versus 0.023 at slice 0, and 0.215 versus 0.048 at slice 4. Maintaining interaction history helps most when within-slice positives are scarce, which is precisely the regime in which a real-time AML system operates. This is leakage-free evidence that per-node memory contributes detection signal.

Third, the two regimes converge at the high-prevalence late slices (0.506 versus 0.497 at slice 11), and in aggregate the continuously-carried run scores slightly below the standard cold-start evaluation on the full test set (AUC-PR 0.271 versus 0.321, Table 4.6). The contribution of per-node memory is thus concentrated where it is operationally most valuable, in the early, low-prevalence portion of the future period, while the headline test figure is best read as a cold-start estimate over the test window.

**4.6 Cross-Model Comparison**

Table 4.8 presents all evaluated models in a single leaderboard ordered by AUC-PR. Because every model now uses the identical chronological split, the leaderboard is directly comparable across tiers, with no protocol asterisks.

**Table 4.8: Complete model leaderboard on the test set (chronological split), ordered by AUC-PR. F1 is at each model's calibrated threshold. GAT pending re-run.**

| Tier | Model | Params | AUC-ROC | AUC-PR | F1 |
|------|-------|--------|---------|--------|-----|
| Temporal | TGN | 86K | 0.9698 | 0.3213 | 0.2915 |
| Static | GCN | 63K | 0.9708 | 0.2056 | 0.1971 |
| Conv | XGBoost | N/A | 0.9393 | 0.1460 | 0.1608 |
| Conv | Random Forest | N/A | 0.9409 | 0.1249 | 0.1857 |
| Static | GAT (1 head) | 64K | 0.9575 | 0.0912 | 0.0898 |
| Temporal | TemporalGCN | 163K | 0.9514 | 0.0604 | 0.1298 |
| Temporal | EvolveGCN-H | 2.2M | 0.9064 | 0.0504 | 0.1038 |
| Static | GraphSAGE | 81K | 0.9452 | 0.0412 | 0.0915 |
| Conv | Logistic Regression | N/A | 0.9378 | 0.0378 | 0.0593 |

Several patterns emerge. First, there is a clear progression at the top of the table: the best conventional model (XGBoost, AUC-PR 0.146) is beaten by the best static GNN (GCN, 0.206), which is in turn beaten decisively by the continuous-time temporal model (TGN, 0.321). Graph structure helps, and fine-grained temporal modelling helps more.

Second, the progression is not automatic with either "graph" or "temporal" labels. GraphSAGE (a GNN) falls below the XGBoost baseline, and both snapshot temporal models fall below the static GCN. Adding graph structure or time only helps when it is modelled at the right granularity; coarse snapshot bucketing discards the transaction-level temporal signal that continuous-time modelling preserves, and mean-pooled neighbourhood aggregation discards the structural signal that spectral convolution preserves.

Third, every model is evaluated under the same chronological protocol, in which training precedes validation, which precedes testing in time (Section 3.3.3). The leaderboard is therefore like-for-like across tiers, and because the test period is strictly later than the training period, the figures are deployment-realistic estimates rather than the more optimistic numbers a split that mixed past and future transactions would yield.

Fourth, parameter count and performance are uncorrelated. EvolveGCN-H has by far the most parameters (2.2M) and among the worst performance, whereas TGN attains the best result in the study with 86K. Efficient architecture, not parameter count, determines detection quality.

**4.7 Reliability and Sensitivity Analysis**

The conclusions of this study rest on two models: GCN as the strongest static architecture and TGN as the strongest overall. To establish that their results are not artefacts of a single random seed or a single hyperparameter choice, both were retrained across three seeds (42, 123, 7) and subjected to a one-at-a-time sensitivity sweep on their most consequential hyperparameter. Seed repetition and sensitivity were concentrated on these two models by design; the remaining seven models are supporting comparisons on which no conclusion rests, and are reported as single documented runs (Section 3.6). Tables 4.9 and 4.10 report the results.

**Table 4.9: Seed stability across three seeds (mean +/- standard deviation).**

| Model | AUC-ROC | AUC-PR | F1 |
|-------|---------|--------|-----|
| GCN | 0.9715 +/- 0.0008 | 0.1776 +/- 0.0203 | 0.2023 +/- 0.0118 |
| TGN | 0.9686 +/- 0.0011 | 0.3396 +/- 0.0131 | 0.3450 +/- 0.0382 |

Two findings matter. AUC-ROC is essentially deterministic for both models (standard deviation around 0.001), so the ranking-quality claims are seed-independent. AUC-PR is more variable, as expected for a metric computed on roughly 1,500 positives among 760,000 test edges, but the separation between the two models survives comfortably: TGN's AUC-PR (0.340 +/- 0.013) and GCN's (0.178 +/- 0.020) do not overlap within a standard deviation, and TGN's is in fact the tighter of the two. The headline advantage of continuous-time temporal modelling is therefore a stable property, not a favourable draw. It is worth recording that the single GCN run reported in Table 4.3 (AUC-PR 0.206) sits at the upper end of the GCN seed distribution; the seed mean of 0.178 is the more representative figure, and the TGN advantage is correspondingly larger against it.

**Table 4.10: One-at-a-time hyperparameter sensitivity (seed 42).**

| Model | Hyperparameter | Value | AUC-ROC | AUC-PR | F1 |
|-------|----------------|-------|---------|--------|-----|
| GCN | dropout | 0.2 | 0.9711 | 0.1683 | 0.2148 |
| GCN | dropout | 0.3 (default) | 0.9708 | 0.2056 | 0.2185 |
| GCN | dropout | 0.5 | 0.9634 | 0.1913 | 0.0889 |
| TGN | pos_weight_mult | 0.005 | 0.9683 | 0.3306 | 0.3643 |
| TGN | pos_weight_mult | 0.01 (default) | 0.9698 | 0.3213 | 0.2915 |
| TGN | pos_weight_mult | 0.02 | 0.9668 | 0.3576 | 0.3797 |

Both models are robust to their principal hyperparameter within a sensible range. GCN is stable between dropout 0.2 and 0.3 and degrades only when over-regularised at 0.5 (where the calibrated F1 collapses). TGN's AUC-PR stays within a narrow 0.33 to 0.36 band as the class-weight moderator varies by a factor of four. Neither model's headline result depends on a knife-edge hyperparameter setting, which is the reliability property the sensitivity analysis was designed to test.

**4.8 Tool Performance Summary**

**Reliability and reproducibility.** All experiments use a fixed seed (42) across NumPy, PyTorch, and Python's random module, and deterministic index-based data splits, so re-running any experiment with the same arguments reproduces its result. Reproduction commands and the exact library versions used for the results in this chapter are documented in Appendix B, and per-model learning curves, showing training loss and validation convergence for every neural model, in Appendix F (Figure F.1).

**Computational profile.** The models differ substantially in how their cost scales with graph size. The static GNNs use full-batch message passing, whose memory footprint grows with the number of nodes and edges and is the binding constraint for dense-attention models such as GAT at this scale. GraphSAGE's neighbourhood sampling and TGN's batched, per-edge processing bound memory independently of the full graph and are therefore inherently better suited to larger deployments. All experiments were feasible on a single commodity workstation.

**Parameter efficiency.** TGN attains the best detection quality in the study with the smallest GNN footprint: 86K parameters against GCN's 63K (comparable) and EvolveGCN-H's 2.2M (worst performance, largest model). Parameter efficiency is relevant to deployment scenarios where model size affects inference latency and memory.

**Interpretability.** One capability the tool does not provide is an explanation for an individual flag. None of the models outputs a human-legible reason for classifying a specific transaction as suspicious. Only GAT exposes attention weights as a by-product of inference, and even these are a contested form of explanation (Section 2.3.3); GCN and TGN expose no such signal, and the best-performing model, TGN, is the hardest to explain, because each prediction depends not only on the current transaction and its neighbourhood but on the accumulated per-node memory of prior interactions. This is a genuine effectiveness limitation rather than a cosmetic one: under FATF (2021) guidance, a detection tool whose outputs cannot be understood by non-experts or communicated to competent authorities is of limited operational use, since a compliance analyst must be able to document and defend a suspicious-activity report. The tool as evaluated therefore delivers strong detection but no native interpretability; what would be required to close this gap is set out as a deployment recommendation in Section 5.3.

**Scalability and generalisability.** All results are on the HI-Small variant (518,581 accounts, 5,078,345 transactions). The IBM AML suite also provides larger variants generated by the same process (Altman et al., 2023), so the architectural findings are expected to transfer, but an empirical scaling study across variants (train and inference time, memory, and throughput as a function of graph size) was not run for this submission and is identified as future work (Section 5.5). Within-dataset generalisation to unseen time periods is demonstrated directly: every model is tested on transactions strictly later than those it was trained on, and TGN's per-slice analysis (Section 4.5.3) shows its per-node memory contributes transferable signal in the early, low-prevalence portion of that future period.
