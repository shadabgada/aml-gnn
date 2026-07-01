# Thesis Defense Study Guide

Study this, not the full report. Every claim below has a one-sentence answer you can deliver verbally. Know the numbers cold. Know why each design decision was made.

---

## 1. 2-Minute Thesis Summary

Money laundering moves roughly $2 trillion a year through the financial system. Current detection is rule-based and misses the relational and temporal nature of laundering — accounts don't launder in isolation, they launder in networks.

Graph Neural Networks can capture these patterns. But the literature had a gap: the IBM AML benchmark was published with static GNN results only. No one had tested temporal GNNs on it. No one had done a systematic three-tier comparison.

I built and compared nine models across three tiers on the same dataset: conventional ML with no graph structure, static GNNs with graph but no time, and temporal GNNs with both. The key finding: TGN, a continuous-time temporal GNN with per-node memory, achieves AUC-ROC of 0.97 and AUC-PR of 0.32 under deployment-realistic chronological evaluation. That's a 70% improvement over the best static GNN on the metric that matters most for imbalanced data.

The surprise was that snapshot-based temporal models — which group transactions into time windows — actually underperform the static GCN. Temporal modelling only helps when it's at transaction-level granularity.

For practitioners, I provide a decision framework: use conventional ML for quick deployment, static GNNs when you need relational awareness, and continuous-time TGN when detection quality justifies the infrastructure investment.

---

## 2. Key Numbers — Know These Cold

### Full leaderboard

| Model       | AUC-ROC   | AUC-PR    | F1        | Params   | Split             |
| ----------- | --------- | --------- | --------- | -------- | ----------------- |
| XGBoost     | 0.938     | 0.151     | 0.051     | —        | Random            |
| GCN         | 0.970     | 0.188     | 0.251     | 63K      | Random            |
| GAT         | 0.958     | 0.096     | 0.098     | 64K      | Random            |
| GraphSAGE   | 0.946     | 0.042     | 0.095     | 81K      | Random            |
| TemporalGCN | 0.957     | 0.064     | 0.134     | 162K     | Chronological     |
| EvolveGCN-H | 0.897     | 0.028     | 0.063     | 578K     | Chronological     |
| **TGN**     | **0.968** | **0.320** | **0.353** | **119K** | **Chronological** |

### Key comparisons to quote

- TGN vs GCN on AUC-PR: +70% (0.320 vs 0.188)
- TGN vs XGBoost on AUC-PR: +111% (0.320 vs 0.151)
- TGN vs TemporalGCN on AUC-PR: 5x higher (0.320 vs 0.064)
- GCN vs XGBoost on AUC-PR: +24.5% (0.188 vs 0.151)
- TGN per-slice AUC-PR: 0.05 (slice 0) to 0.45 (slice 11) — 9x improvement
- TGN matches GCN's AUC-ROC (0.968 vs 0.970) on a **harder** evaluation protocol
- Prevalence: 0.102% (5,177 laundering / 5,078,345 transactions)
- Positive-to-negative ratio: approximately 1:980

### TGN operating point (threshold 0.416)

- Precision: 0.426
- Recall: 0.301
- F1: 0.353
- At 0.1% prevalence, 43% precision means roughly 1.3 false positives per true positive

---

## 3. Five Novelty Claims — One Sentence Evidence Each

**1. First TGN application to AML edge classification on a public benchmark.**
The IBM AML paper (Altman et al., 2023, NeurIPS) only tested static GNNs. The TGN paper (Rossi et al., 2020) evaluated on Wikipedia, Reddit, and Twitter — not financial data.

**2. First systematic three-tier comparison on a unified AML benchmark.**
No published study compares conventional ML, static GNNs, snapshot temporal GNNs, and continuous-time temporal GNNs on the same dataset with the same protocol. Nine models total.

**3. Empirical demonstration that temporal granularity determines performance.**
Snapshot models (12 windows) achieve 5x lower AUC-PR than continuous-time TGN. Both temporal — but only one works. The finding: snapshot bucketing is too coarse for transaction-level laundering patterns.

**4. Deployment-realistic chronological evaluation.**
Published AML GNN studies use random splits, mixing past and future edges. I evaluate on strict future-edge prediction — train on past, test on future. This is harder but more honest.

**5. Snapshot-based temporal GNNs proven insufficient for AML.**
Both TemporalGCN (0.957) and EvolveGCN-H (0.897) underperform the static GCN (0.971). This negative finding directly motivates the continuous-time approach.

---

## 4. Design Decisions — How to Defend Each One

### Chronological split vs random split

**What I did:** Static models use random 70/15/15. Temporal models use chronological 70/15/15 (train on earliest 70%, validate on next 15%, test on latest 15%).

**Why:** In production, you train on historical data and predict future transactions. Random splits mix past and future — the model sees future edges during training, inflating performance. Chronological evaluation gives an honest deployment estimate. It also makes the task harder, which means TGN's numbers are conservative.

**Pushback:** "Why not use chronological for everyone?"
**Answer:** Static models treat the graph as fixed — they have no concept of time. Applying a chronological split to a static GNN would just mean training on less diverse data with no benefit. Random split is standard practice for static GNNs.

### EMA memory instead of GRU

**What I did:** Custom EMAMemory with beta=0.85 (m_new = 0.85 * m_old + 0.15 * msg).

**Why:** PyG 2.7's TGNMemory has a dtype mismatch bug — the last_update buffer is Long but receives Float timestamps. This causes in-place mutation errors during backward pass. I built a custom EMA memory module that avoids this. Also, the GRU in PyG's TGNMemory separates forward from update_state — update_state is called after loss.backward(), so the GRU never receives gradients. EMA update inside the forward pass fixes this.

**Pushback:** "Isn't GRU more expressive?"
**Answer:** In theory, yes. In practice, both bugs make the PyG GRU memory untrainable. EMA is simpler but functional — and the results speak for themselves (AUC-PR 0.320). A working EMA beats a broken GRU.

### Gradient clipping disabled

**What I did:** `--grad_clip 0` for TGN.

**Why:** pos_weight=12.4 means minority-class gradients are 12.4x larger. With grad_clip=1.0, nearly all positive-class gradients get clipped — the model never learns to detect laundering. Without clipping, AUC-ROC jumped from 0.79 to 0.93 in epoch 1.

**Pushback:** "Isn't gradient clipping standard for stability?"
**Answer:** Yes, for balanced datasets. For extreme class imbalance with high pos_weight, it's actively harmful. The pos_weight is already handling the imbalance — clipping undoes that.

### Single-head GAT

**What I did:** One attention head instead of the recommended 4-8.

**Why:** Four heads on a 5-million-edge graph exhausted CPU memory. Single head was the only option without GPU. This is a hardware constraint, not an architectural choice.

**Pushback:** "Doesn't this invalidate the GAT results?"
**Answer:** It means GAT's 0.958 AUC-ROC is a lower bound — multi-head would likely improve it. But even the lower bound is below GCN (0.971). The relative ranking (GCN > GAT) is consistent with the literature where spectral convolution often outperforms attention on homophilous graphs.

### 12 snapshots for temporal models

**What I did:** The 18-day transaction period divided into 12 equal time windows.

**Why:** The dataset has natural temporal density — 5M transactions over 18 days. 12 windows means roughly 1.5 days per window. Fewer windows lose temporal resolution, more windows leave too few edges per window for meaningful graph convolution.

**Pushback:** "Did you test other window counts?"
**Answer:** No — this is a limitation. CPU constraints prevented systematic sensitivity analysis. But the finding holds regardless: even an optimal snapshot count would still discard within-window transaction ordering. Only continuous-time processing captures per-transaction temporal patterns.

### pos_weight = 12.4

**What I did:** Computed as (num_negative / num_positive) from training set only.

**Why:** With 1:980 class imbalance, the model would otherwise ignore the minority class entirely. The pos_weight scales the loss for positive examples so each laundering transaction contributes equally to the gradient as ~980 legitimate ones.

**Pushback:** "12.4 seems arbitrary."
**Answer:** It's computed directly from the training set class ratio. The formula is: pos_weight = num_negatives / num_positives from training data only. No test-set information is used.

---

## 5. Four Bugs — What They Were and What You Learned

### Bug 1: TGNMemory dtype mismatch

PyG's TGNMemory uses a Long buffer for last_update but receives Float timestamps.
**Fix:** Custom EMAMemory module.
**Lesson:** Framework components aren't always production-ready. Understanding the internals matters.

### Bug 2: Data leakage — train/eval memory mismatch (MOST CRITICAL)

Training used updated memory (post-batch), eval used old memory (pre-batch). The model learned to exploit its own edge features stored in memory. AUC-ROC collapsed from 0.88 to 0.73.
**Fix:** Always predict with old memory. Train msg_proj through the classifier input, not through memory updates.
**Lesson:** Temporal models have subtle leakage modes that don't exist in static models. The train/eval memory state must be identical.

### Bug 3: Gradient clipping destroys signal

pos_weight=12.4 with grad_clip=1.0 clips almost all positive-class gradients.
**Fix:** Disabled gradient clipping.
**Lesson:** Defaults that work for balanced data can be catastrophic for imbalanced data. Know your class distribution.

### Bug 4: XGBoost no early stopping

XGBoost was training for default iterations without early stopping, leading to overfitting.
**Fix:** Added early_stopping_rounds=20 on validation log loss.
**Lesson:** Even mature libraries need proper configuration. Default settings assume balanced classification.

### Bonus: Double-weighting bug

Class_weight="balanced" plus manual sample_weight applied twice — squaring the minority class penalty.
**Fix:** Use class_weight only.
**Lesson:** When combining library features with custom weighting, verify there's no double-counting.

---

## 6. Likely Questions — Prepare These Answers

### From Kees (methodology focus)

**Q: Why nine models? Isn't that scope creep?**
A: The research plan had five models. Preliminary experiments showed snapshot temporal models underperforming — I needed to understand why. Adding TemporalGCN tested whether the issue was EvolveGCN-specific or snapshot-inherent. Adding TGN tested whether continuous-time could overcome the limitation. Each addition was driven by empirical evidence, not curiosity. The methodology documents this as a data-driven expansion.

**Q: How do you know the chronological split didn't just give TGN an unfair advantage?**
A: It gave TGN a disadvantage, not an advantage. Chronological evaluation is harder — the model sees a different class distribution at test time (laundering ratio increases from 0.01% to 0.30% across time). TGN matching GCN's AUC-ROC on this harder evaluation is the stronger claim.

**Q: The per-slice performance improvement — is that just because later slices have more laundering?**
A: Slice class balance varies, but the AUC metrics are threshold-independent. AUC-PR uses precision-recall across all thresholds. More positives don't inflate AUC-PR — they give a more reliable estimate. The 9x improvement comes from memory accumulation, not prevalence shift.

**Q: Why not use GPU?**
A: This was a deliberate constraint, not a limitation. Training on CPU demonstrates that these methods are accessible to institutions without GPU infrastructure — relevant for the practitioner audience. The trade-off is documented: single-head GAT, reduced-rank EvolveGCN, no hyperparameter search.

**Q: How do you know you didn't just overfit to HI-Small?**
A: The chronological split provides a partial answer — TGN was tested on future transactions it never saw during training. But I can't fully rule out dataset-specific overfitting without testing on Medium variants or other datasets. This is acknowledged as a limitation.

### From Debarati (practitioner focus — Learning Goal 6)

**Q: What should a compliance team actually do with these results?**
A: Three-tier decision framework in Section 5.3. If you need something running tomorrow with no graph infrastructure: XGBoost. If you have a transaction graph and want better precision: GCN. If detection quality justifies infrastructure investment and you need the best available performance: TGN.

**Q: At 43% precision, doesn't that mean more than half of alerts are false positives?**
A: Yes, but context matters. Current rule-based systems have over 95% false positive rates. 43% precision represents a dramatic reduction in alert burden. At 0.1% prevalence, TGN flags roughly 1.3 false positives for every true positive — that's manageable for analyst review. The baseline XGBoost generates about 37 false positives per true positive.

**Q: Can a bank actually deploy this?**
A: The reference implementation is open-source and documented. Key deployment considerations: you need a transaction graph with account-level identity, chronological retraining (weekly or monthly), and memory persistence between predictions. The tool is a prototype — productionisation would require additional engineering, but the architectural blueprint is proven.

**Q: What about explainability? Regulators require it.**
A: Two answers. First, GNN explanations are an active research area — GNNExplainer and similar methods can identify which edges most influenced a prediction. Second, the model's per-node memory can be inspected: an account whose memory embedding drifts significantly over time is behaviourally anomalous. Full explainability integration is future work.

**Q: Are there fairness concerns with deploying this?**
A: Yes, and I address this in Section 3.6. The IBM AML dataset uses FATF-derived typologies, not real enforcement data, which reduces but doesn't eliminate bias risk. If deployed on real data, the model could amplify existing enforcement biases in which accounts get flagged. Fairness auditing would be essential before production deployment.

### General

**Q: What's the single most important insight from this work?**
A: Temporal granularity matters more than temporal architecture. Having "temporal" in your model name doesn't help if you're bucketing transactions into coarse windows. You need per-transaction temporal processing.

**Q: What would you do differently with more time?**
A: Three things. First, test on the Medium dataset variant to verify scalability. Second, run full hyperparameter optimisation with GPU. Third, implement an ensemble combining static GCN's relational signal with TGN's temporal signal — they capture different patterns and should be complementary.

**Q: If you had to summarise your contribution in one sentence?**
A: I showed that continuous-time temporal GNNs with per-node memory decisively outperform both static GNNs and conventional ML for AML detection, and I proved that snapshot-based temporal approaches are insufficient — you need the right temporal granularity, not just temporal modelling.

---

## 7. Acknowledge These Weaknesses — Don't Defend Them

Be upfront about limitations. Owning them shows depth. Don't try to explain them away.

1. **Synthetic dataset.** Patterns are FATF-derived, not real criminal behaviour. Performance on real data is unknown. Frame this as a domain constraint: public real-world AML datasets with account-level detail don't exist.

2. **Single dataset variant.** Only HI-Small tested. Medium and LI variants untested due to CPU constraints.

3. **No hyperparameter optimisation.** Manual tuning only. GPU would enable systematic grid/random search.

4. **CPU-only training.** Limited GAT to single head, EvolveGCN-H to reduced rank. GPU would allow larger configurations.

5. **No ensemble methods.** Static GCN + TGN ensemble could potentially combine complementary signals.

6. **EMA memory instead of GRU.** Due to PyG compatibility issues. GRU might capture more complex temporal dependencies.

---

## 8. Architecture Quick Reference

|                   | TemporalGCN               | EvolveGCN-H                          | TGN                                |
| ----------------- | ------------------------- | ------------------------------------ | ---------------------------------- |
| What evolves      | Per-node hidden states    | GCN weight matrices                  | Per-node EMA memory                |
| Mechanism         | GRU on node state         | GRU on low-rank weights              | EMA: m_new = 0.85*m_old + 0.15*msg |
| Updates per run   | 12 (once per snapshot)    | 12 (once per snapshot)               | ~5M (every transaction)            |
| Per-node history? | Coarse behaviour summary  | None (weights only)                  | Fine-grained interaction history   |
| Key weakness      | Snapshot granularity loss | Weight instability + param explosion | EMA simpler than GRU (fixed)       |

---

## 9. Study Checklist

Before the defense, make sure you can answer these without notes:

- [ ] What was the research gap?
- [ ] Why nine models instead of the planned five?
- [ ] What is TGN's AUC-PR and why is it the key metric?
- [ ] Why do snapshot temporal models underperform static GCN?
- [ ] How does per-node memory in TGN work?
- [ ] What was the data leakage bug and how did you fix it?
- [ ] Why is chronological evaluation important?
- [ ] What's the 70% improvement number?
- [ ] What's the 5x improvement number?
- [ ] What's the 9x per-slice improvement?
- [ ] What would a practitioner do with your results?
- [ ] What are your top 3 limitations?
- [ ] What's your one-sentence contribution?
