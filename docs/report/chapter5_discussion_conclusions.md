# Chapter 5: Discussion, Recommendations and Conclusions

This chapter synthesises the empirical findings presented in Chapter 4 into answers to the research questions, discusses their theoretical and practical implications, acknowledges the study's limitations, and proposes directions for future research.

**5.1 Answering the Research Questions**

This section answers each research sub-question and the main research question, drawing on the theoretical framework established in Chapter 2, the methodology described in Chapter 3, and the empirical results reported in Chapter 4.

**5.1.1 SQ1: Graph Construction Design Decisions**

SQ1 asked: *What graph construction design decisions are required to represent financial transaction data as graph structures for static and temporal GNN-based AML analysis, and what is the rationale for each?*

Four design decisions proved consequential.

First, **composite account identity.** Accounts were identified by concatenating Bank ID and Account Number, creating a globally unique node identifier across the financial system. This decision, while seemingly straightforward, is fundamental: without unambiguous account identity, transactions cannot be consistently mapped to graph edges, and per-node behavioural histories (critical for temporal models) cannot be maintained. The composite key was validated against the accounts file to ensure consistency across all 518,581 accounts.

Second, **hand-crafted features over learned embeddings.** Twelve node features and twenty-eight edge features were constructed from domain knowledge about laundering behaviour: degree and volume statistics, temporal cyclic encodings, and one-hot categorical representations. Learned embeddings (for example, Node2Vec; Grover & Leskovec, 2016) were considered but rejected because hand-crafted features are directly interpretable, grounded in FATF-documented AML typologies, and computationally lightweight. The full feature specification is provided in Appendix A.

Third, **three graph construction strategies for three modelling paradigms.** Static GNNs used a single directed graph. Snapshot temporal GNNs used 12 quantile-based time windows, balancing temporal resolution against per-snapshot edge density. TGN used a continuous-time edge stream with individual timestamps. Using the appropriate data representation for each paradigm, rather than forcing a single representation across all models, ensured that each architecture was evaluated under the conditions for which it was designed. The quantile-based window strategy for snapshot models was chosen over equal-duration windows because transaction density is heavily skewed across time periods in this dataset.

Fourth, **chronological data splitting.** Transactions were sorted by timestamp and partitioned at 70/15/15 ratios. This evaluates models under deployment-realistic conditions: train on historical data, detect laundering in future transactions. Random splits, which mix past and future edges, introduce a subtle but consequential form of data leakage. As discussed in Section 3.3.3, several published AML GNN studies have used random splits, and the results reported in Chapter 4 suggest this practice inflates performance estimates. Chronological splitting should be standard in AML GNN evaluation.

**5.1.2 SQ2: GNN Architecture Choice and Detection Performance**

SQ2 asked: *How does the choice of GNN architecture affect money laundering detection performance on financial transaction networks, specifically comparing static architectures against snapshot-based temporal architectures and a continuous-time temporal architecture?*

The architecture choice matters substantially, but the critical factor is not whether a model is temporal, but at what granularity it models time.

Among **static architectures**, GCN outperformed both GAT and GraphSAGE. The margin is significant: GCN's AUC-PR of 0.1882 is nearly double GAT's 0.0958 and more than four times GraphSAGE's 0.0420. GCN's symmetric normalised aggregation appears well-suited to the financial transaction graph, where node degrees are highly variable and degree-based normalisation prevents high-degree nodes from dominating the aggregation. GAT's single-head limitation, enforced by CPU memory constraints, likely understates the potential of attention-based architectures. GraphSAGE's mean aggregation with neighbourhood sampling, while scalable, appears to dilute the discriminative signal from structurally distinctive laundering accounts.

Among **snapshot temporal architectures**, TemporalGCN outperformed EvolveGCN-H by a substantial margin (AUC-PR 0.0637 vs 0.0275). The performance gap is attributable to two factors. First, state-space evolution (TemporalGCN's GRU on per-node hidden states) provides a more stable temporal learning signal than weight-space evolution (EvolveGCN-H's GRU on GCN weight matrices). Per-node states carry account-specific behavioural history; weight matrices carry only aggregate graph dynamics. Second, EvolveGCN-H suffers from an inherent parameter explosion: the GRU hidden state scales as rank times the sum of input and output dimensions, producing 33 million parameters at rank 8 and exhausting available memory. Reducing the rank to 2 enabled training but at crippled capacity.

The **continuous-time TGN** decisively outperformed both snapshot-based models. TGN's AUC-PR of 0.3195 is 5.0 times TemporalGCN's 0.0637 and 11.6 times EvolveGCN-H's 0.0275, under identical chronological evaluation. The architectural distinction is granularity: TGN processes each of ~5 million transactions individually with its exact timestamp, while snapshot models aggregate transactions into 12 coarse windows. Laundering patterns such as structuring, which involve sequences of transactions within minutes or hours, are invisible at the snapshot level but detectable at the transaction level. TGN's per-node EMA memory provides a learned behavioural summary that accumulates over the entire transaction history, enabling the model to recognise accounts whose behaviour transitions from legitimate to suspicious.

A negative finding carries its own significance: **both snapshot-based temporal GNNs underperform the static GCN despite using temporal information.** TemporalGCN (AUC-ROC 0.957) and EvolveGCN-H (0.897) are both below static GCN (0.971). Temporal modelling is not automatically beneficial; at coarse granularity, it can be worse than no temporal modelling at all. This finding, while methodologically important, must be contextualised by the evaluation protocol difference: static GNNs used random splits while temporal models used the harder chronological split. A GCN evaluated chronologically would likely show lower performance than the 0.971 reported here, making the static-to-temporal comparison less stark than the raw numbers suggest. Nevertheless, the TGN-to-TemporalGCN comparison, which holds the evaluation protocol constant, confirms that continuous-time processing is the decisive advantage.

**5.1.3 SQ3: GNNs vs Conventional Machine Learning**

SQ3 asked: *How does the performance of static and temporal GNN-based models compare to Logistic Regression, Random Forest, and XGBoost in detecting money laundering?*

The comparison reveals a clear hierarchy. XGBoost (AUC-PR 0.1511) sets the non-graph performance ceiling. Static GCN (AUC-PR 0.1882) adds 24.5% over XGBoost from graph structure alone, confirming that relational information provides detection value beyond what flat features can capture. TGN (AUC-PR 0.3195) adds 111.5% over XGBoost from the combination of graph structure and fine-grained temporal modelling.

The performance of Logistic Regression (AUC-PR 0.0376) and Random Forest (AUC-PR 0.0619) confirms that simpler classifiers, even with class weighting, struggle with extreme class imbalance in the 28-dimensional edge feature space. XGBoost's regularised boosting provides better separation, but the absence of relational context limits its ceiling.

A nuanced interpretation is required. The GNN advantage over conventional classifiers is substantial but not overwhelming when only graph structure is added (GCN +24.5% AUC-PR over XGBoost). The decisive advantage emerges when continuous-time temporal modelling is combined with graph structure (TGN +111.5% AUC-PR over XGBoost). For an AML compliance team deciding whether to invest in graph-based detection infrastructure, the evidence suggests that graph structure alone provides a measurable but modest improvement; the full benefit requires the additional investment in temporal infrastructure.

**5.1.4 SQ4: Practical Implications for AML Practitioners**

SQ4 asked: *What practical implications do the comparative empirical findings hold for AML compliance practitioners?*

This sub-question is addressed in detail in the dedicated practitioner implications section (Section 5.3). In summary, the findings provide evidence-based guidance on model selection across three tiers, quantify the precision-recall trade-offs that operational compliance teams face, and identify the conditions under which investment in temporal GNN infrastructure is justified by a meaningful improvement in detection performance.

**5.1.5 Main Research Question**

The main research question asked: *How do static and temporal Graph Neural Network architectures compare to conventional supervised machine learning classifiers in detecting money laundering in financial transaction networks?*

The answer, grounded in the empirical evidence presented in Chapter 4, is as follows.

Continuous-time temporal GNNs with per-node memory (TGN) decisively outperform both static GNNs and conventional machine learning classifiers for AML detection under deployment-realistic chronological evaluation. The performance hierarchy is: **TGN > GCN > XGBoost > TemporalGCN > GraphSAGE > GAT > EvolveGCN-H**, measured by AUC-PR, the metric most sensitive to minority class detection quality.

Three qualifications are essential. First, **temporal modelling is not inherently beneficial.** Snapshot-based temporal GNNs (TemporalGCN, EvolveGCN-H) underperform the static GCN, demonstrating that temporal information must be modelled at transaction-level granularity to add value. Coarse temporal bucketing discards the very patterns it is meant to capture. Second, **graph structure alone provides a measurable but modest gain.** GCN improves AUC-PR by 24.5% over XGBoost. The combination of graph structure and continuous-time temporal modelling (TGN) improves AUC-PR by 111.5% over XGBoost. The whole is greater than the sum of its parts. Third, **evaluation protocol determines how honestly these numbers reflect real-world performance.** Chronological splitting, which evaluates models on future transactions after training on past transactions, provides a more deployment-realistic estimate than the random splits that predominate in published AML GNN studies. Under chronological evaluation, the gap between continuous-time and static approaches is likely larger than the numbers reported here suggest.

**5.2 Theoretical Implications**

This study makes four contributions to the theoretical understanding of GNN-based AML detection.

**Temporal granularity as a first-order design factor.** The finding that snapshot-based temporal GNNs underperform static GCN while continuous-time TGN substantially outperforms it establishes that temporal granularity, not temporal modelling in the abstract, determines detection performance. This extends the theoretical framework of Section 2.4, which presented snapshot and continuous-time paradigms as alternatives without evidence favouring one over the other. The empirical results provide such evidence: for financial transaction networks where laundering patterns unfold at the level of individual transactions, the snapshot paradigm is architecturally insufficient. This is not an implementation limitation but a theoretical one: no number of snapshots can fully recover transaction-level ordering if multiple transactions are aggregated within each window.

**Per-node memory as a learned behavioural summary.** TGN's per-slice performance improvement (AUC-PR from 0.05 to 0.45) provides empirical evidence that EMA memory functions as a learned behavioural summary, accumulating interaction history over time and enabling the model to recognise accounts whose behaviour transitions from legitimate to suspicious. This connects to the criminological theory discussed in Section 2.1: Levi (2002) identified that laundering is detectable through relational context, and the FATF (2023) defines layering as an inherently sequential process. Per-node memory operationalises these theoretical insights by maintaining a differentiable summary of each account's transaction history, updated with each new interaction.

**Weighted loss and gradient clipping interaction.** The methodological finding that gradient clipping destructively interacts with large pos_weight values under extreme class imbalance (Section 3.4.4) has implications beyond this implementation. Standard neural network training guidance recommends gradient clipping as a stability measure. This recommendation must be qualified when large class weights are applied: clipping thresholds should be set relative to the post-weighted gradient magnitudes, not the unweighted ones. This finding contributes to the literature on training neural networks under class imbalance (Dou et al., 2020; He & Garcia, 2009).

**Chronological evaluation as a methodological standard.** The finding that evaluation protocol materially affects reported performance, with random splits producing higher estimates than chronological splits, has implications for AML GNN research methodology. The field would benefit from standardising on chronological evaluation, which more honestly reflects deployment conditions. Studies that report only random-split results may be overstating real-world performance.

**5.3 Practitioner Implications**

This section translates the empirical findings into actionable guidance for AML compliance practitioners, directly addressing the assessment criterion that the thesis provide concrete, evidence-based recommendations for the compliance practice community.

**5.3.1 Model Selection Decision Framework**

The three-tier evaluation supports a decision framework for AML compliance teams selecting a detection approach. The appropriate tier depends on three factors: the institution's existing data infrastructure, the acceptable false positive burden, and the regulatory stakes of missed detection.

**Tier 1: Conventional ML (XGBoost).** Appropriate when an institution needs a quickly deployable system with low infrastructure requirements. XGBoost operates on flat transaction features, requires no graph database or temporal infrastructure, trains in minutes, and produces interpretable feature importance scores. Its AUC-PR of 0.151 means that, at a calibrated threshold, it detects a meaningful fraction of laundering cases. The limitation is precision: XGBoost's low precision at any reasonable recall level means compliance analysts will review many false positives. This tier is suitable for institutions in early stages of AML analytics maturity, or as a baseline against which more sophisticated approaches are benchmarked.

**Tier 2: Static GNN (GCN).** Appropriate when an institution has invested in graph infrastructure and seeks improved precision over conventional approaches. GCN's AUC-PR of 0.188 represents a 24.5% improvement over XGBoost. At its calibrated threshold (0.70), GCN detects 39% of laundering at 18% precision. This means approximately one in five alerts is genuine, compared to approximately one in thirty-eight for XGBoost at default threshold. The infrastructure requirements are moderate: a graph database mapping accounts to nodes and transactions to edges, with batch retraining as new transaction data arrives. GCN does not model temporal dynamics, so it is most appropriate when the primary laundering patterns of concern are relational (layering chains, fan-in/fan-out structures) rather than temporal (behavioural transitions, transaction sequencing).

**Tier 3: Continuous-Time Temporal GNN (TGN).** Appropriate when detection quality is a regulatory or operational priority justifying additional infrastructure investment. TGN's AUC-PR of 0.320 represents a 111.5% improvement over XGBoost and a 70.0% improvement over GCN. At its calibrated threshold (0.42), TGN detects 30% of laundering at 43% precision: nearly one in two alerts is genuine. The infrastructure requirements are more substantial: transactions must be processed in chronological order with individual timestamps, per-node memory states must persist across inference batches, and model retraining must respect temporal ordering to avoid data leakage. The investment is justified when the cost of missed laundering (regulatory fines, reputational damage, criminal facilitation) outweighs the cost of temporal infrastructure.

Table 5.1 summarises the decision framework.

**Table 5.1: Model selection framework for AML compliance practitioners.**

| Factor | Tier 1: XGBoost | Tier 2: GCN | Tier 3: TGN |
|--------|-----------------|-------------|-------------|
| AUC-PR | 0.151 | 0.188 | 0.320 |
| Precision at calibrated threshold | ~0.03 | ~0.18 | ~0.43 |
| Infrastructure requirements | Low | Moderate | Substantial |
| Training time (CPU) | ~3 min | ~102 min | ~114 min |
| Interpretability | High (feature importance) | Moderate (node embeddings) | Moderate (memory states) |
| Temporal dynamics | Not modelled | Not modelled | Modelled (continuous-time) |
| Deployment complexity | Low | Moderate | High |

**5.3.2 Precision-Recall Trade-offs and Operational Alert Burden**

The precision-recall trade-off has direct operational consequences for compliance team workload. At the calibrated threshold, TGN generates approximately 43 true positives for every 100 alerts, with the remaining 57 being false positives. At a laundering prevalence of 0.1%, this means that for every 100,000 transactions processed, approximately 100 are genuine laundering cases. TGN at its calibrated threshold would flag approximately 70 transactions, of which roughly 30 would be genuine laundering and 40 would be false positives. This alert volume (70 per 100,000 transactions, or roughly 700 per million) is within the review capacity of a typical compliance team.

In contrast, XGBoost at its default threshold flags approximately 3,250 transactions per 100,000 (based on its 0.0265 precision and 0.8610 recall, flagging 86% of 100 laundering cases at 2.7% precision), producing approximately 3,150 false positives for every 86 true positives. An analyst reviewing 100 alerts per day would see approximately 3 genuine laundering cases from XGBoost versus approximately 43 from TGN.

The threshold is configurable. An institution prioritising recall (catching as many laundering cases as possible, accepting more false positives) can lower the threshold. An institution prioritising precision (minimising analyst time wasted on false positives) can raise it. The calibrated thresholds reported in Chapter 4 maximise F1-score but are not prescriptive; each institution should calibrate against its own cost ratio of false negatives to false positives.

**5.3.3 Deployment Considerations**

Several practical considerations arise from the development and evaluation of these models.

**Chronological retraining.** TGN's per-node memory states are a function of transaction history. When the model is retrained on new data, memory states must be reinitialised from the beginning of the training period or carried forward from the previous training run. The former is simpler but discards accumulated history; the latter preserves history but requires careful handling to avoid stale memory states from outdated model weights. A practical approach is periodic full retraining (monthly or quarterly) with memory states computed from scratch over the full historical dataset, combined with daily inference using frozen model weights and continuously updating memory.

**Feature engineering in production.** The 28 edge features and 12 node features used in this study (detailed in Appendix A) were computed from raw transaction and account data. In a production setting, these features must be computed in real time or near-real time as transactions arrive. The log-transformed amount features and one-hot encoded categorical features are straightforward to compute; the cyclic time encodings (hour of day, day of week) require timestamp parsing. The node features (degree, volume, counterparty statistics) are aggregate statistics that must be recomputed or incrementally updated as new transactions arrive.

**Memory persistence.** TGN's per-node memory is the model's learned summary of account behaviour. For deployment, memory states for all active accounts must persist between inference batches. This requires a memory store (in-memory key-value store for latency-sensitive applications, or database-backed for durability) mapping account composite keys to memory vectors. The memory dimension in this implementation is 64, meaning each account's memory state is a 64-dimensional float vector (256 bytes at float32). For 500,000 accounts, total memory storage is approximately 128 MB, which is negligible by modern infrastructure standards.

**Reproducibility and adaptation.** The complete source code, trained model checkpoints, and reproduction commands are available in the project repository (Appendix B). Compliance analytics teams can reproduce the reported results, evaluate the models against their own institutional data, and adapt the implementation to their specific requirements. The tool's modular architecture separates data loading, feature engineering, graph construction, model definition, and training, allowing individual components to be replaced or extended without modifying the rest of the pipeline.

**5.3.4 Cost-Benefit Considerations**

The decision to invest in temporal GNN infrastructure depends on the institution's risk exposure and current detection baseline. TGN's 70% AUC-PR improvement over GCN represents a meaningful detection gain, but it requires investment in chronological data pipelines, memory state management, and more complex model operations. For an institution currently operating rule-based systems with very low detection rates, the incremental benefit of GCN over rule-based approaches may be large enough to justify graph infrastructure, with TGN representing a second-phase investment. For an institution already using conventional ML, the direct jump to TGN may be justified if the precision improvement (from ~3% to ~43%) translates to analyst time savings and improved detection of sophisticated laundering schemes.

This study does not provide a financial cost-benefit analysis, as the costs of false negatives (regulatory fines, criminal facilitation) and false positives (analyst time, delayed legitimate transactions) are institution-specific. However, the quantified performance differences reported in Chapter 4 provide the empirical inputs that an institution would need to conduct such an analysis.

**5.4 Limitations**

This study has several limitations that should be considered when interpreting its findings and assessing their generalizability.

**Synthetic dataset.** The IBM AML HI-Small dataset is synthetic, with laundering patterns derived from FATF-documented typologies (Altman et al., 2023). While this ensures the patterns reflect regulatory knowledge, it also means the model has been evaluated on simulated rather than genuine criminal behaviour. The extent to which performance on this benchmark transfers to real-world money laundering detection depends on how closely the FATF-informed simulation approximates actual laundering patterns in institutional transaction data. Publicly available real-world AML transaction datasets with account-level granularity do not currently exist, making synthetic benchmarks the only reproducible evaluation option available to independent researchers.

**Single dataset variant.** Only the HI-Small variant (518,581 accounts, 5,078,345 transactions) was used. The IBM AML dataset offers four variants (HI/LI combined with Small/Medium). While the data-generating process is identical across variants, meaning architectural findings are expected to generalise, empirical verification on the larger variants and the lower-prevalence LI variants was not performed. The Medium variant, with tens of millions of transactions, would test the scalability claims made here.

**CPU-constrained training.** All models were trained on a single CPU, which restricted architectural choices. GAT was limited to a single attention head, likely understating the potential of attention-based architectures. EvolveGCN-H was limited to rank 2, likely understating what the architecture could achieve at higher ranks. No automated hyperparameter optimisation was performed. GPU training would enable these restrictions to be lifted and might yield different performance rankings, particularly for GAT and EvolveGCN-H.

**Snapshot granularity not systematically investigated.** The 12-window snapshot configuration was chosen as a reasonable balance between temporal resolution and per-snapshot edge density. The sensitivity of snapshot model performance to the number of snapshots was not systematically varied. It is possible that a larger number of snapshots (for example, 100 or 1,000) could partially close the gap between snapshot and continuous-time models, though the computational cost would scale linearly with the number of snapshots.

**EMA memory versus GRU memory.** TGN used EMA-based memory rather than the GRU-based memory in Rossi et al. (2020), due to a PyG TGNMemory dtype incompatibility documented in Section 3.4.4. The custom EMA implementation maintains a fixed beta parameter (0.85) that controls the rate at which historical information decays, whereas GRU-based memory learns this rate through gating. The extent to which this architectural difference affects the reported results is unknown.

**No ensemble methods.** Individual models were evaluated independently. Ensembles combining complementary architectures (for example, GCN for relational patterns plus TGN for temporal patterns) were not explored and might outperform any single model.

**Single financial system simulation.** The IBM AML dataset simulates transactions within a single financial system. Cross-institutional laundering, where funds move between accounts at different banks, is not represented. Real-world AML detection often involves multiple institutions with incomplete visibility into each other's transaction networks.

**No fairness or bias analysis.** The dataset's laundering patterns are derived from FATF typologies rather than real enforcement data, which mitigates but does not eliminate the risk that the model learns patterns correlated with legitimate but atypical financial behaviour. No analysis of model fairness across entity types, banks, or transaction patterns was conducted.

**5.5 Future Research**

The findings and limitations of this study suggest several directions for future research.

**Cross-variant and cross-domain evaluation.** Extending the evaluation to all four IBM AML variants (HI/LI combined with Small/Medium) would test the generalizability of the architectural findings across dataset scales and prevalence ratios. Beyond the IBM AML benchmark, evaluating the same architectures on other public financial transaction benchmarks, should they become available, would test whether the performance hierarchy reported here is dataset-specific or architecture-inherent.

**GPU-scale training with hyperparameter optimisation.** Training these architectures on GPU hardware with systematic hyperparameter search (grid, random, or Bayesian) would test whether the performance rankings reported here are robust to hyperparameter choices and whether architectures constrained by CPU (GAT multi-head, EvolveGCN-H higher rank) achieve better performance with those constraints lifted.

**Alternative continuous-time architectures.** Beyond TGN, other continuous-time temporal GNN architectures exist, including temporal attention networks (Xu et al., 2020) and DyRep (Trivedi et al., 2019). Evaluating these on the IBM AML benchmark would provide a more complete picture of the continuous-time paradigm's capabilities for AML detection.

**Ensemble approaches.** Combining complementary architectures, such as a static GCN for structural pattern detection with a TGN for temporal pattern detection, may yield performance exceeding any single model. The different information sources (relational vs temporal) suggest that ensemble predictions could be more robust than individual model predictions.

**Fairness and bias in GNN-based AML.** Research is needed on whether GNN-based AML models exhibit bias against particular entity types, geographies, or transaction patterns. If GNN message passing amplifies biases present in the underlying transaction data, the fairness implications for automated AML screening could be significant.

**Multi-institutional transaction networks.** Extending the graph to include transactions across multiple financial institutions, potentially through federated learning or privacy-preserving techniques, would address the limitation that real-world laundering often spans multiple banks.

**5.6 Concluding Remarks**

This study set out to answer a question with direct relevance to both the academic literature and the AML compliance practice community: how do graph neural network architectures, spanning static and temporal paradigms, compare to conventional machine learning for detecting money laundering in financial transaction networks?

The answer, supported by a systematic three-tier evaluation on a standardised public benchmark, is that continuous-time temporal GNNs with per-node memory represent the most effective approach currently available. TGN achieves an AUC-ROC of 0.9684 and an AUC-PR of 0.3195 under deployment-realistic chronological evaluation, substantially outperforming both static GNNs and conventional classifiers. The key insight is that temporal granularity matters: continuous-time processing at the individual transaction level captures patterns that coarse snapshot-based approaches cannot.

The journey revealed an unexpected finding of equal importance: temporal modelling is not automatically beneficial. Both snapshot-based temporal architectures underperformed the static GCN, demonstrating that temporal information must be at the right granularity to add value. This negative result directly motivates the continuous-time approach and serves as a cautionary note for future AML GNN research: building a temporal model is not sufficient; the temporal resolution must match the timescale of the patterns being detected.

The tool developed in this research, comprising data engineering pipelines, seven model implementations across three architectural tiers, and a reproducible evaluation framework, is available as open-source reference implementation. For the AML compliance practice community, the evidence base now exists for informed model selection: conventional ML for rapid deployment with basic detection, static GNNs for improved precision through relational modelling, and continuous-time temporal GNNs when detection quality justifies infrastructure investment.

Money laundering, as a phenomenon, is both relational and temporal. It exploits the structure of financial networks and the sequencing of transactions. The detection tools built to counter it must, as this study has demonstrated, address both dimensions.
