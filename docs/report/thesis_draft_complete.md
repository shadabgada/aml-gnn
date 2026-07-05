# Graph Neural Networks Applied to Money Laundering Detection

## Master Thesis

**Final**

---

**Student:** Shadab Gada (500981772)

**Supervisor:** Kees van Montfort, PhD

**Second Assessor:** Debarati Bhaumik, PhD

**Program:** Master Digital Driven Business

**Institution:** Amsterdam University of Applied Sciences

**Date:** June 2026

---

## Table of Contents

**Chapter 1: Introduction**

- 1.1 Background: Money Laundering as a Global Challenge
- 1.2 Current AML Detection Approaches and Their Limitations
- 1.3 The Graph-Structured Nature of Financial Transactions
- 1.4 Problem Statement
- 1.5 Research Objectives
- 1.6 Main Research Question and Sub-Questions
- 1.7 Contributions
- 1.8 Report Structure

**Chapter 2: Theoretical Framework**

- 2.1 Money Laundering Typologies and Regulatory Context
- 2.2 Conventional Machine Learning for AML Detection
- 2.3 Graph Neural Networks
  - 2.3.1 Foundational Architectures: GCN, GAT, and GraphSAGE
  - 2.3.2 GNNs for Financial Crime Detection
  - 2.3.3 Explainability and Interpretability of GNN Predictions
- 2.4 Temporal Graph Neural Networks
  - 2.4.1 Snapshot-Based Approaches: TemporalGCN and EvolveGCN
  - 2.4.2 Continuous-Time Approaches: Temporal Graph Networks
- 2.5 Evaluation Under Class Imbalance
- 2.6 Research Gap Synthesis

**Chapter 3: Research Methodology and Tool Development**

- 3.1 Research Design Overview
- 3.2 Dataset: IBM AML HI-Small
  - 3.2.1 Dataset Selection and Justification
  - 3.2.2 Dataset Characteristics
- 3.3 Data Engineering and Graph Construction (SQ1)
  - 3.3.1 Feature Engineering
  - 3.3.2 Graph Construction for Static and Temporal Models (SQ1)
  - 3.3.3 Chronological Data Splitting
- 3.4 Model Architectures (SQ2 and SQ3)
  - 3.4.1 Conventional ML Baselines
  - 3.4.2 Static GNNs
  - 3.4.3 Snapshot Temporal GNNs
  - 3.4.4 Continuous-Time TGN
  - 3.4.5 Design Justification
- 3.5 Training and Evaluation Protocol (SQ2 and SQ3)
  - 3.5.1 Loss Functions and Class Weighting
  - 3.5.2 Hyperparameter Configuration
  - 3.5.3 Evaluation Metrics and Threshold Calibration
- 3.6 Ethical Considerations, Validity, and Reliability

**Chapter 4: Results, Analyses and Tool Performance**

- 4.1 Baseline Results: Conventional Machine Learning (Tier 1)
- 4.2 Static GNN Results: Graph Structure Without Time (Tier 2)
- 4.3 Temporal GNN Results: Graph Structure With Time (Tier 3)
  - 4.3.1 Snapshot-Based Temporal Models
  - 4.3.2 Continuous-Time TGN
  - 4.3.3 TGN Temporal Generalisation: Per-Slice Analysis
- 4.4 Cross-Model Comparison
- 4.5 Tool Performance Summary

**Chapter 5: Discussion, Recommendations and Conclusions**

- 5.1 Answering the Research Questions
  - 5.1.1 SQ1: Graph Construction Design Decisions
  - 5.1.2 SQ2: GNN Architecture Choice and Detection Performance
  - 5.1.3 SQ3: GNNs vs Conventional Machine Learning
  - 5.1.4 SQ4: Practical Implications for AML Practitioners
  - 5.1.5 Main Research Question
- 5.2 Theoretical Implications
- 5.3 Practitioner Implications
  - 5.3.1 Model Selection Decision Framework
  - 5.3.2 Precision-Recall Trade-offs and Operational Alert Burden
  - 5.3.3 Deployment Considerations
  - 5.3.4 Cost-Benefit Considerations
- 5.4 Limitations
- 5.5 Future Research
- 5.6 Concluding Remarks

**Appendices**

- Appendix A: Complete Feature Specification
- Appendix B: Reproducibility Guide
- Appendix C: Generative AI Usage Declaration
- Appendix D: Full Results Tables
- Appendix E: Hyperparameter Configurations
- Appendix F: Training and Validation Results
- Appendix G: Exploratory Data Analysis Figures

---

---

# Chapter 1: Introduction

**1.1 Background: Money Laundering as a Global Challenge**

Money laundering, the process through which illegally obtained funds are moved through legitimate financial channels to obscure their criminal origin, represents one of the most persistent threats to global economic stability and security (United Nations, 1988). The United Nations Office on Drugs and Crime estimates that between 2% and 5% of global GDP is laundered annually, financing organised crime, drug trafficking, terrorism, and human trafficking (UNODC, 2011). In absolute terms, this amounts to between $800 billion and $2 trillion USD per year, making anti-money laundering (AML) enforcement a critical regulatory and operational priority for financial institutions worldwide (UNODC, 2011).

Financial institutions are legally obligated under frameworks such as the European Union's Anti-Money Laundering Directives and the Financial Action Task Force (FATF) recommendations to detect and report suspicious activity (FATF, 2023). These obligations carry substantial operational costs: institutions collectively spend tens of billions of dollars annually on compliance infrastructure, yet the effectiveness of these systems remains limited. The scale of the problem, combined with the regulatory imperative and the high cost of compliance failures, creates an acute need for more accurate and efficient detection methods.

**1.2 Current AML Detection Approaches and Their Limitations**

Traditional rule-based AML systems apply fixed thresholds and heuristics to flag suspicious transactions: transactions above a certain amount, transfers to high-risk jurisdictions, or activity patterns matching predefined typologies (FATF, 2023). While these systems are interpretable and form the backbone of current compliance operations, they suffer from three fundamental limitations (Jensen & Iosifidis, 2023). First, they are rigid: rules must be explicitly defined and cannot adapt to evolving laundering tactics without manual intervention. Second, they generate extremely high false positive rates: industry reports suggest that over 95% of AML alerts are false positives, creating severe alert fatigue among compliance analysts and diverting resources from genuinely suspicious cases (Chen et al., 2018). Third, they evaluate each transaction in isolation, blind to the relational context that reveals sophisticated laundering schemes.

Conventional machine learning approaches, including logistic regression and tree-based models such as random forests and gradient boosting, have been applied to improve upon rule-based systems (Altman et al., 2023; Chen et al., 2018). These methods analyse individual transaction features such as amount, currency, and payment format, and improve detection rates to an extent. However, they share the third limitation of rule-based systems: they fundamentally fail to capture the relational structure of money laundering. Sophisticated laundering schemes, such as layering through chains of intermediary accounts or structuring (smurfing) across multiple accounts, are only detectable when the broader network of transactions is considered (Johannessen & Jullum, 2025). A single transaction may appear benign; its position within a network of suspicious activity reveals its true nature (Jullum et al., 2020).

**1.3 The Graph-Structured Nature of Financial Transactions**

Financial transaction data is inherently graph-structured. Accounts are entities, transactions are directed interactions between entities, and the patterns that distinguish legitimate activity from laundering behaviour emerge from the topology and dynamics of this interaction network. A laundering operation that distributes funds across ten accounts via fifty transactions is not suspicious at the individual transaction level; it is the collective pattern, the fan-out structure, the timing and sequencing of transfers, that constitutes the laundering signal.

This observation points toward a class of machine learning models specifically designed for graph-structured data: Graph Neural Networks (GNNs). GNNs propagate and aggregate information across nodes and edges, learning representations that capture both local features and the broader relational context of each entity in the network (Kipf & Welling, 2017). Applied to financial transaction networks, where accounts are nodes and transactions are edges, GNNs can learn from the entire web of financial interactions, capturing the relational and contextual patterns that distinguish legitimate activity from laundering behaviour (Johannessen & Jullum, 2025; Weber et al., 2019).

However, money laundering is not only a relational phenomenon. It is also a temporal one. Layering chains and structuring schemes unfold across time-ordered transaction sequences. The order in which transactions occur, the rhythm of account activity, and the evolution of behavioural patterns over time all carry signal that a static graph representation cannot capture. This temporal dimension motivates the exploration of temporal GNN architectures that explicitly model the dynamics of financial transaction networks.

**1.4 Problem Statement**

Despite substantial investment in AML compliance infrastructure and a growing body of research applying machine learning to financial crime detection, a significant gap persists in the empirical evaluation of detection architectures. The IBM Transactions for Anti-Money Laundering dataset (Altman et al., 2023), a large-scale synthetic dataset specifically designed for AML research and published at NeurIPS 2023, was released with baseline results using static GNNs only: GCN, GAT, and GraphSAGE. The dataset paper did not evaluate any temporal GNN architectures. Meanwhile, published temporal GNN work in AML (Alarab & Prakoonwit, 2023) used the Elliptic Bitcoin dataset, which represents a fundamentally different transaction domain. No study has produced a systematic comparative evaluation of temporal and static GNN architectures alongside conventional supervised classifiers within a unified framework on a standardised public banking AML benchmark (Cheng et al., 2024; Johannessen & Jullum, 2025).

Two paradigms exist for capturing temporal dynamics in graph learning: snapshot-based architectures such as TemporalGCN and EvolveGCN-H (Pareja et al., 2020), which partition transactions into time windows and evolve representations across windows, and continuous-time architectures such as the Temporal Graph Network (TGN; Rossi et al., 2020), which processes each transaction individually with its exact timestamp. Neither paradigm has been evaluated on the IBM AML benchmark.

This gap has practical as well as academic consequences. Absent a rigorous, like-for-like comparison spanning all three tiers (conventional machine learning, static GNNs, and temporal GNNs, encompassing both snapshot-based and continuous-time approaches), AML compliance practitioners lack a sound empirical basis for judging which class of model merits investment and what detection-performance trade-offs to expect. Equally, the practical value of any such comparison depends on operational preconditions that raw detection metrics do not capture: whether a flagged transaction can be explained to analysts and competent authorities, as expected under FATF guidance on new technologies for AML/CFT (FATF, 2021); what alert burden a given precision-recall operating point imposes on a compliance team; and how far results obtained on a synthetic benchmark transfer to real institutional deployment. This study therefore frames its practical contribution as evidence-based model-selection guidance together with a critical assessment of these operational preconditions, rather than as a turnkey deployment recommendation. The research problem is accordingly both academic (an unaddressed gap in the comparative evaluation literature) and practical (insufficient comparative evidence, and insufficient attention to the operational conditions, required for informed model selection in AML compliance contexts).

**1.5 Research Objectives**

This research addresses the identified gap through tool development: building and evaluating a GNN-based analytical system trained on the IBM AML HI-Small dataset. The work is guided by five specific objectives:

1. To perform the data engineering work required to transform raw financial transaction data from the IBM AML dataset into graph structures suitable for both static and temporal GNN-based analysis.
2. To implement and evaluate static GNN architectures (GCN, GAT, and GraphSAGE) for money laundering detection on financial transaction graphs.
3. To implement and evaluate temporal GNN architectures spanning both snapshot-based (TemporalGCN, EvolveGCN-H) and continuous-time (TGN) approaches, and to assess whether incorporating temporal transaction dynamics improves detection performance over static models.
4. To compare the detection performance of all GNN-based models against conventional supervised machine learning classifiers (Logistic Regression, Random Forest, and XGBoost) using metrics appropriate for heavily class-imbalanced data.
5. To translate the empirical findings into actionable guidance for AML compliance practitioners, addressing model selection, operational trade-offs, and deployment considerations for graph-based detection tools.

**1.6 Main Research Question and Sub-Questions**

The main research question guiding this study is:

**How do static and temporal Graph Neural Network architectures compare to conventional supervised machine learning classifiers in detecting money laundering in financial transaction networks?**

To answer this question systematically, it is decomposed into four sub-questions:

**SQ1.** What graph construction design decisions are required to represent financial transaction data as graph structures for static and temporal GNN-based AML analysis, and what is the rationale for each?

**SQ2.** How does the choice of GNN architecture affect money laundering detection performance on financial transaction networks, specifically comparing static architectures (GCN, GAT, and GraphSAGE) against snapshot-based temporal architectures (TemporalGCN and EvolveGCN-H) and a continuous-time temporal architecture (TGN)?

**SQ3.** How does the performance of static and temporal GNN-based models compare to Logistic Regression, Random Forest, and XGBoost in detecting money laundering, as measured by AUC-ROC, AUC-PR, Precision, Recall, and F1-score?

**SQ4.** What model-selection guidance for AML compliance practitioners emerges from the comparative findings once the operational preconditions of deployment (the precision-recall and alert-burden trade-off, the explainability required for regulatory reporting, and the ecological validity of synthetic-benchmark results) are taken into account?

These sub-questions are collectively answerable using the available data and experimental infrastructure, feasible within the scope of a master's thesis, directly relevant to the identified research gap, and logically interconnected: SQ1 establishes the data foundation, SQ2 evaluates architectural choices, SQ3 provides the comparative benchmark, and SQ4 translates the aggregate findings into practitioner guidance while critically assessing the operational preconditions that govern deployment.

**1.7 Contributions**

This research makes the following contributions:

**For the academic community:**

- The first application of temporal GNN architectures (TemporalGCN, EvolveGCN-H, and TGN) to the IBM AML benchmark, extending the static GNN results of Altman et al. (2023). TGN (Rossi et al., 2020) represents the first application of a continuous-time temporal GNN to AML edge classification.
- The first systematic three-tier comparative evaluation (conventional ML, static GNNs, snapshot temporal GNNs, continuous-time temporal GNN) on a unified AML benchmark under identical experimental conditions.
- Empirical evidence that continuous-time temporal modelling achieves substantially higher precision-recall performance than snapshot-based temporal approaches for AML detection, and that snapshot-based temporal GNNs can underperform even static models, a finding with implications for future AML GNN architecture design.
- A deployment-realistic chronological evaluation protocol in which models are trained on past transactions and tested on future transactions, providing more honest real-world performance estimates than the random splits predominant in published AML GNN studies.

**For the AML compliance practice community:**

- A comparative evaluation methodology spanning three architectural tiers that enables compliance teams to assess which class of detection model offers the best performance trade-off for their operational context.
- Evidence-based guidance on model selection across three tiers of detection approaches, grounded in a rigorous like-for-like comparison on the same dataset.
- A quantified analysis of the precision-recall trade-offs that operational compliance teams face when deploying graph-based detection tools, including the relationship between threshold selection and false positive burden.

**1.8 Report Structure**

The remainder of this report is structured as follows.

**Chapter 2 (Theoretical Framework)** synthesises the relevant literature across four domains: money laundering typologies and the regulatory context, conventional machine learning for AML, static GNN architectures and their application to financial crime detection, and temporal GNN architectures spanning both snapshot-based and continuous-time approaches. The chapter concludes with a synthesis of the research gap and the theoretical basis for the architectural choices evaluated in this study.

**Chapter 3 (Research Methodology and Tool Development)** describes the dataset, the data engineering and graph construction pipeline, the implementation of each model architecture, and the training and evaluation protocol. Design decisions are explicitly justified with reference to the literature discussed in Chapter 2, and the chapter includes a comparison of alternative design options where relevant.

**Chapter 4 (Results, Analyses and Tool Performance)** presents the empirical findings. Results are reported for each model tier, followed by a cross-model comparison, a temporal generalisation analysis examining TGN performance across time slices, and an assessment of tool scalability.

**Chapter 5 (Discussion, Recommendations and Conclusions)** answers each research sub-question and the main research question, discusses the practical implications of the findings for AML compliance practice, presents the study's theoretical contributions and novelty claims, acknowledges limitations, proposes directions for future research, and provides concluding remarks.

---

# Chapter 2: Theoretical Framework

**2.1 Money Laundering Typologies and Regulatory Context**

Money laundering is the process of disguising the criminal origin of funds by moving them through legitimate financial channels. The Financial Action Task Force (FATF), the global standard-setting body for anti-money laundering regulation, identifies three stages in the laundering process: placement, where illicit funds first enter the financial system; layering, where funds are moved through sequences of transactions to obscure their origin; and integration, where laundered funds re-enter the legitimate economy (FATF, 2023). These stages are not merely descriptive categories. They correspond to distinct behavioural patterns in transaction networks, each of which leaves a different structural signature in the graph of financial interactions.

Placement typically involves depositing large sums into accounts, creating transaction patterns with high individual amounts but relatively simple counterparty structures. Layering is the most structurally complex stage: funds are routed through chains of intermediary accounts, split into smaller amounts (structuring or smurfing), and distributed across multiple destinations (fan-out) before being reaggregated (fan-in). These patterns produce distinctive topological signatures: unusually long transaction chains, accounts with high out-degree relative to in-degree, and clusters of accounts with dense internal connectivity but weak external ties. Integration involves transactions with legitimate businesses, often characterised by amounts and frequencies that blend with normal commercial activity. The regulatory obligation to detect these patterns falls on financial institutions under frameworks such as the European Union's Anti-Money Laundering Directives and the FATF Recommendations, which require institutions to implement systems capable of identifying and reporting suspicious transactions (FATF, 2023).

The academic literature on money laundering has long identified the relational nature of laundering behaviour. Levi (2002) analysed money laundering as a criminological phenomenon, documenting how illicit funds traverse financial systems through networks of accounts and intermediaries, and argued that the practical limitations of detection arise precisely because individual transactions appear legitimate when examined in isolation. The analytical implication is clear: effective detection requires examining transactions in their relational context, not as independent observations.

Two additional considerations are relevant to this study. First, laundering patterns evolve over time in response to changes in detection methods and regulatory enforcement. A static detection system trained on historical patterns will degrade as launderers adapt their techniques, creating a structural need for models that can capture temporal dynamics. Second, the FATF-documented typologies (structuring, layering, fan-in/fan-out) are explicitly encoded in the IBM AML dataset used in this research (Altman et al., 2023), meaning the dataset's laundering patterns reflect real-world regulatory knowledge rather than arbitrary simulation choices. This connection between regulatory typologies and dataset design is methodologically significant: it means a model that learns to detect these patterns in the dataset is learning to detect patterns that the global regulatory framework identifies as suspicious.

**2.2 Conventional Machine Learning for AML Detection**

Conventional machine learning approaches to AML detection treat each transaction as an independent feature vector and apply supervised classification methods to distinguish laundering from legitimate activity. Chen et al. (2018) provided a comprehensive review of machine learning techniques applied to suspicious transaction detection, covering logistic regression, decision trees, support vector machines, and ensemble methods. Their review identified two persistent limitations: first, the extreme class imbalance inherent in AML data, where laundering transactions constitute a tiny fraction of total volume, makes standard classifiers prone to high false positive rates; second, treating transactions as independent observations discards the relational structure that characterises laundering behaviour. Kute et al. (2021) extended this line of review to deep learning and explainable AI approaches for AML, identifying CNNs, autoencoders, and graph deep learning as emerging techniques, and confirming that limited access to real transaction data and extreme class imbalance remain the dominant barriers to progress.

Logistic regression serves as the simplest baseline, modelling the log-odds of a transaction being suspicious as a linear function of its features. Its interpretability is an advantage in compliance contexts where regulatory requirements demand explainable decisions, but its linear decision boundary cannot capture the nonlinear interactions that characterise complex laundering schemes. Random forest classifiers (Breiman, 2001) address this by ensembling multiple decision trees trained on random subsets of features and samples, producing nonlinear decision boundaries while maintaining reasonable interpretability through feature importance scores. XGBoost (Chen & Guestrin, 2016) extends gradient boosting with regularisation and optimised computation, and has become a standard benchmark in tabular classification tasks across domains including financial crime detection.

Several studies have applied these methods to AML detection with varying success. Chen et al. (2018) reported that ensemble methods outperformed linear classifiers on synthetic AML data, but noted that all tabular methods suffered from the same structural limitation: they could not model relationships between transactions. A model can learn that transactions above a certain amount are more likely to be suspicious, but it cannot learn that a transaction is suspicious because it is the third in a chain of five small transfers between the same two accounts. This limitation is not an implementation detail; it is a fundamental consequence of the independence assumption underlying tabular machine learning. The next section discusses model architectures that explicitly relax this assumption.

**2.3 Graph Neural Networks**

**2.3.1 Foundational Architectures: GCN, GAT, and GraphSAGE**

Graph Neural Networks (GNNs) are a class of deep learning models designed to operate on graph-structured data. Unlike conventional neural networks that process independent feature vectors, GNNs propagate and aggregate information across the edges of a graph, allowing each node's representation to incorporate information from its neighbours, its neighbours' neighbours, and so on (Kipf & Welling, 2017). This mechanism, known as message passing, directly addresses the key limitation identified in the previous section: a transaction between two accounts is no longer evaluated in isolation, but in the context of the broader network of financial interactions.

The Graph Convolutional Network (GCN), introduced by Kipf and Welling (2017), is the foundational architecture for graph-based learning. In a GCN, each layer applies a shared linear transformation to node features, then aggregates the transformed features of each node's neighbours using a symmetric normalisation based on node degrees. The normalisation ensures that nodes with many neighbours do not dominate the aggregation. Despite its simplicity, the GCN has proven remarkably effective across a range of graph-based tasks, including node classification, link prediction, and graph classification. Its key strength for AML detection is that it captures homophily; the tendency of connected nodes to share similar properties. In a financial network, accounts participating in laundering schemes tend to be connected to other accounts involved in laundering, creating a signal that GCN aggregation can amplify.

The Graph Attention Network (GAT), proposed by Veličković et al. (2018), extends the GCN by replacing the fixed, degree-based normalisation with learnable attention coefficients. Each neighbour's contribution to a node's updated representation is weighted by an attention score computed from the features of both nodes, allowing the model to learn which connections are most informative. In a financial network, this is intuitively appealing: an account might have hundreds of counterparties, but only a few are relevant to detecting laundering behaviour. GAT's attention mechanism allows the model to focus on those relationships. However, the computational cost of computing pairwise attention scores across all edges can be substantial, and multi-head attention, which the original paper found necessary for stable training, multiplies this cost by the number of heads.

GraphSAGE (Hamilton et al., 2017) addresses a different limitation of the GCN: the transductive assumption. GCN and GAT require the full graph structure to be known at training time, which limits their applicability to settings where new nodes appear after training. GraphSAGE introduces an inductive learning framework based on neighbourhood sampling and aggregation. Rather than operating on the full graph Laplacian, GraphSAGE samples a fixed-size neighbourhood for each node and applies a learned aggregation function (mean, max, or LSTM) to compute node embeddings. This makes GraphSAGE scalable to very large graphs and enables it to generate embeddings for previously unseen nodes, a property relevant to financial networks where new accounts are continuously created.

These three architectures represent a progression of design philosophy: GCN provides the simplest and most computationally efficient graph learning mechanism; GAT adds adaptive, learnable edge weighting at the cost of increased computation; GraphSAGE adds scalability and inductive capability through sampling. All three have been applied to financial crime detection, as discussed in the following section.

**2.3.2 GNNs for Financial Crime Detection**

The application of GNNs to financial crime detection has grown substantially since the late 2010s. Weber et al. (2019) provided one of the earliest demonstrations, applying GCN and GraphSAGE to the Elliptic Bitcoin dataset to classify cryptocurrency transactions as licit or illicit. Their work established the empirical precedent that graph-based models can outperform conventional classifiers on financial transaction data by capturing relational patterns. However, the Elliptic dataset represents a specific cryptocurrency context; its transaction patterns, anonymised participants, and lack of regulatory reporting obligations differ substantially from the banking transaction domain.

Johannessen and Jullum (2025) applied heterogeneous GNNs to real-world banking data from DNB, Norway's largest financial institution, demonstrating that graph-based models outperformed conventional classifiers in detecting money laundering across multiple relationship types. Their work is significant because it used genuine institutional transaction data, providing ecological validity that studies on synthetic or cryptocurrency data cannot claim. However, their dataset is proprietary and non-public, making their results unreproducible by independent researchers and unusable as a benchmark for comparative evaluation. Cheng et al. (2023) proposed group-aware deep graph learning for organised money laundering, using community-centric encoding to capture shared transaction patterns among account groups.

Cheng et al. (2024) provided a comprehensive review of GNN architectures applied to financial fraud detection across domains including credit card fraud, insurance fraud, and money laundering (see also Motie & Raahemi, 2024; Li et al., 2025). Their review confirmed that GCN, GAT, and GraphSAGE are the predominant static architectures used in financial fraud research, and explicitly identified the incorporation of temporal dynamics into GNN architectures as a key future research direction. This finding directly motivates the temporal modelling component of the present study.

Altman et al. (2023) published the IBM Transactions for Anti-Money Laundering dataset at NeurIPS 2023, a large-scale synthetic dataset specifically designed to serve as a public benchmark for GNN-based AML research. Their paper reported baseline results using static GNNs (GCN, GAT, and GraphSAGE) and demonstrated that all three outperformed non-graph baselines. However, the dataset paper evaluated only static architectures and explicitly noted that temporal modelling was left to future work. This creates the empirical gap that the present study addresses.

Dou et al. (2020) addressed a challenge particularly relevant to AML detection: applying GNNs to fraud detection under severe class imbalance. Their work proposed techniques to improve minority fraud node detection in imbalanced graph classification settings, including adapted loss functions and sampling strategies. The class imbalance in their experiments, while substantial, was less extreme than the approximately 1:1000 ratio in the IBM AML dataset, suggesting that additional adaptations may be necessary for AML-specific applications.

A related challenge is fraudster camouflage, in which bad actors deliberately manipulate their attributes and connections to evade detection. Deng et al. (2022) proposed a contrastive graph neural network (CACO-GNN) that learns node representations robust to such camouflage, achieving an 18.5% improvement in F1-macro on real-world fraud datasets. Camouflage is directly relevant to AML, where launderers actively adapt their transaction patterns to avoid rule-based triggers.

Ren et al. (2023) applied dynamic GNNs with self-attentive temporal convolution to collaborative fraud detection, bridging static graph methods and temporal modelling. Tong and Shen (2023) directly coupled GNN-based representation learning with class imbalance handling in a unified architecture for financial transaction fraud detection, demonstrating that imbalance-aware graph learning outperforms generic GNNs on financial benchmarks. More broadly, Ma et al. (2023) surveyed deep graph anomaly detection, establishing a taxonomy spanning node-level, edge-level, and subgraph-level anomalies and identifying dynamic graph anomaly detection and class-imbalanced anomaly detection as open research directions directly relevant to the present study.

**2.3.3 Explainability and Interpretability of GNN Predictions**

A recurring limitation of GNNs, shared with deep neural networks generally, is their opacity: a model may flag a transaction as suspicious without providing a human-legible reason for the decision. In the AML context this is not a peripheral concern but an operational and regulatory one. FATF guidance on new technologies for AML/CFT identifies the interpretability of digital solutions as a central challenge, and states that their effective use requires tools whose outputs can be understood by non-experts and communicated to competent authorities when required (FATF, 2021). A detection model that cannot justify a flag is therefore of limited use to a compliance analyst who must document and defend a suspicious-activity report.

The literature offers several families of methods for explaining GNN predictions, surveyed comprehensively by Yuan et al. (2023). Instance-level (local) methods explain an individual prediction: GNNExplainer (Ying et al., 2019) identifies the compact subgraph and subset of node features most responsible for a given output, and subsequent methods such as PGExplainer and SubgraphX extend this to learned and game-theoretic explanations. Attention-based interpretation reads a model's own attention weights as an importance signal; in a GAT, the attention a node assigns to its neighbours can indicate which counterparties drove a classification, although the faithfulness of attention as an explanation is contested (Jain & Wallace, 2019). Model-agnostic attribution methods, including SHAP (Lundberg & Lee, 2017) and integrated gradients (Sundararajan et al., 2017), attribute a prediction to the model's input features and apply to any architecture, at the cost of treating the model as a black box.

These methods differ in what they can explain, which is directly relevant to the architectures evaluated in this study. Of the models considered here, only GAT produces attention weights as a by-product of inference, offering a limited native form of relational explanation; GCN and the continuous-time TGN expose no such signal and would require post-hoc attribution to be interpreted. Explaining continuous-time temporal models is a particular open problem, since a prediction depends not only on the current transaction and its neighbourhood but on the accumulated per-node memory of prior interactions, which most existing GNN explainers do not model. The absence of a built-in explanation mechanism in the best-performing architecture is revisited as an effectiveness limitation in the tool assessment (Chapter 4) and as a deployment precondition in the practitioner recommendations (Chapter 5).

**2.4 Temporal Graph Neural Networks**

The architectures discussed in Section 2.3 operate on static graphs: all nodes and edges are treated as simultaneously present, and temporal ordering is not modelled. For many real-world graphs, including financial transaction networks, this assumption is unrealistic. Money laundering is an inherently temporal process: transactions occur in sequence, behavioural patterns evolve, and the significance of an interaction depends on when it occurs relative to prior activity. Temporal GNNs address this by incorporating time into the graph learning process. Two broad paradigms exist: snapshot-based approaches that discretise time into a sequence of static graphs, and continuous-time approaches that process individual events with their exact timestamps. Barros et al. (2021) provided a comprehensive survey of dynamic graph embedding methods, classifying approaches along four dimensions (graph type, temporal mechanism, learning technique, and downstream task) and confirming that both paradigms are well-established, with temporal link prediction and anomaly detection as the most active application areas.

**2.4.1 Snapshot-Based Approaches: TemporalGCN and EvolveGCN**

Snapshot-based temporal GNNs partition the timeline into a sequence of intervals and construct a static graph for each interval. A base GNN processes each snapshot, and a recurrent mechanism propagates information across snapshots. This paradigm is straightforward to implement and builds directly on static GNN architectures, but its effectiveness depends critically on the granularity of the snapshot partitioning.

The Temporal Graph Convolutional Network (TemporalGCN) applies this approach by running a shared GCN on each snapshot and using a Gated Recurrent Unit (GRU; Cho et al., 2014) to evolve per-node hidden states across time steps. After processing snapshot t, each node's hidden state is updated as a function of its previous state and the GCN output for that snapshot. This allows the model to accumulate behavioural information over time: an account that receives many small deposits in early snapshots and suddenly sends a large transfer in a later snapshot will have a different hidden state than an account with consistent transaction patterns.

EvolveGCN (Pareja et al., 2020) takes a different approach. Rather than evolving per-node states, EvolveGCN evolves the GCN's weight matrices themselves using a recurrent network. The intuition is that the underlying dynamics of the graph change over time, and the model's parameters should adapt accordingly. EvolveGCN-H, the variant used in this research, uses a GRU to update the GCN weight matrices across snapshots, with the GRU input being a summary of the previous snapshot's node embeddings. Pareja et al. (2020) evaluated EvolveGCN on link prediction tasks using Bitcoin OTC and Reddit datasets, demonstrating improved performance over static baselines. However, EvolveGCN has not previously been evaluated on a standardised public banking AML benchmark.

Two structural concerns arise with the snapshot-based paradigm when applied to AML detection. First, the granularity of snapshot partitioning determines the model's temporal resolution: with N snapshots, the model observes at most N state transitions per node. Laundering operations such as structuring may unfold entirely within a single snapshot window, becoming invisible to the model. Second, snapshots are processed in fixed chronological order with no mechanism to revisit a transaction in light of later information, even though the significance of a transaction is often apparent only retrospectively.

**2.4.2 Continuous-Time Approaches: Temporal Graph Networks**

Continuous-time temporal GNNs address the granularity limitation of snapshot-based approaches by processing each interaction individually with its exact timestamp. Rather than aggregating transactions into time windows, these models maintain per-node memory that updates with every event, enabling them to capture transaction-level temporal dynamics.

The Temporal Graph Network (TGN), introduced by Rossi et al. (2020), is the foundational continuous-time architecture. TGN operates on a stream of timestamped edges. For each edge, the model computes an embedding for both incident nodes by combining three information sources: the node's current memory state, which summarises its interaction history up to that point; the output of a graph convolution over the node's temporal neighbourhood (recent neighbours in the graph); and a time encoding that captures the relative timing of interactions. An edge classifier then combines the source and destination node embeddings with edge features and a time encoding to produce a prediction. After the prediction is made, both nodes' memory states are updated with information from the current interaction using a learned message function and an exponential moving average (EMA) update rule.

TGN introduces several architectural innovations relevant to AML detection. The per-node memory functions as a learned summary of behavioural history: an account that has participated in suspicious patterns in the past will have a different memory state than one with only legitimate history, and this memory influences all subsequent predictions involving that account. The time encoding, implemented as a sinusoidal transformation of the raw timestamp difference (following the Transformer positional encoding approach of Vaswani et al., 2017), allows the model to learn temporal patterns at multiple scales, from minutes to days. The EMA memory update provides a smooth, differentiable mechanism for accumulating history, with the decay parameter beta controlling the trade-off between retaining old information and adapting to new behaviour.

Rossi et al. (2020) evaluated TGN on node classification and link prediction tasks using Wikipedia, Reddit, and Twitter datasets, demonstrating state-of-the-art performance compared to both static GNNs and snapshot-based temporal models. However, TGN has not previously been applied to AML edge classification on a public banking benchmark. The IBM AML dataset paper (Altman et al., 2023) evaluated only static GNNs, and no published study has applied continuous-time temporal GNNs to this dataset. This represents the primary research gap addressed by the present study.

A temporal GNN variant relevant to this work is the graph-based LSTM approach applied by Alarab and Prakoonwit (2023) to money laundering detection on the Elliptic Bitcoin dataset. Their work combined a temporal GCN with an LSTM to capture transaction dynamics, and explicitly identified that existing GNN-based AML studies had largely neglected temporal information. While their approach demonstrated that temporal modelling improves AML detection in the cryptocurrency domain, it was snapshot-based rather than continuous-time, and was evaluated on a dataset with fundamentally different characteristics from banking transaction data.

**2.5 Evaluation Under Class Imbalance**

The evaluation of machine learning models on heavily class-imbalanced data requires careful metric selection. In the IBM AML HI-Small dataset, laundering transactions constitute approximately 0.1% of all transactions. Under such conditions, classification accuracy is a misleading performance indicator: a model that classifies every transaction as legitimate achieves 99.9% accuracy but detects zero laundering cases.

He and Garcia (2009) provided a comprehensive analysis of class imbalance challenges in machine learning, reviewing resampling techniques, cost-sensitive learning approaches, and evaluation metric selection. Guo et al. (2017) extended this survey to the deep learning era, covering resampling, cost-sensitive, and ensemble methods across diverse application domains. He and Garcia demonstrated that precision, recall, and the F1-score provide more informative performance assessment than accuracy under class imbalance, and that the Area Under the Receiver Operating Characteristic curve (AUC-ROC) and the Area Under the Precision-Recall curve (AUC-PR) offer complementary perspectives on model discrimination. AUC-ROC measures overall discriminative power across all classification thresholds and is insensitive to class distribution, making it useful for comparing models across datasets with different imbalance ratios. AUC-PR, by contrast, focuses on the minority class and is more sensitive to improvements in detecting the rare positive cases. For AML detection, where the operational cost of false negatives (missed laundering) is high but the cost of false positives (alert fatigue from over-alerting) is also substantial, both metrics are relevant.

The choice of loss function during training is equally important. Standard binary cross-entropy loss treats false positives and false negatives symmetrically, which is inappropriate when the classes are severely imbalanced. Weighted binary cross-entropy, where the minority class contribution is scaled by a factor inversely proportional to its prevalence, provides a standard remedy. However, the magnitude of the weight introduces a new consideration. With a laundering prevalence of approximately 0.1%, the inverse-frequency weight exceeds 1000. This means minority class gradients are three orders of magnitude larger than majority class gradients during training, which has implications for gradient-based optimisation that are discussed in the context of model-specific training in Chapter 3.

Dou et al. (2020) addressed the intersection of GNNs and class imbalance, proposing techniques specific to graph-based fraud detection. Their work demonstrated that standard GNN training procedures can be inadequate under extreme class imbalance because the message-passing mechanism propagates information from both classes, potentially diluting the minority class signal. They proposed adapted training strategies, including class-balanced sampling of training edges, which inform the training protocol adopted in this study. Wu et al. (2024) extended this direction with dual-channel graph convolution and label-aware sampling to jointly address disassortativity and class imbalance in fraud graphs.

**2.6 Research Gap Synthesis**

The literature reviewed in this chapter converges on an identifiable gap. The IBM AML dataset (Altman et al., 2023) was designed and published as a public benchmark for GNN-based AML research, providing a standardised platform on which different architectures can be compared under identical conditions. The dataset paper established baseline results for static GNNs (GCN, GAT, and GraphSAGE) and demonstrated that they outperform non-graph baselines. However, the paper did not evaluate any temporal GNN architectures, explicitly leaving temporal modelling to future work.

Separately, the temporal GNN literature has developed architectures capable of capturing transaction dynamics: snapshot-based approaches such as TemporalGCN and EvolveGCN (Pareja et al., 2020), and continuous-time approaches such as TGN (Rossi et al., 2020). These architectures have been evaluated on social network, citation, and cryptocurrency datasets, but none of the three has previously been applied to a public banking AML benchmark. EvolveGCN was evaluated on Bitcoin OTC and Reddit (Pareja et al., 2020), the temporal GCN of Alarab and Prakoonwit (2023) was applied to the Elliptic Bitcoin dataset, and TGN was evaluated on Wikipedia, Reddit, and Twitter (Rossi et al., 2020). The present study is the first to apply all three to a standardised banking AML benchmark, and the first to apply TGN to AML edge classification in any domain. The single published study applying temporal GNNs to AML (Alarab & Prakoonwit, 2023) used the Elliptic Bitcoin dataset, which represents a fundamentally different transaction domain and used a snapshot-based rather than continuous-time approach.

The consequence of this gap is twofold. Academically, there is no published evidence on whether continuous-time temporal modelling improves AML detection over static GNNs on a standardised banking benchmark. Practically, AML compliance teams lack the comparative empirical evidence needed to assess whether the additional complexity of temporal GNN architectures is justified by a meaningful improvement in detection performance.

The present study addresses this gap by conducting a systematic three-tier comparative evaluation: conventional machine learning classifiers, static GNN architectures, and temporal GNN architectures spanning both snapshot-based and continuous-time approaches, all trained and evaluated on the same IBM AML HI-Small dataset under identical experimental conditions. The theoretical framework established in this chapter provides the basis for the architectural choices and evaluation protocol described in Chapter 3.

---

# Chapter 3: Research Methodology and Tool Development

**3.1 Research Design Overview**

This study is applied research with a tool-development orientation, structured according to the Design Science Research (DSR) paradigm: the established methodology for research whose central output is a purposeful artifact created to address a real-world problem and evaluated against defined objectives (Hevner et al., 2004; Peffers et al., 2007). DSR is appropriate here because the research does not test hypotheses about an existing phenomenon but designs, builds, and evaluates an analytical tool (a GNN-based money-laundering detector) and, through that evaluation, generates transferable knowledge about which classes of model are effective for AML detection and under what conditions. The study follows the six-activity Design Science Research Methodology (DSRM) process model of Peffers et al. (2007), each activity of which maps onto a component of this report, as summarised in Table 3.1.

**Table 3.1: Mapping of the DSRM activities (Peffers et al., 2007) onto this thesis.**

| DSRM activity | Realisation in this thesis |
| ------------- | -------------------------- |
| 1. Problem identification and motivation | Problem statement (Section 1.4) and research-gap synthesis (Section 2.6): no unified three-tier AML comparison exists, and no temporal GNN has been evaluated on the IBM AML benchmark. |
| 2. Define the objectives of a solution | Research objectives (Section 1.5) and the performance indicators for imbalanced detection (AUC-ROC, AUC-PR, F1), derived from the literature in Chapter 2. |
| 3. Design and development | Data engineering, graph construction, and the implementation of nine model architectures (Sections 3.3-3.4). |
| 4. Demonstration | Application of all models to the IBM AML HI-Small benchmark (Chapter 4). |
| 5. Evaluation | Comparative, deployment-realistic (chronological) evaluation, including the ablation and per-slice analyses (Chapter 4). |
| 6. Communication | This report and the accompanying defence, including the practitioner guidance in Chapter 5. |

The DSR paradigm was selected in preference to the Cross-Industry Standard Process for Data Mining (CRISP-DM), the main alternative process model for data-driven projects (Wirth & Hipp, 2000). CRISP-DM offers a well-structured six-phase cycle (business understanding, data understanding, data preparation, modelling, evaluation, and deployment) and is widely used in industrial analytics. However, it is a process model for delivering a data-mining solution to a specific client, not a research methodology: it does not require the explicit identification of a research gap, the derivation of design objectives from the literature, or the production of generalisable knowledge as an output, each of which is central to a research thesis and to the comparative contribution of this study. DSR retains a comparable build-and-evaluate rigour while additionally framing the artifact as a vehicle for answering research questions and contributing to the knowledge base, which makes it the more appropriate foundation. The one respect in which CRISP-DM is more prescriptive, its treatment of data preparation, is incorporated pragmatically within the DSR design-and-development activity through the data-engineering pipeline detailed in Section 3.3.

Operationally, the design comprises five stages that instantiate these activities. First, a structured literature study established the theoretical foundation and confirmed the research gap (problem identification; objectives). Second, the IBM AML HI-Small dataset was selected through systematic comparison against an alternative candidate and subjected to data engineering (design and development). Third, nine model architectures spanning three tiers were implemented: three conventional supervised classifiers (Logistic Regression, Random Forest, XGBoost), three static GNNs (GCN, GAT, GraphSAGE), and three temporal GNNs (TemporalGCN, EvolveGCN-H, TGN) (design and development). Fourth, all models were trained and evaluated under identical, deployment-realistic conditions using metrics appropriate for heavily class-imbalanced data (demonstration; evaluation). Fifth, the empirical findings were analysed comparatively and translated into practitioner guidance and an assessment of deployment preconditions (evaluation; communication). The design intentionally spans three tiers, rather than comparing GNN variants alone, to isolate the contribution of graph structure and of temporal modelling: Tier 1 (conventional ML) establishes the performance achievable without graph structure, Tier 2 (static GNNs) measures the gain from relational modelling, and Tier 3 (temporal GNNs) measures the additional gain from temporal modelling and contrasts snapshot-based with continuous-time approaches. This allows the study to answer not only which model performs best, but why.

During implementation, the scope of the temporal modelling tier expanded beyond the single architecture (EvolveGCN) specified in the research plan. Preliminary experiments revealed that snapshot-based temporal models underperformed the static GCN, which prompted the addition of TemporalGCN to verify whether the limitation was specific to EvolveGCN's weight-space evolution or inherent to the snapshot paradigm, and subsequently the addition of continuous-time TGN to test whether finer temporal granularity could overcome the limitation. This expansion was a data-driven methodological decision, grounded in empirical observations made during the research process. It is documented here transparently as part of the methodological narrative.

**3.2 Dataset: IBM AML HI-Small**

**3.2.1 Dataset Selection and Justification**

The dataset used in this study is the IBM Transactions for Anti-Money Laundering dataset (Altman et al., 2023), publicly available on Kaggle (https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml). The IBM AML dataset was chosen over the main alternative, the Synthetic AML Dataset (SAML-D; Oztas et al., 2023), for two reasons. First, the IBM AML dataset is structured natively as a graph: a dedicated accounts file defines each unique account as a persistent entity, and a transactions file captures directed interactions between accounts. This maps directly onto the node-and-edge representation required for GNN-based analysis. SAML-D, by contrast, is a flat tabular dataset without an explicit account-level structure, making the construction of stable node identities (a prerequisite for temporal GNNs) a non-trivial and ambiguous preprocessing step. Second, the IBM AML dataset's laundering patterns are derived from FATF-documented AML typologies, including structuring, layering, and fan-in/fan-out schemes (Altman et al., 2023; FATF, 2023), ensuring the synthetic patterns reflect real-world regulatory knowledge. The dataset was published at NeurIPS 2023 specifically as a public benchmark for GNN-based AML research (Altman et al., 2023).

The dataset is available in four variants (HI/LI combined with Small/Medium). This study uses the HI-Small variant (518,581 accounts, 5,078,345 transactions, 5,177 laundering, 0.102% prevalence). The HI variants contain a higher laundering ratio, providing more positive cases for training. The Small variant was chosen for computational feasibility: all models were trained on CPU, and the Medium variants (tens of millions of transactions) would have made full training runs for all nine architectures infeasible within the project timeline. The four variants share an identical data-generating process, so architectural findings from HI-Small are expected to generalise, though empirical verification on larger variants is noted as future work.

**3.2.2 Dataset Characteristics**

The HI-Small dataset comprises two CSV files. The accounts file contains 518,581 rows with five fields: Bank Name, Bank ID, Account Number, Entity ID, and Entity Name. The transactions file contains 5,078,345 rows with eleven fields: Timestamp, From Bank, Account (from), To Bank, Account (to), Amount Received, Receiving Currency, Amount Paid, Payment Currency, Payment Format, and Is Laundering.

Account identity is composite, formed by concatenating Bank ID and Account Number. This composite key is consistent across the accounts and transactions files, ensuring unambiguous mapping between account entities and their transaction history. Transaction timestamps are provided as string-formatted dates (for example, "2022/09/01 00:20"), which are parsed to Unix seconds for temporal modelling. The timestamp range spans approximately 18 days (2022/09/01 through 2022/09/19), representing a compressed but transaction-dense financial activity period.

The accounts file characterises each of the 518,581 accounts with bank membership, a unique account identifier, and an entity type derived from the Entity Name field. Entity types include corporations, individuals, shell companies, and other categories. These entity types apply to both senders and receivers: every transaction links a sending account to a receiving account, and each account's entity type and bank affiliation are known from the accounts file. The account-level features (Section 3.3.1) are derived from this information combined with aggregated transaction statistics computed from the training set.

The payment format field contains seven categories: ACH, Bitcoin, Cash, Cheque, Credit Card, Reinvestment, and Wire. The currency fields include 15 currency codes. These categorical variables form the basis for one-hot encoded edge features, as described in Section 3.3.1.

The class distribution is severely imbalanced: 5,177 laundering transactions among 5,078,345 total, yielding a prevalence of 0.1019% and a positive-to-negative ratio of approximately 1:980. This extreme imbalance has direct implications for loss function design, evaluation metric selection, and model training strategy, as discussed in Sections 3.5.1 and 3.5.3.

**3.3 Data Engineering and Graph Construction (SQ1)**

This section addresses SQ1 by describing and justifying the graph construction design decisions required to represent financial transaction data for static and temporal GNN-based analysis.

**3.3.1 Feature Engineering**

Feature engineering was performed on the raw transaction and account data to construct node-level and edge-level feature matrices suitable for GNN input. Features were computed from the training set only to prevent data leakage from validation and test partitions into model training. Categorical variables were encoded using one-hot or ordinal encoding fitted on the training set and applied to all splits.

**Node features.** Twelve node-level features were constructed for each account, drawn from three sources:

1. Bank and entity identifiers: the account's bank name and bank ID are label-encoded (each unique bank assigned an integer, then standardised to zero mean and unit variance), and the entity type is extracted from the entity name field (for example, "Corporation #33520" becomes "Corporation") and label-encoded. This captures whether an account belongs to a corporation, an individual, or another entity category.
2. Transaction statistics: ten aggregated statistics computed from the account's transaction history within the training set, including out-degree and in-degree (number of transactions sent and received), total and average amounts sent and received, and number of unique counterparties. All count and amount features are log1p-transformed to compress their long-tailed distributions.
3. All features are standardised (z-scored) so that zero represents the mean across all accounts.

To make this concrete, consider an account with the following profile after feature engineering: high out-degree (+1.89, roughly 47 outgoing transactions, well above the mean), low in-degree (-0.22, roughly 3 incoming transactions), high total amount sent (+0.67, approximately $234,000), and low amount received (-0.15, approximately $8,200). This account sends far more money than it receives, to many more counterparties than it receives from: a fan-out pattern characteristic of structuring behaviour. The node features encode this behavioural signature without the model needing to traverse the graph.

**Edge features.** Twenty-eight edge-level features were constructed for each transaction:

1. Amount: the log1p-transformed amount received and amount paid (two features). Log transformation compresses the long-tailed amount distribution, preventing a small number of very large transactions from dominating the feature space.
2. Cyclic time: four features encoding the hour of day and day of week as sine and cosine pairs. Rather than representing 14:30 as the scalar 14.5 (where 23:59 and 00:01 appear 23 hours apart), the sine and cosine of (2 * pi * hour / 24) place all times on a circle where adjacent moments are always close. The same principle applies to day of week: Monday and Sunday are neighbours on the 7-day circle, which a linear encoding would not capture.
3. Payment format: seven one-hot columns, one per category. A transaction paid via ACH produces the column pattern [0, 1, 0, 0, 0, 0, 0]; a cheque produces [0, 0, 1, 0, 0, 0, 0]; a domestic wire produces [0, 0, 0, 0, 1, 0, 0]. Exactly one column is 1 for each transaction; all others are 0.
4. Currency: fifteen one-hot columns following the same principle, one per currency code. A USD transaction sets the USD column to 1; a EUR transaction sets the EUR column to 1.

The seven payment format columns and fifteen currency columns are left unstandardised (since one-hot values are already bounded to {0, 1}), while the amount and cyclic time features are standardised to zero mean and unit variance. This mixed encoding strategy preserves the interpretability of categorical features while normalising the scale of continuous features. The complete list of all 12 node features and 28 edge features with their types and computation methods is provided in Appendix A.

**Comparison with alternatives.** An alternative feature engineering approach would have been to use learned node embeddings (for example, Node2Vec; Grover & Leskovec, 2016) rather than hand-crafted features. The advantage of learned embeddings is that they can capture structural properties of the graph that hand-crafted features might miss, such as community membership and higher-order neighbourhood patterns. The disadvantage is that they require a separate pretraining stage, add computational overhead, and produce features that are less interpretable. Hand-crafted features were selected because they are directly interpretable, grounded in domain knowledge about what distinguishes laundering accounts (high counterparty count, unusual temporal patterns, transaction volume extremes), and computationally lightweight. The 28-dimensional edge feature vector and 12-dimensional node feature vector are compact enough to keep model parameter counts manageable while providing sufficient signal for the classification task.

**3.3.2 Graph Construction for Static and Temporal Models (SQ1)**

Two graph construction strategies were employed, corresponding to the static and temporal modelling paradigms.

**Static graph construction.** For static GNNs and conventional baselines, a single directed graph was constructed in which each unique account identifier maps to a node and each transaction maps to a directed edge from the originating account to the destination account. The graph was built using PyTorch Geometric (PyG; Fey & Lenssen, 2019). Edge indices, node feature matrices, edge feature matrices, and edge labels were assembled into a PyG Data object. Edge directions were preserved to capture the inherently directional nature of financial transactions: a transfer from account A to account B is structurally and semantically different from a transfer from B to A.

**Temporal snapshot construction.** For snapshot-based temporal GNNs (TemporalGCN and EvolveGCN-H), the transaction timeline was divided into 12 windows using a quantile-based strategy: window boundaries were placed such that each window contains approximately the same number of transactions. This strategy was chosen over fixed-width (equal time duration) windows because transaction density in the dataset is heavily skewed, with some periods containing orders of magnitude more transactions than others. Fixed-width windows would produce snapshots with highly variable edge counts, causing some snapshots to be too sparse for meaningful graph convolution and others to be too dense for efficient computation. Quantile-based windows ensure that each snapshot has sufficient and comparable edge density, which is important for stable GNN training across snapshots.

The 12-window granularity was chosen to balance temporal resolution against per-snapshot edge density. With approximately 5 million transactions total, each snapshot contains roughly 420,000 transactions, providing adequate density for GCN operations. A larger number of snapshots (for example, 24 or 48) would increase temporal resolution but reduce per-snapshot edge counts and increase training time linearly with the number of snapshots. The sensitivity of model performance to snapshot granularity was not systematically investigated, which is noted as a limitation.

**Continuous-time data construction.** For TGN, the temporal data builder processes transactions in strict chronological order without binning into windows. Each edge retains its individual timestamp as a continuous value (Unix seconds). The data is chronologically sorted and divided into training (70% of edges, earliest in time), validation (15%), and test (15%, latest in time) partitions by index. This preserves the natural temporal ordering: the model is trained on past transactions and evaluated on future ones.

**Design justification across paradigms.** The use of three different graph construction strategies (static, snapshot, continuous-time) is not an inconsistency but a reflection of the different modelling paradigms. Static GNNs require a single graph and gain no benefit from temporal information in the data structure. Snapshot temporal GNNs require a sequence of static graphs and benefit from temporal binning. TGN requires individual timestamps and would be degraded by binning. Using the appropriate data representation for each paradigm ensures that each model is evaluated under the conditions for which it was designed, enabling a fair comparison of the paradigms themselves rather than of suboptimal instantiations.

**3.3.3 Chronological Data Splitting**

Every model tier is evaluated under a single, uniform chronological (time-based) split. After all transactions are sorted by timestamp, the earliest 70% of edges form the training set, the next 15% the validation set, and the latest 15% the test set, partitioned by index so that all training edges precede all validation edges, which in turn precede all test edges. For the snapshot temporal models the same ordering is expressed at snapshot granularity: the 12 chronological snapshots are assigned as snapshots 0 through 7 to training, snapshot 8 to validation, and snapshots 9 through 11 to testing. Applying one protocol across the conventional baselines, the static GNNs, and the temporal GNNs alike ensures that every model is measured on the same task, so that differences in performance reflect the models themselves rather than differences in how past and future transactions are mixed.

This chronological protocol evaluates every model under deployment-realistic conditions. In a production AML system a model is trained on historical data and must detect laundering in future transactions. A random split that mixes past and future edges across training and test introduces a subtle leakage, in which the model sees future edges during training and past edges during testing, inflating performance estimates relative to real deployment. Because several published AML GNN studies adopt random splits (Altman et al., 2023; Weber et al., 2019), the chronological protocol used here yields a more deployment-relevant, and deliberately harder, performance estimate, a point revisited in the cross-model comparison in Chapter 4.

A consequence of chronological splitting is that the class distribution varies across partitions, since the laundering ratio is not constant over time. In the IBM AML HI-Small dataset, the laundering ratio increases from approximately 0.01% in the earliest time window to 0.30% in the latest. The chronological split means that the test set has a higher laundering prevalence than the training set, which is both realistic (laundering patterns may intensify over time in a real system) and challenging (the model is evaluated on a distribution that differs from its training distribution). The pos_weight for loss computation was computed from the training set only, consistent with the principle that no test-set information may influence model training.

**3.4 Model Architectures (SQ2 and SQ3)**

This section addresses SQ2 by describing the implementation of each model architecture and justifying the design choices with reference to the theoretical framework established in Chapter 2.

**3.4.1 Conventional ML Baselines**

Three supervised classifiers were implemented as baselines that operate on flat feature vectors without access to graph structure: Logistic Regression, Random Forest, and XGBoost. These models were selected to represent a progression of complexity and to establish the performance floor against which GNN-based models are compared, directly addressing SQ3.

**Logistic Regression** was implemented using scikit-learn (Pedregosa et al., 2011) with L2 regularisation and the lbfgs solver. Class imbalance was addressed via class_weight="balanced", which automatically weights the minority class inversely proportional to its frequency in the training set. No sample_weight was applied, avoiding the double-weighting issue in which class_weight and sample_weight simultaneously scale the minority class loss, effectively squaring the intended penalty.

**Random Forest** (Breiman, 2001) was implemented with 200 estimators, a maximum depth of 20, a minimum of 10 samples per leaf, and class_weight="balanced". The depth cap and leaf-size floor constrain the trees enough to limit memorisation of the majority class while retaining sufficient capacity to model the minority-class decision boundary.

**XGBoost** (Chen & Guestrin, 2016) was implemented with 300 estimators, a maximum tree depth of 8, and a learning rate of 0.05, with early stopping that monitors validation-set log loss and halts after 20 rounds without improvement. Validating early stopping on held-out data rather than on the training set is methodologically important: training-set monitoring provides no signal about generalisation and invites overfitting.

All three baselines receive the identical 28-dimensional edge feature vectors used by the GNN models. The key difference is that the baselines treat each edge independently, while the GNNs additionally receive node features and the graph adjacency structure. Any performance difference between the baselines and the GNNs can therefore be attributed to the graph structural information, since the edge-level input features are held constant.

**3.4.2 Static GNNs**

The three static GNN architectures described in Section 2.3.1 were implemented using PyTorch Geometric (Fey & Lenssen, 2019). All three share a common architectural template: an edge classification model consisting of node encoding layers, edge feature projection, and a classifier head.

**GCN (Kipf & Welling, 2017).** The implementation uses two GCN convolutional layers with hidden dimension 128 and ReLU activation. Each GCN layer applies the symmetric normalised graph Laplacian convolution to propagate node features across edges. After convolution, the final-layer node embeddings for the source and destination nodes of each edge are concatenated with the projected edge features and passed through a two-layer MLP classifier with dropout (p=0.3) to produce a scalar logit per edge. The total parameter count is 63,489.

**GAT (Veličković et al., 2018).** The implementation uses two GAT convolutional layers with hidden dimension 128 and a single attention head. The original GAT paper reported that multi-head attention (typically 4 or 8 heads) was important for stable training. However, preliminary experiments with 4 heads on the full HI-Small graph (5 million edges) caused memory exhaustion on CPU. With a single head, the model has 64,001 parameters and completed training successfully. The use of a single head likely reduces the expressiveness of the attention mechanism, since the model cannot attend to different relational patterns in parallel, but trade-offs of this kind are unavoidable when training large-graph models on CPU-constrained hardware.

**GraphSAGE (Hamilton et al., 2017).** The implementation uses two SAGEConv layers with hidden dimension 128 and mean aggregation. Mean aggregation was chosen over max or LSTM aggregation for computational efficiency: max aggregation discards distributional information about neighbour features, and LSTM aggregation imposes an arbitrary ordering on an unordered neighbour set. The implementation uses neighbourhood sampling with a fixed sample size and L2 normalisation of embeddings, which the original paper found important for training stability. The total parameter count is 81,409.

**3.4.3 Snapshot Temporal GNNs**

**TemporalGCN.** The implementation consists of a shared two-layer GCN (128 hidden dimensions) that processes each snapshot independently, combined with a GRU (Cho et al., 2014) that evolves per-node hidden states across the snapshot sequence. After the GCN produces node embeddings for snapshot t, the GRU updates each node's hidden state as a learned combination of its previous state and the new GCN output. The GRU hidden dimension matches the GCN output dimension (128). The edge classifier concatenates source and destination node states with edge features and passes them through a two-layer MLP. The key difference from the static GCN is that the node states feeding into the edge classifier are not raw GCN outputs but GRU-evolved states that incorporate information from all preceding snapshots. The total parameter count is 162,561.

**EvolveGCN-H (Pareja et al., 2020).** EvolveGCN-H evolves the GCN weight matrices across snapshots rather than per-node states. The GCN weight matrix at snapshot t is expressed as the sum of a base weight matrix and a low-rank adaptation: W\_t = W\_base + A\_t @ B\_t, where A\_t and B\_t are low-rank factors. A GRU receives the mean-pooled node embedding from the previous snapshot as input and produces updated weight factors for the next snapshot. The low-rank dimension is a critical hyperparameter: it determines both the expressive capacity of the weight adaptation and the total parameter count.

The low-rank dimension is set to 2. The parameter count grows rapidly with rank, because the GRU hidden-state dimension scales as the rank multiplied by the sum of the input and output GCN dimensions; at higher ranks the model exceeds available memory. Even at rank 2 the model has 2,213,673 parameters, far more than any other model in this study, while its expressive capacity for this task remains limited (Section 4.5). This parameter growth is inherent to evolving the weight matrices rather than the node states.

**3.4.4 Continuous-Time TGN**

The Temporal Graph Network (TGN; Rossi et al., 2020) was implemented with a custom EMA-based memory module, rather than the GRU-based TGNMemory provided by PyTorch Geometric. The design choices behind this implementation, including the departure from the default memory module, are set out below together with their rationale.

**Architecture.** The TGN model comprises five components. First, a TimeEncoder maps raw timestamp differences to a 16-dimensional sinusoidal encoding, using the sine-cosine transformation from the Transformer positional encoding (Vaswani et al., 2017) applied at 16 logarithmically spaced frequencies. This encoding captures temporal patterns at multiple scales, from sub-minute to multi-day intervals. Second, a NodeProjection MLP maps each node's raw features to an initial 128-dimensional memory state. Third, a MessageProjection MLP transforms the concatenation of source memory, destination memory, edge features, and time encoding into a 128-dimensional message vector representing the information content of the current interaction. Fourth, an EdgeClassifier MLP concatenates source memory, destination memory, projected edge features, time encoding, and the projected message, and passes the result through a two-layer MLP with dropout (p=0.3) to produce a scalar logit. Fifth, an EMAMemory module maintains and updates per-node memory states. The total parameter count is 289,217 (using the default memory_dim=128, time_dim=16), or 85,905 in the more compact configuration used for the final results (memory_dim=64, time_dim=8).

**EMA memory design.** The memory update follows an exponential moving average: m\_new = beta * m\_old + (1 - beta) * aggregated\_message, with beta=0.85. EMA was chosen over the GRU-based memory in PyG's TGNMemory for three reasons. First, PyG's TGNMemory stores a `last_update` buffer as a Long tensor, but the TGN model processes Float timestamps, causing a dtype mismatch and in-place mutation errors during the backward pass. Second, PyG's TGNMemory separates the forward pass (reading memory) from the `update_state` method (writing memory), with `update_state` called after `loss.backward()`. This means the GRU inside TGNMemory never receives gradients: it is updated but never trained. Third, EMA memory is simpler: it has a single parameter (beta) versus the GRU's six (input, forget, and output gate weights and biases), reducing overfitting risk while providing the same functional behaviour of accumulating interaction history with controlled decay.

The EMA update occurs inside the forward pass, ensuring the message projection MLP receives gradients through the memory update pathway during training. However, predictions are always made using the old memory state (before the current batch's update), preventing the model from exploiting information from the current interaction that would not be available at prediction time in deployment. This design choice was the resolution of a data leakage issue discovered during development: when predictions used new memory (containing the current batch's messages), the model learned to exploit its own edge features in memory, achieving misleadingly high training performance that collapsed during evaluation because the evaluation path correctly used old memory. Always predicting with old memory ensures consistency between training and evaluation behaviour.

**Gradient clipping.** The pos_weight for TGN's binary cross-entropy loss is computed from the training set as the inverse frequency of the positive class, yielding a value of approximately 1244. With the `pos_weight_mult` parameter set to 0.01, the effective pos_weight is 12.4. This means minority class gradients are 12.4 times larger than majority class gradients during training. When gradient clipping was enabled (default value 1.0), nearly all positive-class gradients exceeded the clipping threshold and were truncated. The model received uniformly clipped gradients for laundering examples, regardless of whether the example was easy or hard, preventing it from learning a discriminative boundary for the minority class. Disabling gradient clipping entirely (`grad_clip=0`) resolved this issue: Epoch 1 validation AUC-ROC increased from 0.794 to 0.934.

This finding has a methodological implication beyond this specific implementation: when using weighted loss functions with large positive class weights, gradient clipping interacts destructively with class imbalance by selectively suppressing the minority class learning signal. The standard practice of applying gradient clipping as a stability measure must be reconsidered under extreme class imbalance.

**3.4.5 Design Justification**

The selection of nine architectures across three tiers is justified by the research objective of isolating the contribution of graph structure and temporal modelling to detection performance. A narrower comparison, for example, comparing only GCN against EvolveGCN, would identify which temporal model performs better but could not determine whether either outperforms non-graph baselines. A broader comparison, adding architectures such as Graph Isomorphism Networks (Xu et al., 2019) or temporal attention-based models, would provide more comprehensive coverage but at the cost of computational feasibility: each additional architecture requires a full training cycle on 5 million edges.

The specific static architectures (GCN, GAT, GraphSAGE) were chosen because they represent the three dominant design philosophies in static GNN research: spectral convolution, attention-based aggregation, and sampling-based inductive learning. They are also the architectures for which the IBM AML dataset paper (Altman et al., 2023) reported baseline results, enabling direct comparison. The temporal architectures (TemporalGCN, EvolveGCN-H, TGN) were chosen to span the two temporal modelling paradigms: snapshot-based (with both state-space and weight-space evolution) and continuous-time. This coverage ensures that the study's findings about temporal granularity are not specific to a single architecture or paradigm.

**3.5 Training and Evaluation Protocol (SQ2 and SQ3)**

This section describes the training and evaluation protocol shared across all models, addressing SQ3 by establishing the conditions under which the comparative evaluation is conducted.

**3.5.1 Loss Functions and Class Weighting**

All models were trained to minimise weighted binary cross-entropy loss. For a training set with N\_neg legitimate transactions and N\_pos laundering transactions, the pos_weight is computed as N\_neg / N\_pos. For the HI-Small training partition, this yields a value of approximately 1244.

For static GNNs and snapshot temporal models, a pos_weight_multiplier of 0.1 was applied, yielding an effective pos_weight of approximately 124. For TGN, a lower multiplier of 0.01 was used (effective pos_weight approximately 12.4), following empirical observation that the larger multiplier produced unstable training in combination with the per-batch memory updates. The difference arises because TGN processes edges in batches with online memory updates, creating a noisier gradient environment than the full-graph training of static GNNs. A lower pos_weight reduces the variance of minority class gradients, stabilising training.

The use of pos_weight rather than alternative class imbalance handling techniques (oversampling the minority class, undersampling the majority class, or using focal loss) was chosen for two reasons. Oversampling and undersampling modify the effective training distribution and can distort the temporal structure of the data: oversampling repeats transactions, creating artificial temporal dependencies, while undersampling discards potentially informative legitimate transactions. Focal loss (Lin et al., 2017) down-weights easy examples to focus training on hard ones, which is conceptually appealing for AML but introduces an additional focusing hyperparameter. Weighted cross-entropy is the simplest and most transparent approach, and its single parameter (pos_weight) has a clear interpretation.

**3.5.2 Hyperparameter Configuration**

Hyperparameters for the conventional and static-GNN models were set from architectural defaults in the original papers, adjusted where necessary for training stability on this dataset. No automated hyperparameter optimisation (grid, random, or Bayesian search) was performed, which is acknowledged as a limitation. The full configurations are listed in Appendix E (Table E.1).

The continuous-time TGN required more deliberate configuration selection. Rather than ad hoc tuning, a sequence of configurations was trained under the chronological protocol and compared on validation AUC-ROC and AUC-PR before the final model was fixed and the test set evaluated once. Six development runs (Appendix E, Table E.2) varied the model capacity, the class-weight multiplier applied to pos_weight, the learning rate, and the gradient-clipping setting. Two findings determined the outcome. First, gradient clipping had to be disabled: under the large positive-class weight required by the 0.1% prevalence, clipping suppressed the minority-class gradient and the model failed to learn the positive class, leaving validation AUC-PR at the prevalence floor (Section 3.4.4); five of the six runs left the minority class effectively undetected, and only once clipping was removed did a configuration learn it, reaching a best validation AUC-ROC of 0.946. Second, a subsequent compact configuration (memory dimension 64, time-encoding dimension 8; 85,905 parameters) matched that larger configuration on validation while using roughly half the parameters, and was therefore adopted as the final reported model (learning rate 0.003, pos_weight multiplier 0.01, gradient clipping disabled, EMA beta 0.85). The candidate runs and their validation outcomes are given in Appendix E (Table E.2).

**3.5.3 Evaluation Metrics and Threshold Calibration**

All models were evaluated using five metrics: AUC-ROC, AUC-PR, Precision, Recall, and F1-score. As established in Section 2.5, these metrics are appropriate for heavily class-imbalanced data because they are not inflated by the majority class, unlike accuracy.

Precision, Recall, and F1-score are threshold-dependent: they are computed at a specific classification threshold. Reporting these metrics at a single default threshold (0.5) is standard practice but can be misleading when the optimal threshold for the minority class differs substantially from 0.5, as is typical under extreme class imbalance. To address this, each model's classification threshold was calibrated on the validation set by selecting the threshold that maximised validation F1-score. Both default-threshold (0.5) and calibrated-threshold metrics are reported in Chapter 4. The calibrated threshold was then applied to the test set for the final evaluation. This calibration procedure ensures that threshold-dependent metrics reflect each model's best achievable performance rather than an arbitrary cutoff.

For TGN, an additional evaluation was performed: per-time-slice analysis. The chronologically ordered test set was divided into 12 equal slices, and metrics were computed independently for each slice. This analysis tests whether model performance improves as more interaction history accumulates in per-node memory, providing evidence for or against temporal generalisation. A model whose performance is flat across slices shows no benefit from memory accumulation; a model whose performance improves monotonically across slices demonstrates that per-node memory captures useful behavioural signal over time.

**3.6 Ethical Considerations, Validity, and Reliability**

**Ethical considerations.** This research uses a synthetic dataset (IBM AML HI-Small) that does not contain real personal or financial information. No primary data collection from human subjects was conducted. The dataset is publicly available and was created specifically for academic research purposes (Altman et al., 2023). The laundering labels are synthetic and do not represent accusations against real individuals or institutions.

Two ethical considerations nonetheless apply. First, the AML detection tool developed in this research could, if deployed, contribute to automated decision-making with significant consequences for individuals whose accounts are flagged as suspicious. The tool is an analytical prototype, not a production system, and its outputs should be understood as decision support for human compliance analysts, not as automated determinations of criminal activity. This distinction is important: the models presented here detect patterns statistically associated with laundering, not laundering itself. Second, the model's performance characteristics have fairness implications. If the underlying transaction data reflects biases in which accounts or transaction patterns are flagged as suspicious, the model may amplify those biases. The IBM AML dataset's laundering patterns are derived from FATF typologies rather than from real-world enforcement data, which mitigates but does not eliminate this concern.

**Research validity.** Because the central claims rest on quantitative measurements, the design incorporates explicit safeguards for internal, external, and construct validity, consistent with the evaluation rigour required of design science research (Hevner et al., 2004; Sekaran & Bougie, 2019).

*Internal validity.* The principal threat is information leakage from test data into training. Several controls address this: all feature encoders and scalers (Section 3.3.1) are fitted on the training partition only and applied unchanged to validation and test; the class weight is computed solely from the training partition; and decision thresholds are calibrated on validation and only then applied to test (Section 3.5.3). For the continuous-time TGN, predictions are always computed from the node memory as it stood before the current batch, so a transaction is never used to predict its own label; where memory is warm-started for the temporal-generalisation analysis (Section 4.5.3), it is advanced only over training and validation edges that precede the test period, preserving past-to-future ordering. Critically, all three tiers are evaluated under a single chronological protocol in which models are trained on the earliest transactions and tested on the latest. Using one protocol across all tiers removes the cross-tier comparability threat that a mixture of random and chronological splits would introduce, so that observed performance differences are attributable to the models rather than to differences in evaluation difficulty. One consequence of chronological splitting is that laundering prevalence differs across partitions, rising over the observation window; this is reported explicitly (Sections 3.3.3 and 4.1) and accounted for when interpreting per-slice results, so that a rise in precision-recall performance is not mistaken for a modelling effect when it reflects a change in class balance.

*External validity.* The clearest limitation is the use of a synthetic benchmark. The IBM AML dataset is generated from FATF-documented typologies (Section 3.2), which grounds its patterns in regulatory knowledge, but it remains a simulation: performance on it estimates performance against those encoded typologies, not against the full diversity of real institutional behaviour. The study is further confined to a single variant (HI-Small), a single simulated financial system without cross-institutional transfers, and an eighteen-day window; generalisation to larger and lower-prevalence variants, to multi-institutional networks, and to longer horizons is not established and is identified as future work (Section 5.5). A specific artifact of the generator is a small tail of roughly 1,100 transactions after the main activity period that carries a very high laundering rate and accounts for close to half of all test-set positives; this raises the aggregate test metrics relative to the realistic low-prevalence regime, which is why the per-slice analysis (Section 4.5.3) is reported, so that performance on the sparse, operationally representative slices can be read separately from this dense tail. The reported figures should therefore be read as a rigorous, like-for-like comparison of architectures under controlled conditions, not as absolute performance guarantees for production, a distinction reflected in the framing of the practical contribution (Sections 1.4 and 5.3).

*Construct validity.* The ability to distinguish laundering from legitimate transactions under extreme imbalance is measured with AUC-ROC, AUC-PR, precision, recall, and F1 rather than accuracy, which is dominated by the majority class and would rate a trivial all-negative classifier at 99.9% (Section 2.5). AUC-PR is emphasised as the measure most sensitive to minority-class detection. The construct learned is the dataset's synthetic laundering label, which is a proxy for real laundering rather than laundering itself.

**Reliability and reproducibility.** All experiments used a fixed random seed (42) across NumPy, PyTorch, and Python's random module, and all data splits are deterministic (chronological sort followed by index-based partitioning at 70/15/15 ratios), so re-running any experiment with the same arguments reproduces identical results. Reliability in the stronger sense of stability across configurations is addressed for the two models central to the comparison, GCN and TGN, through repeated runs across multiple seeds and a sensitivity analysis over the principal hyperparameters (Section 4.7); the reported figures for these models are accompanied by the observed variation. The selection of the final TGN configuration from a set of candidate configurations evaluated on the validation set is documented transparently (Section 3.5.2 and Appendix E), so that the single reported result is not mistaken for a configuration-independent property of the architecture.

The complete software environment with pinned versions, reproduction commands, and data-split documentation is provided in Appendix B.

**Tool documentation.** The tool's architecture, module structure, and development process are documented within this chapter (Sections 3.3-3.5) and in the reproducibility guide in Appendix B. Appendix A provides the complete feature specification (12 node features, 28 edge features) referenced in Section 3.3.1. Appendix D contains the full results tables for every model. Together, these materials are intended to enable an independent researcher or practitioner to understand, reproduce, and adapt the tool.

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

The three conventional supervised classifiers operate on flat edge feature vectors without access to graph structure or temporal information, establishing the performance floor against which GNN-based models are compared. Their full test-set metrics are given in Appendix D (Table D.1), training and validation results in Appendix F, and all nine models are consolidated in the cross-model leaderboard (Table 4.5).

XGBoost is the strongest conventional classifier on AUC-PR (0.1460), the metric of primary interest under extreme class imbalance, followed closely by Random Forest (0.1249). Logistic Regression matches both on AUC-ROC (0.9378) but achieves far lower AUC-PR (0.0378), indicating that its strong ranking performance does not translate into effective identification of the minority class. At the default 0.5 threshold, Logistic Regression achieves very high recall (0.9295) at near-zero precision (0.0135): it flags almost all laundering transactions, but at a false-positive rate that would be operationally unworkable. Thresholds calibrated for F1 on the validation set improve the operating point of all three (for example, XGBoost reaches F1 0.161 at threshold 0.94, and Random Forest F1 0.186 at threshold 0.90), but do not change the ranking (Appendix F).

The two tree ensembles are close because both can exploit the non-linear interactions in the 28-dimensional edge feature vector (amount, payment format, currency, and temporal encodings), whereas the linear model cannot. The key insight from the baseline tier is nonetheless the ceiling it exposes: even the best conventional classifier reaches only AUC-PR 0.146, and at the default threshold XGBoost flags roughly 40 false positives for every genuine alert. This reflects the limitation identified in Section 2.2: without access to relational information, individual transaction features carry only partial signal for distinguishing laundering from legitimate activity.

**4.3 Static GNN Results: Graph Structure Without Time (Tier 2)**

The three static GNN architectures incorporate graph structure through message passing but treat all transactions as simultaneously present, without temporal ordering. They are evaluated on the same chronological split as every other model, so their numbers are directly comparable to the baselines above and the temporal models below; full metrics are given in Appendix D (Table D.2) and the leaderboard (Table 4.5).

GCN is the strongest static GNN, achieving AUC-ROC 0.9708 and AUC-PR 0.2056 with only 63,489 parameters. At its calibrated threshold of 0.673, GCN detects 52.3% of laundering transactions at 12.1% precision. Compared to the best baseline (XGBoost, AUC-PR 0.1460), GCN adds 0.0596 AUC-PR, a 41% relative improvement, confirming that graph structural information contributes measurable detection value beyond what flat features provide. Section 4.4 isolates this contribution more directly through an ablation.

GraphSAGE achieves the lowest static GNN performance (AUC-ROC 0.9452, AUC-PR 0.0412), below the XGBoost baseline on AUC-PR. Mean aggregation with neighbourhood sampling, while computationally efficient, appears to lose discriminative signal. In a graph where laundering accounts are structurally distinctive (high degree, unusual counterparty patterns, Section 4.1), averaging neighbour features may dilute the very signal the model needs to detect. Max or LSTM aggregation might preserve more of this signal at increased computational cost.

GAT reaches AUC-ROC 0.9575 but an AUC-PR of only 0.0912, below both GCN and the XGBoost baseline (0.1460), though above GraphSAGE. It is evaluated with single-head attention: multi-head attention over the full five-million-edge graph is prohibitively memory-intensive, its cost scaling with the number of edges multiplied by the number of heads, and the single-head form has correspondingly limited capacity to learn multiple relational patterns in parallel, which the original GAT formulation identifies as important (Velickovic et al., 2018). That a memory-bounded single-head GAT underperforms the simpler spectral convolution of GCN is consistent with the view that, at this graph scale, the cost of dense attention is not repaid by a commensurate gain in detection quality. The memory behaviour of attention on large graphs is discussed further in Section 5.4.

Comparing GCN to the original IBM AML dataset paper (Altman et al., 2023), the AUC-ROC reported here (0.9708) is broadly consistent with their findings, though direct numeric comparison is complicated by differences in feature construction and evaluation protocol.

**4.4 Isolating the Contribution of Graph Structure**

A central premise of this thesis is that graph structure adds detection signal beyond what hand-crafted features provide. Section 4.3 shows that the best static GNN outperforms the best flat-feature baseline, but that comparison confounds two differences: the GNN sees the graph, and it also uses a different model class. To isolate the value of graph structure specifically, an ablation was run in which a single model class (gradient-boosted trees, the strongest baseline) is given three progressively richer flat feature sets, and compared against the GCN that uses the same information through message passing. All four settings use the identical chronological split.

**Table 4.2: Graph-versus-features ablation (chronological split). Node features are the 12-dimensional per-account features; edge features are the 28-dimensional per-transaction features; message passing is the GCN of Section 4.3.**

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

The two snapshot-based temporal architectures discretise the eighteen-day window into 12 snapshots and evolve either node states (TemporalGCN) or GCN weights (EvolveGCN-H) across them; their full metrics are given in Appendix D (Table D.3) and the leaderboard (Table 4.5).

TemporalGCN achieves AUC-ROC 0.9514 with 162,561 parameters. Despite incorporating temporal information through GRU-evolved node states across 12 snapshots, it underperforms the static GCN (AUC-ROC 0.9708, AUC-PR 0.2056) on the same split. Since the evaluation protocol is now identical, the gap cannot be attributed to an easier test set for the static model; it points instead to the snapshot resolution itself. Structuring and layering schemes that unfold across individual transactions within a single snapshot window are invisible to a model that only observes 12 aggregated states.

EvolveGCN-H is the weakest GNN across all three tiers (AUC-ROC 0.9064, AUC-PR 0.0504). It is also by far the largest model in the study at 2,213,673 parameters, roughly thirty-five times the size of the GCN, yet it delivers the worst GNN performance. This combination of parameter explosion and weak accuracy is the signature of the architecture's design: evolving the GCN weight matrices themselves, rather than node states, produces a large and unstable parameter space whose optimisation does not converge to a competitive solution on this task. The result is a clear negative finding: weight-space evolution is not merely expensive here but architecturally ill-suited to transaction-level AML detection.

**4.5.2 Continuous-Time TGN**

Table 4.3 presents the test set results for the continuous-time TGN, which processes each transaction at its exact timestamp and maintains a per-node memory updated by an exponential moving average (Section 3.4).

**Table 4.3: TGN results on the test set (chronological split), 85,905 parameters. The first two rows use the standard cold-start memory, in which per-node memory begins empty at the start of the test period and updates as transactions are processed; they share the same AUC values and differ only in decision threshold and therefore operating point. The third row does not start from an empty memory: it first replays the training and validation transactions to populate each node's memory, then carries and updates that memory continuously through the test period (Section 4.5.3).**

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

The test set (the latest 15% of transactions, approximately 761,000 edges) was divided into 12 equal slices by edge count after chronological sorting, matching the 12-window configuration of the snapshot models. Performance was measured under two memory regimes. Under memory reset per slice, per-node memory is cleared at the start of every slice, so no interaction history carries across slice boundaries. Under memory carried continuously, per-node memory is warm-started from the training and validation periods and maintained across the entire test set, so that by later slices it encodes all preceding history. Both regimes are leakage-free: every prediction uses the memory state from before the current transaction. Table 4.4 reports AUC-PR for both regimes alongside the laundering prevalence of each slice.

**Table 4.4: TGN per-slice AUC-PR under memory reset per slice versus memory carried continuously, with laundering prevalence (selected slices).**

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

Third, the two regimes converge at the high-prevalence late slices (0.506 versus 0.497 at slice 11), and in aggregate the continuously-carried run scores slightly below the standard cold-start evaluation on the full test set (AUC-PR 0.271 versus 0.321, Table 4.3). The contribution of per-node memory is thus concentrated where it is operationally most valuable, in the early, low-prevalence portion of the future period, while the headline test figure is best read as a cold-start estimate over the test window.

**4.6 Cross-Model Comparison**

Table 4.5 presents all evaluated models in a single leaderboard ordered by AUC-PR. Because every model now uses the identical chronological split, the leaderboard is directly comparable across tiers, with no protocol asterisks.

**Table 4.5: Complete model leaderboard on the test set (chronological split), ordered by AUC-PR. F1 is at each model's calibrated threshold.**

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

The conclusions of this study rest on two models: GCN as the strongest static architecture and TGN as the strongest overall. To establish that their results are not artefacts of a single random seed or a single hyperparameter choice, both were retrained across three seeds (42, 123, 7) and subjected to a one-at-a-time sensitivity sweep on their most consequential hyperparameter. Seed repetition and sensitivity were concentrated on these two models by design; the remaining seven models are supporting comparisons on which no conclusion rests, and are reported as single documented runs (Section 3.6). Tables 4.6 and 4.7 report the results.

**Table 4.6: Seed stability across three seeds (mean +/- standard deviation).**

| Model | AUC-ROC | AUC-PR | F1 |
|-------|---------|--------|-----|
| GCN | 0.9715 +/- 0.0008 | 0.1776 +/- 0.0203 | 0.2023 +/- 0.0118 |
| TGN | 0.9686 +/- 0.0011 | 0.3396 +/- 0.0131 | 0.3450 +/- 0.0382 |

Two findings matter. AUC-ROC is essentially deterministic for both models (standard deviation around 0.001), so the ranking-quality claims are seed-independent. AUC-PR is more variable, as expected for a metric computed on roughly 1,500 positives among 760,000 test edges, but the separation between the two models survives comfortably: TGN's AUC-PR (0.340 +/- 0.013) and GCN's (0.178 +/- 0.020) do not overlap within a standard deviation, and TGN's is in fact the tighter of the two. The headline advantage of continuous-time temporal modelling is therefore a stable property, not a favourable draw. It is worth recording that the single GCN run in the leaderboard (Table 4.5, AUC-PR 0.206) sits at the upper end of the GCN seed distribution; the seed mean of 0.178 is the more representative figure, and the TGN advantage is correspondingly larger against it.

**Table 4.7: One-at-a-time hyperparameter sensitivity (seed 42).**

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

**Post-hoc feature attribution.** Although the models are not natively interpretable, model-agnostic attribution can recover which features drive a flag, the low-cost first step recommended in Section 5.3.5. Applying permutation feature importance to the GCN, in which each feature is randomly shuffled and the resulting drop in test AUC-PR is measured, produces an interpretable ranking (Appendix F, Figure F.2). The single most important feature is the ACH payment channel: permuting it alone removes most of the model's discriminative power (an AUC-PR drop of 0.175 from a baseline of 0.202), consistent with the exploratory finding that ACH carries the highest laundering rate (Section 4.1). Transaction amount and time-of-week features follow, and among the account-level features the structural ones, out-degree, in-degree, and counterparty counts, are the most important. This last point closes a loop with the ablation (Section 4.2): those relational features are nearly useless to a flat classifier (node-only AUC-PR 0.019), yet the GCN relies on them through message passing, confirming that the value of the graph lies in how these features are propagated rather than in the features as static inputs. A ranking of this kind gives a compliance analyst a documentable rationale for the features underlying a flag, illustrating in practice the interpretability recommendation developed in Section 5.3.5.

**Scalability and generalisability.** All results are on the HI-Small variant (518,581 accounts, 5,078,345 transactions). The IBM AML suite also provides larger variants generated by the same process (Altman et al., 2023), so the architectural findings are expected to transfer, but an empirical scaling study across variants (train and inference time, memory, and throughput as a function of graph size) was not run for this submission and is identified as future work (Section 5.5). Within-dataset generalisation to unseen time periods is demonstrated directly: every model is tested on transactions strictly later than those it was trained on, and TGN's per-slice analysis (Section 4.5.3) shows its per-node memory contributes transferable signal in the early, low-prevalence portion of that future period.

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

Among **static architectures**, GCN outperformed both GAT and GraphSAGE. The margin is clear: GCN's AUC-PR of 0.2056 is more than double GAT's 0.0912 and roughly five times GraphSAGE's 0.0412. GCN's symmetric normalised aggregation appears well-suited to the financial transaction graph, where node degrees are highly variable and degree-based normalisation prevents high-degree nodes from dominating the aggregation. GAT's single-head configuration, required because multi-head attention over the full graph is prohibitively memory-intensive at this scale, likely understates the potential of attention-based architectures. GraphSAGE's mean aggregation with neighbourhood sampling, while scalable, appears to dilute the discriminative signal from structurally distinctive laundering accounts.

Among **snapshot temporal architectures**, TemporalGCN modestly outperformed EvolveGCN-H (AUC-PR 0.0604 vs 0.0504), though both are weak. The gap, such as it is, is attributable to two factors. First, state-space evolution (TemporalGCN's GRU on per-node hidden states) provides a more stable temporal learning signal than weight-space evolution (EvolveGCN-H's GRU on GCN weight matrices). Per-node states carry account-specific behavioural history; weight matrices carry only aggregate graph dynamics. Second, EvolveGCN-H suffers from an inherent parameter explosion: the GRU hidden state scales as rank times the sum of input and output dimensions, so the parameter count grows steeply with rank and exceeds available memory at higher ranks. Even at rank 2 the model has 2,213,673 parameters, more than any other model in this study, yet its capacity for this task remains limited.

The **continuous-time TGN** decisively outperformed both snapshot-based models. TGN's AUC-PR of 0.3213 is more than five times TemporalGCN's 0.0604 and more than six times EvolveGCN-H's 0.0504, under identical chronological evaluation. The architectural distinction is granularity: TGN processes each of ~5 million transactions individually with its exact timestamp, while snapshot models aggregate transactions into 12 coarse windows. Laundering patterns such as structuring, which involve sequences of transactions within minutes or hours, are invisible at the snapshot level but detectable at the transaction level. TGN's per-node EMA memory provides a learned behavioural summary that accumulates over the entire transaction history, enabling the model to recognise accounts whose behaviour transitions from legitimate to suspicious.

A negative finding carries its own significance: **both snapshot-based temporal GNNs underperform the static GCN despite using temporal information.** TemporalGCN (AUC-ROC 0.9514) and EvolveGCN-H (0.9064) are both below static GCN (0.9708). Temporal modelling is not automatically beneficial; at coarse granularity, it can be worse than no temporal modelling at all. Because every model is evaluated under the same chronological protocol, this comparison is direct: the snapshot models' shortfall reflects architecture rather than an easier or harder evaluation. It confirms that continuous-time processing, not temporal modelling in the abstract, is the decisive advantage.

**5.1.3 SQ3: GNNs vs Conventional Machine Learning**

SQ3 asked: *How does the performance of static and temporal GNN-based models compare to Logistic Regression, Random Forest, and XGBoost in detecting money laundering?*

The comparison reveals a clear hierarchy. XGBoost (AUC-PR 0.1460) sets the non-graph performance ceiling. Static GCN (AUC-PR 0.2056) adds 41% over XGBoost from graph structure alone, confirming that relational information provides detection value beyond what flat features can capture. TGN (AUC-PR 0.3213) adds 120% over XGBoost from the combination of graph structure and fine-grained temporal modelling.

Logistic Regression (AUC-PR 0.0378) struggles badly with the extreme class imbalance in the 28-dimensional edge feature space, while Random Forest (AUC-PR 0.1249) approaches XGBoost, showing that non-linear tree ensembles recover much of the signal available in the flat features. XGBoost's regularised boosting provides the best non-graph separation, but the absence of relational context limits its ceiling.

A nuanced interpretation is required. The GNN advantage over conventional classifiers is substantial but not overwhelming when only graph structure is added (GCN +41% AUC-PR over XGBoost). The decisive advantage emerges when continuous-time temporal modelling is combined with graph structure (TGN +120% AUC-PR over XGBoost). For an AML compliance team deciding whether to invest in graph-based detection infrastructure, the evidence suggests that graph structure alone provides a measurable but modest improvement; the full benefit requires the additional investment in temporal infrastructure.

**5.1.4 SQ4: Practical Implications for AML Practitioners**

SQ4 asked: *What practical implications do the comparative empirical findings hold for AML compliance practitioners?*

This sub-question is addressed in detail in the dedicated practitioner implications section (Section 5.3). In summary, the findings provide evidence-based guidance on model selection across three tiers, quantify the precision-recall trade-offs that operational compliance teams face, and identify the conditions under which investment in temporal GNN infrastructure is justified by a meaningful improvement in detection performance. Crucially, the findings also delimit what the evidence can support: detection performance is only a precondition for deployment, and the analysis identifies explainability, in line with FATF (2021), as a further precondition that none of the evaluated models meets natively (Section 5.3.5). The practitioner guidance is therefore framed as evidence on model selection subject to those preconditions, rather than as an unqualified endorsement for deployment.

**5.1.5 Main Research Question**

The main research question asked: *How do static and temporal Graph Neural Network architectures compare to conventional supervised machine learning classifiers in detecting money laundering in financial transaction networks?*

The answer, grounded in the empirical evidence presented in Chapter 4, is as follows.

Continuous-time temporal GNNs with per-node memory (TGN) decisively outperform both static GNNs and conventional machine learning classifiers for AML detection under deployment-realistic chronological evaluation. The performance hierarchy is: **TGN > GCN > XGBoost > Random Forest > GAT > TemporalGCN > EvolveGCN-H > GraphSAGE > Logistic Regression**, measured by AUC-PR, the metric most sensitive to minority class detection quality.

Three qualifications are essential. First, **temporal modelling is not inherently beneficial.** Snapshot-based temporal GNNs (TemporalGCN, EvolveGCN-H) underperform the static GCN, demonstrating that temporal information must be modelled at transaction-level granularity to add value. Coarse temporal bucketing discards the very patterns it is meant to capture. Second, **graph structure alone provides a measurable but modest gain.** GCN improves AUC-PR by 41% over XGBoost. The combination of graph structure and continuous-time temporal modelling (TGN) improves AUC-PR by 120% over XGBoost. The whole is greater than the sum of its parts. Third, **evaluation protocol determines how honestly these numbers reflect real-world performance.** This study evaluates every model on future transactions after training on past ones, a more deployment-realistic protocol than the random splits that predominate in published AML GNN studies, which mix past and future transactions and tend to overstate performance.

**5.2 Theoretical Implications**

This study makes four contributions to the theoretical understanding of GNN-based AML detection.

**Temporal granularity as a first-order design factor.** The finding that snapshot-based temporal GNNs underperform static GCN while continuous-time TGN substantially outperforms it establishes that temporal granularity, not temporal modelling in the abstract, determines detection performance. This extends the theoretical framework of Section 2.4, which presented snapshot and continuous-time paradigms as alternatives without evidence favouring one over the other. The empirical results provide such evidence: for financial transaction networks where laundering patterns unfold at the level of individual transactions, the snapshot paradigm is architecturally insufficient. This is not an implementation limitation but a theoretical one: no number of snapshots can fully recover transaction-level ordering if multiple transactions are aggregated within each window.

**Per-node memory as a learned behavioural summary.** The per-slice analysis (Section 4.5.3) provides empirical evidence that EMA memory functions as a learned behavioural summary: at the low-prevalence early test slices, carrying memory continuously across the test period yields markedly higher precision-recall than resetting it, showing that accumulated interaction history contributes detection signal precisely where within-slice positives are scarce. This connects to the criminological theory discussed in Section 2.1: Levi (2002) identified that laundering is detectable through relational context, and the FATF (2023) defines layering as an inherently sequential process. Per-node memory operationalises these theoretical insights by maintaining a differentiable summary of each account's transaction history, updated with each new interaction.

**Weighted loss and gradient clipping interaction.** The methodological finding that gradient clipping destructively interacts with large pos_weight values under extreme class imbalance (Section 3.4.4) has implications beyond this implementation. Standard neural network training guidance recommends gradient clipping as a stability measure. This recommendation must be qualified when large class weights are applied: clipping thresholds should be set relative to the post-weighted gradient magnitudes, not the unweighted ones. This finding contributes to the literature on training neural networks under class imbalance (Dou et al., 2020; He & Garcia, 2009).

**Chronological evaluation as a methodological standard.** Because laundering prevalence and transaction patterns drift over the observation window, a random split that mixes past and future transactions leaks future information into training and yields optimistic estimates. Evaluating on strictly future transactions, as done here, more honestly reflects deployment conditions. The field would benefit from standardising on chronological evaluation; studies that report only random-split results may overstate real-world performance.

**5.3 Practitioner Implications**

This section translates the empirical findings into actionable guidance for AML compliance practitioners, directly addressing the assessment criterion that the thesis provide concrete, evidence-based recommendations for the compliance practice community.

**5.3.1 Model Selection Decision Framework**

The three-tier evaluation supports a decision framework for AML compliance teams selecting a detection approach. The appropriate tier depends on three factors: the institution's existing data infrastructure, the acceptable false positive burden, and the regulatory stakes of missed detection.

**Tier 1: Conventional ML (XGBoost).** Appropriate when an institution needs a quickly deployable system with low infrastructure requirements. XGBoost operates on flat transaction features, requires no graph database or temporal infrastructure, trains in minutes, and produces interpretable feature importance scores. Its AUC-PR of 0.146 means that, at a calibrated threshold, it detects a meaningful fraction of laundering cases. The limitation is precision: XGBoost's low precision at any reasonable recall level means compliance analysts will review many false positives. This tier is suitable for institutions in early stages of AML analytics maturity, or as a baseline against which more sophisticated approaches are benchmarked.

**Tier 2: Static GNN (GCN).** Appropriate when an institution has invested in graph infrastructure and seeks improved precision over conventional approaches. GCN's AUC-PR of 0.206 represents a 41% improvement over XGBoost. At its calibrated threshold (0.67), GCN detects 52% of laundering at 12% precision, so roughly one in eight alerts is genuine. The infrastructure requirements are moderate: a graph database mapping accounts to nodes and transactions to edges, with batch retraining as new transaction data arrives. GCN does not model temporal dynamics, so it is most appropriate when the primary laundering patterns of concern are relational (layering chains, fan-in/fan-out structures) rather than temporal (behavioural transitions, transaction sequencing).

**Tier 3: Continuous-Time Temporal GNN (TGN).** Appropriate when detection quality is a regulatory or operational priority justifying additional infrastructure investment. TGN's AUC-PR of 0.321 represents a 120% improvement over XGBoost and a 56% improvement over GCN. At its calibrated threshold (0.16), TGN detects 42% of laundering at 22% precision, so roughly one in five alerts is genuine; raising the threshold trades recall for precision, and at the default threshold TGN reaches 84% precision at 17% recall, an operating point suited to teams prioritising alert quality over coverage. The infrastructure requirements are more substantial: transactions must be processed in chronological order with individual timestamps, per-node memory states must persist across inference batches, and model retraining must respect temporal ordering to avoid data leakage. The investment is justified when the cost of missed laundering (regulatory fines, reputational damage, criminal facilitation) outweighs the cost of temporal infrastructure.

Table 5.1 summarises the decision framework.

**Table 5.1: Model selection framework for AML compliance practitioners.**

| Factor | Tier 1: XGBoost | Tier 2: GCN | Tier 3: TGN |
|--------|-----------------|-------------|-------------|
| AUC-PR | 0.146 | 0.206 | 0.321 |
| Precision at calibrated threshold | ~0.10 | ~0.12 | ~0.22 |
| Infrastructure requirements | Low | Moderate | Substantial |
| Interpretability | High (feature importance) | Moderate (node embeddings) | Moderate (memory states) |
| Temporal dynamics | Not modelled | Not modelled | Modelled (continuous-time) |
| Deployment complexity | Low | Moderate | High |

**5.3.2 Precision-Recall Trade-offs and Operational Alert Burden**

The precision-recall trade-off has direct operational consequences for compliance team workload. At its calibrated (F1-optimal) threshold, TGN reaches about 22% precision at 42% recall. At a laundering prevalence of 0.1%, for every 100,000 transactions processed approximately 100 are genuine laundering cases; TGN would then flag roughly 190 transactions, of which about 42 would be genuine and 148 false positives, catching a little under half of all laundering. Raising the threshold sharply reduces this burden: at the default threshold TGN reaches about 84% precision at 17% recall, flagging only around 20 transactions per 100,000 (about 17 genuine, 3 false) while detecting fewer cases. The threshold is the lever a compliance team sets against its own review capacity and risk tolerance.

In contrast, XGBoost at its default threshold flags roughly 3,550 transactions per 100,000 (its 0.0245 precision and 0.8706 recall flag about 87 of every 100 laundering cases, but at only 2.5% precision), producing approximately 3,460 false positives for every 87 true positives. An analyst reviewing 100 alerts would see roughly two to three genuine laundering cases from XGBoost, against about 22 from TGN at its calibrated threshold.

The threshold is configurable. An institution prioritising recall (catching as many laundering cases as possible, accepting more false positives) can lower the threshold. An institution prioritising precision (minimising analyst time wasted on false positives) can raise it. The calibrated thresholds reported in Chapter 4 maximise F1-score but are not prescriptive; each institution should calibrate against its own cost ratio of false negatives to false positives.

**5.3.3 Deployment Considerations**

Several practical considerations arise from the development and evaluation of these models.

**Chronological retraining.** TGN's per-node memory states are a function of transaction history. When the model is retrained on new data, memory states must be reinitialised from the beginning of the training period or carried forward from the previous training run. The former is simpler but discards accumulated history; the latter preserves history but requires careful handling to avoid stale memory states from outdated model weights. A practical approach is periodic full retraining (monthly or quarterly) with memory states computed from scratch over the full historical dataset, combined with daily inference using frozen model weights and continuously updating memory.

**Feature engineering in production.** The 28 edge features and 12 node features used in this study (detailed in Appendix A) were computed from raw transaction and account data. In a production setting, these features must be computed in real time or near-real time as transactions arrive. The log-transformed amount features and one-hot encoded categorical features are straightforward to compute; the cyclic time encodings (hour of day, day of week) require timestamp parsing. The node features (degree, volume, counterparty statistics) are aggregate statistics that must be recomputed or incrementally updated as new transactions arrive.

**Memory persistence.** For deployment, TGN's per-node memory states must persist between inference batches via a key-value store mapping account composite keys to memory vectors. At 64 dimensions (256 bytes per account at float32), total storage for 500,000 accounts is approximately 128 MB, negligible by modern infrastructure standards.

**Reproducibility and adaptation.** The complete source code, trained model checkpoints, and reproduction commands are available in the project repository (Appendix B). Compliance analytics teams can reproduce the reported results, evaluate the models against their own institutional data, and adapt the implementation to their specific requirements. The tool's modular architecture separates data loading, feature engineering, graph construction, model definition, and training, allowing individual components to be replaced or extended without modifying the rest of the pipeline.

**5.3.4 Cost-Benefit Considerations**

The decision to invest in temporal GNN infrastructure depends on the institution's risk exposure and current detection baseline. TGN's 56% AUC-PR improvement over GCN represents a meaningful detection gain, but it requires investment in chronological data pipelines, memory state management, and more complex model operations. For an institution currently operating rule-based systems with very low detection rates, the incremental benefit of GCN over rule-based approaches may be large enough to justify graph infrastructure, with TGN representing a second-phase investment. For an institution already using conventional ML, the direct jump to TGN may be justified if the substantial precision improvement over flat-feature models translates to analyst time savings and improved detection of sophisticated laundering schemes.

This study does not provide a financial cost-benefit analysis, as the costs of false negatives (regulatory fines, criminal facilitation) and false positives (analyst time, delayed legitimate transactions) are institution-specific. However, the quantified performance differences reported in Chapter 4 provide the empirical inputs that an institution would need to conduct such an analysis.

**5.3.5 Explainability as a Deployment Precondition**

Detection performance is a necessary but not sufficient condition for deployment. FATF (2021) guidance on new technologies for AML/CFT states that their effective use requires tools whose outputs can be understood by non-experts and communicated to competent authorities when required. None of the architectures evaluated here meets this condition natively: a compliance analyst who receives a TGN alert cannot, from the model alone, explain why the transaction was flagged, which is precisely the justification a suspicious-activity report demands. Explainability is therefore a precondition for regulatory deployment, not an optional refinement.

The recommendation is to pair the detector with a post-hoc explanation layer matched to the architecture and the operational need. Model-agnostic feature-attribution methods, such as permutation feature importance, SHAP (Lundberg & Lee, 2017), and integrated gradients (Sundararajan et al., 2017), apply to any of the models and are the lowest-cost first step: they attribute a flag to the transaction and account features that drove it, giving an analyst a documentable rationale. This study provides a worked example: applying permutation feature importance to the GCN (Section 4.8, Figure F.2) identifies the transaction and account features that drive its flags, showing that even without native interpretability a model-agnostic method can furnish an analyst with a concrete rationale. Where a relational explanation is required, GNNExplainer (Ying et al., 2019) can identify the subgraph of counterparties most responsible for a flag, and a GAT's attention weights offer a limited native signal. The continuous-time TGN is the hardest case: because its predictions depend on accumulated per-node memory, a faithful explanation must account for an account's interaction history rather than the current transaction alone, and existing explainers do not yet model this. A practical path is therefore to run detection with the strongest model while generating analyst-facing explanations with a model-agnostic attribution method on the flagged transaction's features, treating memory-aware relational explanation as a development priority (Section 5.5). Institutions adopting these tools should budget for this explanation layer from the outset rather than treating interpretability as a post-deployment addition.

**5.4 Limitations**

This study has several limitations that should be considered when interpreting its findings and assessing their generalizability.

**No explanation of flagged transactions.** None of the evaluated models produces a human-legible justification for an individual flag, and the best-performing model, TGN, is the hardest to explain because its predictions depend on accumulated per-node memory (Section 2.3.3). This is an operational and regulatory limitation, not merely a technical one: FATF (2021) guidance expects AML tools to be explainable to non-experts and to competent authorities. The study evaluates detection performance but does not implement or evaluate an explanation mechanism; interpretability is addressed as a deployment recommendation (Section 5.3.5) and a future-research direction (Section 5.5) rather than delivered here.

**Synthetic dataset.** The IBM AML HI-Small dataset is synthetic, with laundering patterns derived from FATF-documented typologies (Altman et al., 2023). While this ensures the patterns reflect regulatory knowledge, it also means the model has been evaluated on simulated rather than genuine criminal behaviour. The extent to which performance on this benchmark transfers to real-world money laundering detection depends on how closely the FATF-informed simulation approximates actual laundering patterns in institutional transaction data. Publicly available real-world AML transaction datasets with account-level granularity do not currently exist, making synthetic benchmarks the only reproducible evaluation option available to independent researchers.

**Single dataset variant.** Only the HI-Small variant (518,581 accounts, 5,078,345 transactions) was used. The IBM AML dataset offers four variants (HI/LI combined with Small/Medium). While the data-generating process is identical across variants, meaning architectural findings are expected to generalise, empirical verification on the larger variants and the lower-prevalence LI variants was not performed. The Medium variant, with tens of millions of transactions, would test the scalability claims made here.

**Memory-constrained training.** Full-graph training restricted some architectural choices. GAT was limited to a single attention head, because multi-head attention over the full graph is prohibitively memory-intensive at this scale, likely understating the potential of attention-based architectures. EvolveGCN-H was limited to rank 2, likely understating what the architecture could achieve at higher ranks. No automated hyperparameter optimisation was performed. Training with more memory would allow these restrictions to be lifted and might yield different performance rankings, particularly for GAT and EvolveGCN-H.

**Snapshot granularity not systematically investigated.** The 12-window snapshot configuration was chosen as a reasonable balance between temporal resolution and per-snapshot edge density. The sensitivity of snapshot model performance to the number of snapshots was not systematically varied. It is possible that a larger number of snapshots (for example, 100 or 1,000) could partially close the gap between snapshot and continuous-time models, though the computational cost would scale linearly with the number of snapshots.

**EMA memory versus GRU memory.** TGN uses a custom EMA-based memory rather than the learned GRU-based memory of Rossi et al. (2020), for the design reasons set out in Section 3.4. The EMA update maintains a fixed beta parameter (0.85) that controls the rate at which historical information decays, whereas GRU-based memory would learn this rate through gating. The extent to which this simplification affects the reported results is not separately quantified.

**No ensemble methods.** Individual models were evaluated independently. Ensembles combining complementary architectures (for example, GCN for relational patterns plus TGN for temporal patterns) were not explored and might outperform any single model.

**Single financial system simulation.** The IBM AML dataset simulates transactions within a single financial system. Cross-institutional laundering, where funds move between accounts at different banks, is not represented. Real-world AML detection often involves multiple institutions with incomplete visibility into each other's transaction networks.

**No fairness or bias analysis.** The dataset's laundering patterns are derived from FATF typologies rather than real enforcement data, which mitigates but does not eliminate the risk that the model learns patterns correlated with legitimate but atypical financial behaviour. No analysis of model fairness across entity types, banks, or transaction patterns was conducted.

**5.5 Future Research**

The findings and limitations of this study suggest several directions for future research.

**Explainable GNN-based AML detection.** The most consequential extension is to make the detectors interpretable for compliance use. This includes evaluating model-agnostic attribution methods (permutation feature importance, SHAP, integrated gradients) and graph-specific explainers such as GNNExplainer on flagged transactions, and, most challengingly, developing memory-aware explanation methods for continuous-time models such as TGN, whose predictions depend on accumulated interaction history that current explainers do not represent. The resulting explanations would themselves need to be validated with compliance practitioners for whether they support the documentation and communication that a suspicious-activity report, and FATF (2021) expectations, require.

**Cross-variant and cross-domain evaluation.** Extending the evaluation to all four IBM AML variants (HI/LI combined with Small/Medium) would test the generalizability of the architectural findings across dataset scales and prevalence ratios. Beyond the IBM AML benchmark, evaluating the same architectures on other public financial transaction benchmarks, should they become available, would test whether the performance hierarchy reported here is dataset-specific or architecture-inherent.

**GPU-scale training with hyperparameter optimisation.** Training these architectures on GPU hardware with systematic hyperparameter search (grid, random, or Bayesian) would test whether the performance rankings reported here are robust to hyperparameter choices and whether architectures constrained by CPU (GAT multi-head, EvolveGCN-H higher rank) achieve better performance with those constraints lifted.

**Alternative continuous-time architectures.** Beyond TGN, other continuous-time temporal GNN architectures exist, including temporal attention networks (Xu et al., 2020) and DyRep (Trivedi et al., 2019). Evaluating these on the IBM AML benchmark would provide a more complete picture of the continuous-time paradigm's capabilities for AML detection.

**Ensemble approaches.** Combining complementary architectures, such as a static GCN for structural pattern detection with a TGN for temporal pattern detection, may yield performance exceeding any single model. The different information sources (relational vs temporal) suggest that ensemble predictions could be more robust than individual model predictions.

**Fairness and bias in GNN-based AML.** Research is needed on whether GNN-based AML models exhibit bias against particular entity types, geographies, or transaction patterns. If GNN message passing amplifies biases present in the underlying transaction data, the fairness implications for automated AML screening could be significant.

**Multi-institutional transaction networks.** Extending the graph to include transactions across multiple financial institutions, potentially through federated learning or privacy-preserving techniques, would address the limitation that real-world laundering often spans multiple banks.

**5.6 Concluding Remarks**

This study set out to answer how static and temporal graph neural network architectures compare to conventional machine learning for detecting money laundering in financial transaction networks. The answer, supported by a systematic three-tier evaluation on a standardised public benchmark, is that continuous-time temporal GNNs with per-node memory achieve the strongest detection performance, but temporal modelling is not automatically beneficial: snapshot-based approaches underperformed the static GCN, demonstrating that temporal information must be at the right granularity to add value. Money laundering is both relational and temporal; the detection tools built to counter it must address both dimensions.

# References

Alarab, I., & Prakoonwit, S. (2023). Graph-based LSTM for anti-money laundering: Experimenting with temporal graph convolutional network for bitcoin data. *Neural Processing Letters*, *55*(1), 689-707. https://doi.org/10.1007/s11063-022-10904-8

Altman, E., Blanuša, J., von Niederhäusern, L., Egressy, B., Anghel, A., & Atasu, K. (2023). Realistic synthetic financial transactions for anti-money laundering models. In *Advances in Neural Information Processing Systems 36 (NeurIPS 2023): Datasets and Benchmarks Track*. https://proceedings.neurips.cc/paper_files/paper/2023/hash/5f38404edff6f3f642d6fa5892479c42-Abstract-Datasets_and_Benchmarks.html

Barros, C. D. T., Mendonça, M. R. F., Vieira, A. B., & Ziviani, A. (2021). A survey on embedding dynamic graphs. *ACM Computing Surveys*, *55*(1), Article 10, 1-37. https://doi.org/10.1145/3483595

Breiman, L. (2001). Random forests. *Machine Learning*, *45*(1), 5-32. https://doi.org/10.1023/A:1010933404324

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 785-794). ACM. https://doi.org/10.1145/2939672.2939785

Chen, Z., Van Khoa, L. D., Teoh, E. N., Nazir, A., Karuppiah, E. K., & Lam, K. S. (2018). Machine learning techniques for anti-money laundering (AML) solutions in suspicious transaction detection: A review. *Knowledge and Information Systems*, *57*(2), 245-285. https://doi.org/10.1007/s10115-017-1144-z

Cheng, D., Ye, Y., Xiang, S., Ma, Z., Zhang, Y., & Jiang, C. (2023). Anti-money laundering by group-aware deep graph learning. *IEEE Transactions on Knowledge and Data Engineering*. https://doi.org/10.1109/TKDE.2023.3272396

Cheng, D., Zou, Y., Xiang, S., & Jiang, C. (2024). Graph neural networks for financial fraud detection: A review. *Frontiers of Computer Science*, *19*(5), Article 19505. https://doi.org/10.1007/s11704-024-40474-y

Cho, K., Van Merrienboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning phrase representations using RNN encoder-decoder for statistical machine translation. In *Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP)* (pp. 1724-1734). ACL. https://doi.org/10.3115/v1/D14-1179

Deng, Z., Xin, G., Liu, Y., Wang, W., & Wang, B. (2022). Contrastive graph neural network-based camouflaged fraud detector. *Information Sciences*, *618*, 39-52. https://doi.org/10.1016/j.ins.2022.10.072

Dou, Y., Liu, Z., Sun, L., Deng, Y., Peng, H., & Yu, P. S. (2020). Enhancing graph neural network-based fraud detectors against camouflaged fraudsters. In *Proceedings of the 29th ACM International Conference on Information and Knowledge Management (CIKM)* (pp. 315-324). ACM. https://doi.org/10.1145/3340531.3411903

FATF. (2021). *Opportunities and challenges of new technologies for AML/CFT*. Financial Action Task Force. https://www.fatf-gafi.org/en/publications/Digitaltransformation/Opportunities-challenges-new-technologies-for-aml-cft.html

FATF. (2023). *International standards on combating money laundering and the financing of terrorism and proliferation: The FATF recommendations*. Financial Action Task Force. https://www.fatf-gafi.org/en/publications/Fatfrecommendations/Fatf-recommendations.html

Fey, M., & Lenssen, J. E. (2019). Fast graph representation learning with PyTorch Geometric. In *ICLR 2019 Workshop on Representation Learning on Graphs and Manifolds*. https://arxiv.org/abs/1903.02428

Grover, A., & Leskovec, J. (2016). node2vec: Scalable feature learning for networks. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 855-864). ACM. https://doi.org/10.1145/2939672.2939754

Guo, H., Li, Y., Shang, J., Gu, M., Huang, Y., & Gong, B. (2017). Learning from class-imbalanced data: Review of methods and applications. *Expert Systems with Applications*, *73*, 220-239. https://doi.org/10.1016/j.eswa.2016.12.035

Hamilton, W. L., Ying, R., & Leskovec, J. (2017). Inductive representation learning on large graphs. In *Advances in Neural Information Processing Systems 30 (NeurIPS 2017)* (pp. 1024-1034). https://papers.nips.cc/paper/2017/hash/5dd9db5e033da9c6fb5ba83c7a7ebea9-Abstract.html

He, H., & Garcia, E. A. (2009). Learning from imbalanced data. *IEEE Transactions on Knowledge and Data Engineering*, *21*(9), 1263-1284. https://doi.org/10.1109/TKDE.2008.239

Hevner, A. R., March, S. T., Park, J., & Ram, S. (2004). Design science in information systems research. *MIS Quarterly*, *28*(1), 75-105. https://doi.org/10.2307/25148625

Jain, S., & Wallace, B. C. (2019). Attention is not explanation. In *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies* (pp. 3543-3556). https://doi.org/10.18653/v1/N19-1357

Jensen, R. I., & Iosifidis, A. (2023). Qualifying and raising anti-money laundering alarms with deep learning. *Expert Systems with Applications*, *214*, 119037. https://doi.org/10.1016/j.eswa.2022.119037

Johannessen, F., & Jullum, M. (2025). Finding money launderers using heterogeneous graph neural networks. *Journal of Finance and Data Science*, *11*, Article 100175. https://doi.org/10.1016/j.jfds.2025.100175

Jullum, M., Løland, A., Huseby, R. B., Ånonsen, G., & Lorentzen, J. (2020). Detecting money laundering transactions with machine learning. *Journal of Money Laundering Control*, *23*(1), 173-186. https://doi.org/10.1108/JMLC-07-2019-0055

Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. In *International Conference on Learning Representations (ICLR 2017)*. https://arxiv.org/abs/1609.02907

Kute, D. V., Pradhan, B., Shukla, N., & Alamri, A. (2021). Deep learning and explainable artificial intelligence techniques applied for detecting money laundering – A critical review. *IEEE Access*, *9*, 82300-82317. https://doi.org/10.1109/ACCESS.2021.3086230

Levi, M. (2002). Money laundering and its regulation. *The Annals of the American Academy of Political and Social Science*, *582*(1), 181-194. https://doi.org/10.1177/000271620258200113

Li, E., Chen, M., Xiang, S., & Chen, L. (2025). Graph learning-empowered financial fraud detection: Progress and future directions. *Intelligent Computing*, *4*, Article 0146. https://doi.org/10.34133/icomputing.0146

Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollar, P. (2017). Focal loss for dense object detection. In *Proceedings of the IEEE International Conference on Computer Vision (ICCV)* (pp. 2980-2988). IEEE. https://doi.org/10.1109/ICCV.2017.324

Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. In *Advances in Neural Information Processing Systems 30 (NeurIPS 2017)* (pp. 4765-4774). https://papers.nips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html

Ma, X., Wu, J., Xue, S., Yang, J., Zhou, C., Sheng, Q. Z., Xiong, H., & Akoglu, L. (2023). A comprehensive survey on graph anomaly detection with deep learning. *IEEE Transactions on Knowledge and Data Engineering*, *35*(12), 12012-12038. https://doi.org/10.1109/TKDE.2021.3118815

Motie, S., & Raahemi, B. (2024). Financial fraud detection using graph neural networks: A systematic review. *Expert Systems with Applications*, *240*, Article 122156. https://doi.org/10.1016/j.eswa.2023.122156

Oztas, B., Cetinkaya, D., Adedoyin, F., & Budka, M. (2023). SAML-D: A synthetic anti-money laundering dataset with controlled complexity. *Data in Brief*, *51*, 109692. https://doi.org/10.1016/j.dib.2023.109692

Pareja, A., Domeniconi, G., Chen, J., Ma, T., Suzumura, T., Kanezashi, H., Kaler, T., Schardl, T. B., & Leiserson, C. E. (2020). EvolveGCN: Evolving graph convolutional networks for dynamic graphs. In *Proceedings of the 34th AAAI Conference on Artificial Intelligence (AAAI 2020)* (pp. 5363-5370). AAAI Press. https://doi.org/10.1609/aaai.v34i04.5984

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, *12*, 2825-2830. https://www.jmlr.org/papers/volume12/pedregosa11a/pedregosa11a.pdf

Peffers, K., Tuunanen, T., Rothenberger, M. A., & Chatterjee, S. (2007). A design science research methodology for information systems research. *Journal of Management Information Systems*, *24*(3), 45-77. https://doi.org/10.2753/MIS0742-1222240302

Ren, L., Hu, R., Li, D., Liu, Y., Wu, J., Zang, Y., & Hu, W. (2023). Dynamic graph neural network-based fraud detectors against collaborative fraudsters. *Knowledge-Based Systems*, *278*, Article 110888. https://doi.org/10.1016/j.knosys.2023.110888

Rossi, E., Chamberlain, B., Frasca, F., Eynard, D., Monti, F., & Bronstein, M. (2020). Temporal graph networks for deep learning on dynamic graphs. *arXiv preprint arXiv:2006.10637*. https://arxiv.org/abs/2006.10637

Sekaran, U., & Bougie, R. (2019). *Research methods for business: A skill-building approach* (8th ed.). Wiley.

Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic attribution for deep networks. In *Proceedings of the 34th International Conference on Machine Learning (ICML)* (pp. 3319-3328). PMLR.

Tong, G., & Shen, J. (2023). Financial transaction fraud detector based on imbalance learning and graph neural network. *Applied Soft Computing*, *149*, Article 110984. https://doi.org/10.1016/j.asoc.2023.110984

Trivedi, R., Farajtabar, M., Biswal, P., & Zha, H. (2019). DyRep: Learning representations over dynamic graphs. In *International Conference on Learning Representations (ICLR 2019)*. https://openreview.net/forum?id=HyePrhR5KX

United Nations. (1988). *United Nations Convention against Illicit Traffic in Narcotic Drugs and Psychotropic Substances*. United Nations Treaty Series, 1582, 95. https://treaties.un.org/doc/Treaties/1990/11/19901101%2006-35%20AM/Ch_VI_19p.pdf

UNODC. (2011). *Estimating illicit financial flows resulting from drug trafficking and other transnational organized crimes*. United Nations Office on Drugs and Crime. https://www.unodc.org/documents/data-and-analysis/Studies/Illicit_financial_flows_2011_web.pdf

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention is all you need. In *Advances in Neural Information Processing Systems 30 (NeurIPS 2017)* (pp. 5998-6008). https://papers.nips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html

Veličković, P., Cucurull, G., Casanova, A., Romero, A., Lio, P., & Bengio, Y. (2018). Graph attention networks. In *International Conference on Learning Representations (ICLR 2018)*. https://arxiv.org/abs/1710.10903

Weber, M., Domeniconi, G., Chen, J., Weidele, D. K. I., Bellei, C., Robinson, T., & Leiserson, C. E. (2019). Anti-money laundering in bitcoin: Experimenting with graph convolutional networks for financial forensics. In *KDD 2019 Workshop on Anomaly Detection in Finance*. https://arxiv.org/abs/1908.02591

Wirth, R., & Hipp, J. (2000). CRISP-DM: Towards a standard process model for data mining. In *Proceedings of the 4th International Conference on the Practical Application of Knowledge Discovery and Data Mining* (pp. 29-39).

Wu, J., Hu, R., Li, D., Ren, L., Hu, W., & Zang, Y. (2024). A GNN-based fraud detector with dual resistance to graph disassortativity and imbalance. *Information Sciences*, *669*, Article 120580. https://doi.org/10.1016/j.ins.2024.120580

Xu, K., Hu, W., Leskovec, J., & Jegelka, S. (2019). How powerful are graph neural networks? In *International Conference on Learning Representations (ICLR 2019)*. https://arxiv.org/abs/1810.00826

Xu, D., Ruan, C., Korpeoglu, E., Kumar, S., & Achan, K. (2020). Inductive representation learning on temporal graphs. In *International Conference on Learning Representations (ICLR 2020)*. https://arxiv.org/abs/2002.07962

Ying, R., Bourgeois, D., You, J., Zitnik, M., & Leskovec, J. (2019). GNNExplainer: Generating explanations for graph neural networks. In *Advances in Neural Information Processing Systems 32 (NeurIPS 2019)* (pp. 9240-9251). https://papers.nips.cc/paper/2019/hash/d80b7040b773199015de6d3b4293c8ff-Abstract.html

Yuan, H., Yu, H., Gui, S., & Ji, S. (2023). Explainability in graph neural networks: A taxonomic survey. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, *45*(5), 5782-5799. https://doi.org/10.1109/TPAMI.2022.3204236

---

# Appendices

**Appendix A: Complete Feature Specification**

This appendix provides the exhaustive specification of all features used by the models in this study. Section 3.3.1 provides illustrative examples; this appendix provides the complete reference.

**A.1 Node Features (12 features)**

Node features are computed per account from the accounts file and aggregated transaction statistics. All count and amount features are log1p-transformed before standardisation. Categorical features are label-encoded then standardised.

**Table A.1: Complete node feature specification.**

| Index | Feature Name | Type | Source | Computation |
|-------|-------------|------|--------|-------------|
| 0 | bank_name | Categorical | accounts.csv | Label-encoded, then standardised (z-score) |
| 1 | bank_id | Categorical | accounts.csv | Label-encoded, then standardised (z-score) |
| 2 | entity_type | Categorical | accounts.csv | Extracted from Entity Name (e.g., "Corporation #33520" becomes "Corporation"), label-encoded, standardised |
| 3 | degree_out | Numeric | transactions.csv | Number of transactions sent by this account (log1p) |
| 4 | total_amount_out | Numeric | transactions.csv | Sum of amounts sent (log1p) |
| 5 | avg_amount_out | Numeric | transactions.csv | Mean amount sent (log1p) |
| 6 | num_counterparties_out | Numeric | transactions.csv | Number of unique receiving accounts (log1p) |
| 7 | degree_in | Numeric | transactions.csv | Number of transactions received by this account (log1p) |
| 8 | total_amount_in | Numeric | transactions.csv | Sum of amounts received (log1p) |
| 9 | avg_amount_in | Numeric | transactions.csv | Mean amount received (log1p) |
| 10 | num_counterparties_in | Numeric | transactions.csv | Number of unique sending accounts (log1p) |
| 11 | degree_total | Numeric | transactions.csv | degree_out + degree_in (log1p) |

All 12 features are standardised to zero mean and unit variance using a StandardScaler fitted on the training set only. Accounts with no transaction history receive zero values for all transaction statistic features after joining.

**A.2 Edge Features (28 features)**

Edge features are computed per transaction. Amount features are log1p-transformed. Cyclic time features are encoded as sine-cosine pairs. Categorical features are one-hot encoded. The first 6 features (amount_log1p, hour_sin, hour_cos, dow_sin, dow_cos, amount_paid_log1p) are standardised; the 22 one-hot features (7 payment format + 15 currency) are left unstandardised since they are bounded to {0, 1}.

**Table A.2: Complete edge feature specification.**

| Index | Feature Name | Type | Source | Computation |
|-------|-------------|------|--------|-------------|
| 0 | amount_log1p | Numeric | transactions.csv | log1p(Amount Received), standardised |
| 1 | hour_sin | Numeric | transactions.csv | sin(2 * pi * hour / 24), standardised |
| 2 | hour_cos | Numeric | transactions.csv | cos(2 * pi * hour / 24), standardised |
| 3 | dow_sin | Numeric | transactions.csv | sin(2 * pi * day_of_week / 7), standardised |
| 4 | dow_cos | Numeric | transactions.csv | cos(2 * pi * day_of_week / 7), standardised |
| 5 | amount_paid_log1p | Numeric | transactions.csv | log1p(Amount Paid), standardised |
| 6-12 | pmt_{category} | One-hot (7) | transactions.csv | Payment Format one-hot: ACH, Bitcoin, Cash, Cheque, Credit Card, Reinvestment, Wire. Exactly one column = 1 per transaction. |
| 13-27 | cur_{code} | One-hot (15) | transactions.csv | Currency one-hot: one column per currency code (USD, EUR, GBP, etc.). Exactly one column = 1 per transaction. |

**A.3 Feature Engineering Design Notes**

Features are computed from the training set only. The fitted encoders (LabelEncoder for categorical node features and edge payment/currency fields, StandardScaler for numeric features) are then applied to validation and test sets without refitting. This prevents data leakage from validation and test partitions into model training.

Log1p transformation is applied to all amount and count features because transaction amounts and degree distributions are heavily long-tailed: a small number of accounts send or receive orders of magnitude more transactions and larger amounts than the typical account. Without log transformation, these extreme values would dominate the standardised feature space.

Cyclic time encoding using sine-cosine pairs ensures that temporally adjacent moments have similar feature representations. Under a linear encoding, 23:59 and 00:01 would be separated by 23.98 units; under the cyclic encoding, they are separated by the Euclidean distance between (sin(23:59), cos(23:59)) and (sin(00:01), cos(00:01)), which is small. The same principle applies to day of week, where Monday and Sunday are neighbours on the 7-day circle.

---

**Appendix B: Reproducibility Guide**

This appendix provides the complete set of commands and configuration required to reproduce all experimental results reported in this thesis.

**B.1 Environment**

All experiments were conducted with the following software versions:

- Python 3.11.6
- PyTorch 2.12.0
- PyTorch Geometric 2.8.0
- scikit-learn 1.9.0
- XGBoost 3.2.0
- NumPy 2.4.6
- Pandas 3.0.3
- SciPy 1.17.1
- Matplotlib 3.7+
- Seaborn 0.12+

Baseline and continuous-time (TGN) experiments were run with the CPU build of PyTorch; the static and snapshot GNNs were trained with a CUDA-enabled build (PyTorch 2.5.1, CUDA 12.1) with PyTorch Geometric 2.8.0. Results are deterministic within a given build; minor numerical variation can occur across CPU and GPU builds. The complete dependency list is specified in `requirements.txt` at the project root. Install with:

```
pip install -r requirements.txt
```

**B.2 Reproducibility Guarantees**

All experiments use a fixed random seed (42) across NumPy, PyTorch, and Python's random module. Data splits are deterministic: transactions are chronologically sorted, then partitioned at 70/15/15 ratios by index. Model initialisation is controlled by the fixed seed. Training procedures do not involve stochastic data augmentation.

Under these conditions, re-running any experiment with the same command-line arguments produces numerically identical results.

**B.3 Reproduction Commands**

**Conventional ML baselines (Tier 1):**

```
python experiments/run_baselines.py --variant HI-Small --seed 42
```

This trains Logistic Regression, Random Forest, and XGBoost on flat edge features and reports AUC-ROC, AUC-PR, Precision, Recall, and F1-score for all three models.

**Static GNNs (Tier 2):**

```
python experiments/run_gnn.py --variant HI-Small --model gcn --seed 42 --epochs 100
python experiments/run_gnn.py --variant HI-Small --model gat --seed 42 --epochs 100 --heads 1
python experiments/run_gnn.py --variant HI-Small --model sage --seed 42 --epochs 100 --aggregator mean
```

Or run all three sequentially:

```
python experiments/run_gnn.py --variant HI-Small --model all --seed 42 --epochs 100
```

**Snapshot temporal GNNs (Tier 3a):**

```
python experiments/run_temporal.py --variant HI-Small --model temporal_gcn --seed 42 --epochs 60
python experiments/run_temporal.py --variant HI-Small --model evolve_gcn_h --seed 42 --epochs 60 --rank 2
```

**Continuous-time TGN (Tier 3b):**

```
python experiments/run_tgn.py --variant HI-Small --epochs 100 --lr 0.003 --pos_weight_mult 0.01 --grad_clip 0 --memory_dim 64 --time_dim 8 --seed 42
```

The `--grad_clip 0` argument is critical: with the effective positive-class weight, gradient clipping destroys the minority-class learning signal (see Section 3.4.4 for the full explanation). The `--memory_dim 64 --time_dim 8` arguments pin the reported 85,905-parameter configuration.

**B.4 Data Splits**

All splits are chronological (time-based):
1. Transactions are sorted by their Unix timestamp.
2. The earliest 70% of edges are assigned to training.
3. The next 15% are assigned to validation.
4. The latest 15% are assigned to testing.

For snapshot temporal models, the 12 quantile-based windows are chronologically ordered: windows 0-7 = training, window 8 = validation, windows 9-11 = testing.

The chronological split ensures that models are trained on past transactions and evaluated on future transactions, mirroring deployment conditions.

**B.5 Expected Output**

Running all reproduction commands produces the following expected test-set metrics (F1 at each model's calibrated threshold; minor variation may occur across library versions and CPU/GPU builds):

| Model | AUC-ROC | AUC-PR | F1 |
|-------|---------|--------|-----|
| Logistic Regression | 0.9378 | 0.0378 | 0.0593 |
| Random Forest | 0.9409 | 0.1249 | 0.1857 |
| XGBoost | 0.9393 | 0.1460 | 0.1608 |
| GCN | 0.9708 | 0.2056 | 0.1971 |
| GAT (1 head) | 0.9575 | 0.0912 | 0.0898 |
| GraphSAGE | 0.9452 | 0.0412 | 0.0915 |
| TemporalGCN | 0.9514 | 0.0604 | 0.1298 |
| EvolveGCN-H | 0.9064 | 0.0504 | 0.1038 |
| TGN | 0.9698 | 0.3213 | 0.2915 |

**B.6 Project Structure**

```
src/
├── data/           - Data loading, feature engineering, graph construction
├── models/         - Model implementations (GCN, GAT, GraphSAGE, TemporalGNN, TGN, baselines)
├── training/       - Training loops and evaluation harness
└── utils/          - Configuration, metrics, logging

experiments/        - CLI runners for each model tier
docs/               - RESULTS.md, THESIS_NARRATIVE.md, report chapters
results/            - Training logs and model checkpoints
```

---

**Appendix C: Generative AI Usage Declaration**

This appendix declares the use of generative AI tools in the preparation of this thesis, in accordance with the Amsterdam University of Applied Sciences Master Project module guide requirements (Appendix F).

**Tool used:** ChatGPT (OpenAI).

**Nature of use:**
- Assistance with implementing and debugging Python code for GNN architectures and model training loops, based on the author's own architectural design decisions and research methodology.
- Reviewing chapter drafts the author had written for clarity, consistency, and grammatical correctness.
- Formatting of tables and structural organisation of report content.

**Nature of author contribution:**
- All experimental design, implementation, and execution was performed by the author.
- All research questions, methodological decisions, and conclusions were formulated by the author.
- All literature review, citation selection, and theoretical framework development was performed by the author.
- The author wrote all chapter drafts, provided all substantive content (experimental results, architectural descriptions, methodological reasoning), directed the revision process, reviewed all AI-suggested edits for accuracy and appropriateness, and takes full responsibility for the final content of this thesis.

**Verification:** All factual claims, numerical results, and citations in this thesis have been verified by the author against primary sources (experimental logs, published papers, and the assessment rubric).

---

**Representative Examples of Prompts Used**

The following examples illustrate the types of prompts used with generative AI tools during this research. The following prompts illustrate the nature of interactions with the tool.

**Prompt 1: Draft Review (Chapter 4):**

"I'm writing Chapter 4 (Results) of my AML GNN thesis. I've attached the chapter as .txt. Please review it and flag any claims that aren't supported by the numbers in my results tables. My key findings are: TGN achieves 0.968 AUC-ROC on chronological split, matching static GCN's 0.971 on random split, and TGN's AUC-PR of 0.32 is 70% higher than GCN's 0.19. I want to make sure I'm comparing fairly given the different evaluation protocols."

**Prompt 2: TGN Data Leakage Debugging (Section 3.4.4):**

"My TGN model has a training/eval mismatch: train AUC-ROC hits 0.99 by epoch 5 but validation collapses to 0.73 around epoch 10. Here's my forward() method. The memory update appears to happen before the prediction, which would introduce data leakage during training. Can you trace through the memory update order and confirm?"

---

**Appendix D: Full Results Tables**

This appendix provides the complete per-tier results tables referenced from Chapter 4, including the full precision, recall, and threshold detail summarised there. All results use the uniform chronological split.

**Table D.1: Conventional ML baseline results (chronological split, threshold 0.50).**

| Model | AUC-ROC | AUC-PR | Precision | Recall | F1 |
|-------|---------|--------|-----------|--------|-----|
| Logistic Regression | 0.9378 | 0.0378 | 0.0135 | 0.9295 | 0.0267 |
| Random Forest | 0.9409 | 0.1249 | 0.0376 | 0.7687 | 0.0717 |
| XGBoost | 0.9393 | 0.1460 | 0.0245 | 0.8706 | 0.0476 |

**Table D.2: Static GNN results (chronological split, calibrated thresholds).**

| Model | Params | AUC-ROC | AUC-PR | Precision | Recall | F1 | Thresh |
|-------|--------|---------|--------|-----------|--------|-----|--------|
| GCN | 63,489 | 0.9708 | 0.2056 | 0.1214 | 0.5234 | 0.1971 | 0.6732 |
| GAT (1 head) | 64,001 | 0.9575 | 0.0912 | 0.0487 | 0.5766 | 0.0898 | 0.5247 |
| GraphSAGE | 81,409 | 0.9452 | 0.0412 | 0.0541 | 0.2972 | 0.0915 | 0.4951 |

**Table D.3: Temporal GNN results (chronological split, calibrated thresholds; TGN warm row carries memory continuously).**

| Model | Params | AUC-ROC | AUC-PR | Precision | Recall | F1 | Thresh |
|-------|--------|---------|--------|-----------|--------|-----|--------|
| TemporalGCN | 162,561 | 0.9514 | 0.0604 | 0.1041 | 0.1724 | 0.1298 | 0.8415 |
| EvolveGCN-H | 2,213,673 | 0.9064 | 0.0504 | 0.1319 | 0.0856 | 0.1038 | 0.8415 |
| TGN (cold-start) | 85,905 | 0.9698 | 0.3213 | 0.2228 | 0.4215 | 0.2915 | 0.1585 |
| TGN (warm memory) | 85,905 | 0.9601 | 0.2708 | 0.7276 | 0.1454 | 0.2424 | 0.50 |

**Table D.4: TGN per-slice AUC-PR across all 12 chronological test slices, memory reset per slice versus carried continuously, with laundering prevalence.**

| Slice | Prevalence | AUC-PR (reset per slice) | AUC-PR (carried continuously) |
|-------|-----------|--------------------------|-------------------------------|
| 0 | 0.068% | 0.023 | 0.095 |
| 1 | 0.065% | 0.020 | 0.041 |
| 2 | 0.077% | 0.047 | 0.073 |
| 3 | 0.117% | 0.060 | 0.096 |
| 4 | 0.091% | 0.048 | 0.215 |
| 5 | 0.113% | 0.039 | 0.123 |
| 6 | 0.080% | 0.055 | 0.118 |
| 7 | 0.071% | 0.015 | 0.052 |
| 8 | 0.060% | 0.015 | 0.018 |
| 9 | 0.169% | 0.036 | 0.103 |
| 10 | 0.296% | 0.198 | 0.248 |
| 11 | 1.252% | 0.506 | 0.497 |

---

**Appendix E: Hyperparameter Configurations**

This appendix presents the hyperparameter configurations used for all neural network models (summarised in Table E.1). All experiments used these settings unless otherwise noted in the methodology chapter.

**Table E.1: Hyperparameter configurations.**

| Parameter       | Static GNNs | TemporalGCN   | EvolveGCN-H   | TGN    |
| --------------- | ----------- | ------------- | ------------- | ------ |
| Hidden dim      | 128         | 128           | 128           | 128    |
| Num layers      | 2           | 2             | 2             | N/A    |
| Dropout         | 0.3         | 0.3           | 0.3           | 0.3    |
| Learning rate   | 0.001       | 0.001         | 0.001         | 0.003  |
| Weight decay    | 0.0005      | 0.0005        | 0.0005        | 0.0005 |
| Grad clip       | 1.0         | 1.0           | 1.0           | 0      |
| Pos weight mult | 0.1         | 0.1           | 0.1           | 0.01   |
| Epochs (max)    | 100         | 60            | 60            | 100    |
| Patience        | 25          | 25            | 25            | 25     |
| Batch size      | Full graph  | Full snapshot | Full snapshot | 2048   |
| Memory dim      | N/A         | N/A           | N/A           | 64     |
| Time dim        | N/A         | N/A           | N/A           | 8      |
| EMA beta        | N/A         | N/A           | N/A           | 0.85   |
| Rank            | N/A         | N/A           | 2             | N/A    |
| GAT heads       | 1           | N/A           | N/A           | N/A    |
| SAGE aggregator | mean        | N/A           | N/A           | N/A    |

**Table E.2: TGN configuration search. Six development runs compared on the validation set under the chronological protocol, and the selected final configuration. The runs varied the model capacity (parameter count), the class-weight multiplier, the learning rate, and the gradient-clipping setting; the decisive change that enabled minority-class learning was disabling gradient clipping (Section 3.4.4). The final row is the compact configuration adopted for the reported results.**

| Run | Params | pos_weight mult | Learning rate | Best val AUC-ROC | Val AUC-PR | Minority class learned |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 289,217 | 0.10 | 0.001 | 0.89 | 0.012 | No |
| 2 | 289,217 | 0.01 | 0.003 | 0.85 | 0.011 | No |
| 3 | 102,177 | 0.01 | 0.003 | 0.66 | 0.003 | No |
| 4 | 102,177 | 0.01 | 0.003 | 0.88 | 0.012 | No |
| 5 | 118,561 | 0.01 | 0.003 | 0.84 | 0.009 | No |
| 6 | 118,561 | 0.01 | 0.003 | 0.946 | 0.087 | Yes |
| Final | 85,905 | 0.01 | 0.003 | 0.946 | 0.098 | Yes (selected) |

---

**Appendix F: Training and Validation Results**

This appendix provides the complete training, validation, and test set metrics for the conventional ML baselines (Table F.1), learning curves for the six neural models (Figure F.1), and a post-hoc feature-importance analysis of the GCN (Figure F.2). The training set metrics indicate how well each model fits the training data; the validation set metrics were used for early stopping and threshold calibration. The gap between training and test performance for the tree ensembles reflects the difficulty of generalising the minority class under extreme imbalance and the distribution shift of the chronological split.

**Table F.1: Conventional ML baseline results across all splits (chronological split, threshold 0.50).**

| Model               | Split | AUC-ROC | AUC-PR | Precision | Recall | F1     |
| ------------------- | ----- | ------- | ------ | --------- | ------ | ------ |
| Logistic Regression | train | 0.9007  | 0.0102 | 0.0060    | 0.8344 | 0.0118 |
| Logistic Regression | val   | 0.9022  | 0.0115 | 0.0071    | 0.8566 | 0.0141 |
| Logistic Regression | test  | 0.9378  | 0.0378 | 0.0135    | 0.9295 | 0.0267 |
| Random Forest       | train | 0.9963  | 0.2985 | 0.0225    | 0.9968 | 0.0439 |
| Random Forest       | val   | 0.8949  | 0.0319 | 0.0188    | 0.5868 | 0.0363 |
| Random Forest       | test  | 0.9409  | 0.1249 | 0.0376    | 0.7687 | 0.0717 |
| XGBoost             | train | 0.9796  | 0.0775 | 0.0121    | 0.8869 | 0.0238 |
| XGBoost             | val   | 0.8926  | 0.0410 | 0.0134    | 0.7632 | 0.0263 |
| XGBoost             | test  | 0.9393  | 0.1460 | 0.0245    | 0.8706 | 0.0476 |

**Figure F.1: Learning curves for the six neural models.** (`results/curves/learning_curves.png`) Validation AUC-ROC (solid) and training loss (dashed) per epoch. For every model the training loss declines smoothly to a plateau and the validation AUC-ROC converges, indicating that each model was adequately trained; GraphSAGE shows a slight late decline in validation AUC-ROC consistent with mild overfitting. Early stopping (patience 25) selects the best-validation checkpoint in each case. For the two static GNNs whose training-set ranking metrics were also logged, training and validation AUC-ROC remain close (GCN reaches training AUC-ROC 0.950 against validation 0.947; GraphSAGE 0.952 against 0.923), giving no indication of severe overfitting.

**Figure F.2: GCN permutation feature importance (top 15).** (`results/curves/permutation_importance.png`) The drop in test AUC-PR when each input feature is randomly permuted (mean over five permutations; baseline AUC-PR 0.202), for the fifteen most important features. Edge (transaction) features are shown in blue, node (account) features in orange. The ACH payment channel is by far the most important, followed by transaction amount and day-of-week; the most important account-level features are the structural ones (out-degree, in-degree, and counterparty counts). Generated by `experiments/run_permutation_importance.py` (seed 42).

---

**Appendix G: Exploratory Data Analysis Figures**

This appendix contains the figures supporting the dataset characterisation in Section 4.1. All figures are generated deterministically by the EDA script (`experiments/run_eda.py`, seed 42) from the HI-Small transactions and accounts files, and the underlying summary statistics are saved to `results/eda/eda_stats.json`.

**Figure G.1: Class balance.** (`results/eda/01_class_balance.png`) Distribution of laundering versus legitimate transactions, illustrating the 0.102% prevalence and the resulting extreme class imbalance.

**Figure G.2: Degree and counterparty distributions.** (`results/eda/02_degree_counterparties.png`) Right-skewed account degree and counterparty-count distributions on logarithmic axes, showing the heavy-tailed, hub-and-spoke topology (median degree 6, maximum 169,756).

**Figure G.3: Connected components.** (`results/eda/03_components.png`) Distribution of connected-component sizes, showing the single giant component that spans 72.2% of active accounts.

**Figure G.4: Structural position, laundering versus other accounts.** (`results/eda/04_structure_vs_laundering.png`) Comparison of degree and counterparty distributions for laundering-involved accounts against all others, showing that laundering accounts occupy higher-degree, more connected positions (median degree 22 versus 6).

**Figure G.5: Temporal distribution.** (`results/eda/05_temporal.png`) Left: transaction volume per day on a logarithmic scale, showing that volume falls from millions per day during the first ten days to only hundreds, then tens, per day thereafter. Right: daily laundering rate (line) with daily transaction volume overlaid (grey bars, secondary logarithmic axis). The high laundering rates in the final days (around 60%) are computed over this negligible tail (roughly 1,100 transactions in total), not over comparable volume; across the high-volume period (days 0 to 9) the laundering rate stays between 0.03% and 0.21%. This laundering-dense tail is a characteristic of the synthetic data generator and is discussed in relation to the per-slice analysis in Section 4.5.3.

**Figure G.6: Feature distributions by class.** (`results/eda/06_feature_distributions.png`) Distributions of transaction amount and payment channel for laundering versus legitimate transactions, showing the larger amounts (median $8,667 versus $1,408) and the elevated ACH laundering rate.

**Figure G.7: Typology signatures.** (`results/eda/07_typologies.png`) Counts of accounts exhibiting fan-out, fan-in, layering pass-through, and structuring signatures within the laundering-labelled subgraph, confirming that the FATF typologies from Chapter 2 are empirically present.
