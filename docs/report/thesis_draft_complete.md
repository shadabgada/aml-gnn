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
- 3.6 Ethical Considerations and Reproducibility

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

This gap has practical consequences. Without a rigorous, like-for-like comparison spanning all three tiers (conventional machine learning, static GNNs, and temporal GNNs, encompassing both snapshot-based and continuous-time approaches), AML compliance practitioners lack the empirical evidence needed to make informed decisions about which class of model to invest in, what performance trade-offs to expect, and under what conditions temporal modelling adds sufficient value to justify its additional complexity. The research problem is therefore both academic (an unaddressed gap in the comparative evaluation literature) and practical (insufficient evidence for practitioner model selection in AML compliance contexts).

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

**SQ4.** What practical implications do the comparative empirical findings hold for AML compliance practitioners, with respect to model selection, operational deployment, and the trade-offs between static and temporal approaches?

These sub-questions are collectively answerable using the available data and experimental infrastructure, feasible within the scope of a master's thesis, directly relevant to the identified research gap, and logically interconnected: SQ1 establishes the data foundation, SQ2 evaluates architectural choices, SQ3 provides the comparative benchmark, and SQ4 translates the aggregate findings into practice.

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

Conventional machine learning approaches to AML detection treat each transaction as an independent feature vector and apply supervised classification methods to distinguish laundering from legitimate activity. Chen et al. (2018) provided a comprehensive review of machine learning techniques applied to suspicious transaction detection, covering logistic regression, decision trees, support vector machines, and ensemble methods. Their review identified two persistent limitations: first, the extreme class imbalance inherent in AML data, where laundering transactions constitute a tiny fraction of total volume, makes standard classifiers prone to high false positive rates; second, treating transactions as independent observations discards the relational structure that characterises laundering behaviour.

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

Johannessen and Jullum (2025) applied heterogeneous GNNs to real-world banking data from DNB, Norway's largest financial institution, demonstrating that graph-based models outperformed conventional classifiers in detecting money laundering across multiple relationship types. Their work is significant because it used genuine institutional transaction data, providing ecological validity that studies on synthetic or cryptocurrency data cannot claim. However, their dataset is proprietary and non-public, making their results unreproducible by independent researchers and unusable as a benchmark for comparative evaluation.

Cheng et al. (2024) provided a comprehensive review of GNN architectures applied to financial fraud detection across domains including credit card fraud, insurance fraud, and money laundering. Their review confirmed that GCN, GAT, and GraphSAGE are the predominant static architectures used in financial fraud research, and explicitly identified the incorporation of temporal dynamics into GNN architectures as a key future research direction. This finding directly motivates the temporal modelling component of the present study.

Altman et al. (2023) published the IBM Transactions for Anti-Money Laundering dataset at NeurIPS 2023, a large-scale synthetic dataset specifically designed to serve as a public benchmark for GNN-based AML research. Their paper reported baseline results using static GNNs (GCN, GAT, and GraphSAGE) and demonstrated that all three outperformed non-graph baselines. However, the dataset paper evaluated only static architectures and explicitly noted that temporal modelling was left to future work. This creates the empirical gap that the present study addresses.

Dou et al. (2020) addressed a challenge particularly relevant to AML detection: applying GNNs to fraud detection under severe class imbalance. Their work proposed techniques to improve minority fraud node detection in imbalanced graph classification settings, including adapted loss functions and sampling strategies. The class imbalance in their experiments, while substantial, was less extreme than the approximately 1:1000 ratio in the IBM AML dataset, suggesting that additional adaptations may be necessary for AML-specific applications.

**2.4 Temporal Graph Neural Networks**

The architectures discussed in Section 2.3 operate on static graphs: all nodes and edges are treated as simultaneously present, and temporal ordering is not modelled. For many real-world graphs, including financial transaction networks, this assumption is unrealistic. Money laundering is an inherently temporal process: transactions occur in sequence, behavioural patterns evolve, and the significance of an interaction depends on when it occurs relative to prior activity. Temporal GNNs address this by incorporating time into the graph learning process. Two broad paradigms exist: snapshot-based approaches that discretise time into a sequence of static graphs, and continuous-time approaches that process individual events with their exact timestamps.

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

He and Garcia (2009) provided a comprehensive analysis of class imbalance challenges in machine learning, reviewing resampling techniques, cost-sensitive learning approaches, and evaluation metric selection. They demonstrated that precision, recall, and the F1-score provide more informative performance assessment than accuracy under class imbalance, and that the Area Under the Receiver Operating Characteristic curve (AUC-ROC) and the Area Under the Precision-Recall curve (AUC-PR) offer complementary perspectives on model discrimination. AUC-ROC measures overall discriminative power across all classification thresholds and is insensitive to class distribution, making it useful for comparing models across datasets with different imbalance ratios. AUC-PR, by contrast, focuses on the minority class and is more sensitive to improvements in detecting the rare positive cases. For AML detection, where the operational cost of false negatives (missed laundering) is high but the cost of false positives (alert fatigue from over-alerting) is also substantial, both metrics are relevant.

The choice of loss function during training is equally important. Standard binary cross-entropy loss treats false positives and false negatives symmetrically, which is inappropriate when the classes are severely imbalanced. Weighted binary cross-entropy, where the minority class contribution is scaled by a factor inversely proportional to its prevalence, provides a standard remedy. However, the magnitude of the weight introduces a new consideration. With a laundering prevalence of approximately 0.1%, the inverse-frequency weight exceeds 1000. This means minority class gradients are three orders of magnitude larger than majority class gradients during training, which has implications for gradient-based optimisation that are discussed in the context of model-specific training in Chapter 3.

Dou et al. (2020) addressed the intersection of GNNs and class imbalance, proposing techniques specific to graph-based fraud detection. Their work demonstrated that standard GNN training procedures can be inadequate under extreme class imbalance because the message-passing mechanism propagates information from both classes, potentially diluting the minority class signal. They proposed adapted training strategies, including class-balanced sampling of training edges, which inform the training protocol adopted in this study.

**2.6 Research Gap Synthesis**

The literature reviewed in this chapter converges on an identifiable gap. The IBM AML dataset (Altman et al., 2023) was designed and published as a public benchmark for GNN-based AML research, providing a standardised platform on which different architectures can be compared under identical conditions. The dataset paper established baseline results for static GNNs (GCN, GAT, and GraphSAGE) and demonstrated that they outperform non-graph baselines. However, the paper did not evaluate any temporal GNN architectures, explicitly leaving temporal modelling to future work.

Separately, the temporal GNN literature has developed architectures capable of capturing transaction dynamics: snapshot-based approaches such as TemporalGCN and EvolveGCN (Pareja et al., 2020), and continuous-time approaches such as TGN (Rossi et al., 2020). These architectures have been evaluated on social network, citation, and cryptocurrency datasets, but none of the three has previously been applied to a public banking AML benchmark. EvolveGCN was evaluated on Bitcoin OTC and Reddit (Pareja et al., 2020), the temporal GCN of Alarab and Prakoonwit (2023) was applied to the Elliptic Bitcoin dataset, and TGN was evaluated on Wikipedia, Reddit, and Twitter (Rossi et al., 2020). The present study is the first to apply all three to a standardised banking AML benchmark, and the first to apply TGN to AML edge classification in any domain. The single published study applying temporal GNNs to AML (Alarab & Prakoonwit, 2023) used the Elliptic Bitcoin dataset, which represents a fundamentally different transaction domain and used a snapshot-based rather than continuous-time approach.

The consequence of this gap is twofold. Academically, there is no published evidence on whether continuous-time temporal modelling improves AML detection over static GNNs on a standardised banking benchmark. Practically, AML compliance teams lack the comparative empirical evidence needed to assess whether the additional complexity of temporal GNN architectures is justified by a meaningful improvement in detection performance.

The present study addresses this gap by conducting a systematic three-tier comparative evaluation: conventional machine learning classifiers, static GNN architectures, and temporal GNN architectures spanning both snapshot-based and continuous-time approaches, all trained and evaluated on the same IBM AML HI-Small dataset under identical experimental conditions. The theoretical framework established in this chapter provides the basis for the architectural choices and evaluation protocol described in Chapter 3.

---

# Chapter 3: Research Methodology and Tool Development

**3.1 Research Design Overview**

This study is classified as applied research with a tool development orientation. The research design follows a deductive approach in which theoretical frameworks from the academic literature on graph neural networks and anti-money laundering detection, as reviewed in Chapter 2, are operationalised as a concrete analytical tool and empirically tested on a standardised financial transaction dataset (Sekaran & Bougie, 2019).

The research design comprises five stages. First, a structured literature study established the theoretical foundation for GNN-based AML detection, identified the appropriate architectures, and confirmed the research gap motivating the comparative evaluation. Second, the IBM AML HI-Small dataset was selected through a systematic comparison against an alternative candidate and subjected to data engineering to construct graph representations suitable for both static and temporal GNN analysis. Third, nine model architectures spanning three tiers were implemented: three conventional supervised classifiers (Logistic Regression, Random Forest, XGBoost), three static GNNs (GCN, GAT, GraphSAGE), and three temporal GNNs (TemporalGCN, EvolveGCN-H, TGN). Fourth, all models were trained and evaluated under identical experimental conditions using metrics appropriate for heavily class-imbalanced data. Fifth, the empirical findings were analysed comparatively and translated into practitioner guidance.

The research design intentionally spans three tiers, rather than comparing GNN variants alone, to isolate the contribution of graph structure and temporal modelling to detection performance. Tier 1 (conventional ML) establishes the performance achievable without access to graph structure. Tier 2 (static GNNs) measures the gain from relational modelling. Tier 3 (temporal GNNs) measures the additional gain from temporal modelling and compares snapshot-based against continuous-time approaches. This design allows the study to answer not only which model performs best, but why.

During implementation, the scope of the temporal modelling tier expanded beyond the single architecture (EvolveGCN) specified in the research plan. Preliminary experiments revealed that snapshot-based temporal models underperformed the static GCN, which prompted the addition of TemporalGCN to verify whether the limitation was specific to EvolveGCN's weight-space evolution or inherent to the snapshot paradigm, and subsequently the addition of continuous-time TGN to test whether finer temporal granularity could overcome the limitation. This expansion was a data-driven methodological decision, grounded in empirical observations made during the research process. It is documented here transparently as part of the methodological narrative.

**3.2 Dataset: IBM AML HI-Small**

**3.2.1 Dataset Selection and Justification**

The dataset used in this study is the IBM Transactions for Anti-Money Laundering dataset (Altman et al., 2023), publicly available on Kaggle (https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml). The IBM AML dataset was chosen over the main alternative, the Synthetic AML Dataset (SAML-D; Oztas et al., 2023), for two reasons. First, the IBM AML dataset is structured natively as a graph: a dedicated accounts file defines each unique account as a persistent entity, and a transactions file captures directed interactions between accounts. This maps directly onto the node-and-edge representation required for GNN-based analysis. SAML-D, by contrast, is a flat tabular dataset without an explicit account-level structure, making the construction of stable node identities (a prerequisite for temporal GNNs) a non-trivial and ambiguous preprocessing step. Second, the IBM AML dataset's laundering patterns are derived from FATF-documented AML typologies, including structuring, layering, and fan-in/fan-out schemes (Altman et al., 2023; FATF, 2023), ensuring the synthetic patterns reflect real-world regulatory knowledge. The dataset was published at NeurIPS 2023 specifically as a public benchmark for GNN-based AML research (Altman et al., 2023).

The dataset is available in four variants (HI/LI combined with Small/Medium). This study uses the HI-Small variant (518,581 accounts, 5,078,345 transactions, 5,177 laundering, 0.102% prevalence). The HI variants contain a higher laundering ratio, providing more positive cases for training. The Small variant was chosen for computational feasibility: all models were trained on CPU, and the Medium variants (tens of millions of transactions) would have made full training runs for all nine architectures infeasible within the project timeline. The four variants share an identical data-generating process, so architectural findings from HI-Small are expected to generalise, though empirical verification on larger variants is noted as future work.

**3.2.2 Dataset Characteristics**

The HI-Small dataset comprises two CSV files. The accounts file contains 518,581 rows with five fields: Bank Name, Bank ID, Account Number, Entity ID, and Entity Name. The transactions file contains 5,078,345 rows with eleven fields: Timestamp, From Bank, Account (from), To Bank, Account (to), Amount Received, Receiving Currency, Amount Paid, Payment Currency, Payment Format, and Is Laundering.

Account identity is composite, formed by concatenating Bank ID and Account Number. This composite key is consistent across the accounts and transactions files, ensuring unambiguous mapping between account entities and their transaction history. Transaction timestamps are provided as string-formatted dates (for example, "2022/09/01 00:20"), which are parsed to Unix seconds for temporal modelling. The timestamp range spans approximately 18 days (2022/09/01 through 2022/09/19), representing a compressed but transaction-dense financial activity period.

The accounts file characterises each of the 518,581 accounts with bank membership, a unique account identifier, and an entity type derived from the Entity Name field. Entity types include corporations, individuals, shell companies, and other categories. These entity types apply to both senders and receivers: every transaction links a sending account to a receiving account, and each account's entity type and bank affiliation are known from the accounts file. The account-level features (Section 3.3.1) are derived from this information combined with aggregated transaction statistics computed from the training set.

The payment format field contains seven categories: ACH, Cheque, Credit Card, Domestic Wire, International Wire, Cash, and an unknown or missing category. The currency fields include 15 currency codes. These categorical variables form the basis for one-hot encoded edge features, as described in Section 3.3.1.

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

The evaluation protocol differs between model tiers. For conventional ML baselines and static GNNs, a random 70/15/15 split was applied across the full dataset. Random splitting is standard practice for static models because these architectures treat the graph as a fixed structure without temporal ordering. For temporal models (TemporalGCN, EvolveGCN-H, TGN), a chronological (time-based) data split was used. For snapshot temporal models, the 12 snapshots were chronologically ordered and assigned: snapshots 0 through 7 to training, snapshot 8 to validation, and snapshots 9 through 11 to testing. For TGN, the continuous temporal edge stream was partitioned at the same 70/15/15 ratios by edge index after chronological sorting to ensure strict temporal ordering: all training edges precede all validation edges, which precede all test edges.

This chronological split strategy has a specific advantage over random shuffling: it evaluates models under deployment-realistic conditions. In a production AML system, models are trained on historical data and must detect laundering in future transactions. Random splits, which mix past and future edges across train and test sets, introduce a subtle form of data leakage: the model sees edges from the future during training and edges from the past during testing, inflating performance estimates relative to real-world deployment conditions. Several published AML GNN studies have used random splits (Altman et al., 2023; Weber et al., 2019). This study's use of chronological splits provides a more honest and deployment-relevant performance estimate, though it also makes the evaluation task harder, a point discussed in the cross-model comparison in Chapter 4.

A consequence of chronological splitting is that the class distribution varies across partitions, since the laundering ratio is not constant over time. In the IBM AML HI-Small dataset, the laundering ratio increases from approximately 0.01% in the earliest time window to 0.30% in the latest. The chronological split means that the test set has a higher laundering prevalence than the training set, which is both realistic (laundering patterns may intensify over time in a real system) and challenging (the model is evaluated on a distribution that differs from its training distribution). The pos_weight for loss computation was computed from the training set only, consistent with the principle that no test-set information may influence model training.

**3.4 Model Architectures (SQ2 and SQ3)**

This section addresses SQ2 by describing the implementation of each model architecture and justifying the design choices with reference to the theoretical framework established in Chapter 2.

**3.4.1 Conventional ML Baselines**

Three supervised classifiers were implemented as baselines that operate on flat feature vectors without access to graph structure: Logistic Regression, Random Forest, and XGBoost. These models were selected to represent a progression of complexity and to establish the performance floor against which GNN-based models are compared, directly addressing SQ3.

**Logistic Regression** was implemented using scikit-learn (Pedregosa et al., 2011) with L2 regularisation and the liblinear solver. Class imbalance was addressed via class_weight="balanced", which automatically weights the minority class inversely proportional to its frequency in the training set. No sample_weight was applied, avoiding the double-weighting issue in which class_weight and sample_weight simultaneously scale the minority class loss, effectively squaring the intended penalty.

**Random Forest** (Breiman, 2001) was implemented with 100 estimators, a maximum depth of 10, and class_weight="balanced". The depth limit was applied to mitigate overfitting to the majority class, which preliminary experiments showed was severe without regularisation: unconstrained trees would grow deep enough to memorise individual legitimate transactions, producing near-perfect training scores but generalising poorly.

**XGBoost** (Chen & Guestrin, 2016) was implemented with default hyperparameters and early stopping configured to monitor validation set log loss with a patience of 20 rounds. Early stopping on the validation set, rather than on training data as in an earlier implementation, is methodologically important: monitoring training data provides no signal about generalisation and can lead to overfitting.

All three baselines receive the identical 28-dimensional edge feature vectors used by the GNN models. The key difference is that the baselines treat each edge independently, while the GNNs additionally receive node features and the graph adjacency structure. Any performance difference between the baselines and the GNNs can therefore be attributed to the graph structural information, since the edge-level input features are held constant.

**3.4.2 Static GNNs**

The three static GNN architectures described in Section 2.3.1 were implemented using PyTorch Geometric (Fey & Lenssen, 2019). All three share a common architectural template: an edge classification model consisting of node encoding layers, edge feature projection, and a classifier head.

**GCN (Kipf & Welling, 2017).** The implementation uses two GCN convolutional layers with hidden dimension 128 and ReLU activation. Each GCN layer applies the symmetric normalised graph Laplacian convolution to propagate node features across edges. After convolution, the final-layer node embeddings for the source and destination nodes of each edge are concatenated with the projected edge features and passed through a two-layer MLP classifier with dropout (p=0.3) to produce a scalar logit per edge. The total parameter count is 63,489.

**GAT (Veličković et al., 2018).** The implementation uses two GAT convolutional layers with hidden dimension 128 and a single attention head. The original GAT paper reported that multi-head attention (typically 4 or 8 heads) was important for stable training. However, preliminary experiments with 4 heads on the full HI-Small graph (5 million edges) caused memory exhaustion on CPU. With a single head, the model has 64,001 parameters and completed training successfully. The use of a single head likely reduces the expressiveness of the attention mechanism, since the model cannot attend to different relational patterns in parallel, but trade-offs of this kind are unavoidable when training large-graph models on CPU-constrained hardware.

**GraphSAGE (Hamilton et al., 2017).** The implementation uses two SAGEConv layers with hidden dimension 128 and mean aggregation. Mean aggregation was chosen over max or LSTM aggregation for computational efficiency: max aggregation discards distributional information about neighbour features, and LSTM aggregation imposes an arbitrary ordering on an unordered neighbour set. The implementation uses neighbourhood sampling with a fixed sample size and L2 normalisation of embeddings, which the original paper found important for training stability. The total parameter count is 81,409.

**3.4.3 Snapshot Temporal GNNs**

**TemporalGCN.** The implementation consists of a shared two-layer GCN (128 hidden dimensions) that processes each snapshot independently, combined with a GRU (Cho et al., 2014) that evolves per-node hidden states across the snapshot sequence. After the GCN produces node embeddings for snapshot t, the GRU updates each node's hidden state as a learned combination of its previous state and the new GCN output. The GRU hidden dimension matches the GCN output dimension (128). The edge classifier concatenates source and destination node states with edge features and passes them through a two-layer MLP. The key difference from the static GCN is that the node states feeding into the edge classifier are not raw GCN outputs but GRU-evolved states that incorporate information from all preceding snapshots. The total parameter count is 161,793.

**EvolveGCN-H (Pareja et al., 2020).** EvolveGCN-H evolves the GCN weight matrices across snapshots rather than per-node states. The GCN weight matrix at snapshot t is expressed as the sum of a base weight matrix and a low-rank adaptation: W\_t = W\_base + A\_t @ B\_t, where A\_t and B\_t are low-rank factors. A GRU receives the mean-pooled node embedding from the previous snapshot as input and produces updated weight factors for the next snapshot. The low-rank dimension is a critical hyperparameter: it determines both the expressive capacity of the weight adaptation and the total parameter count.

Two implementations were attempted. With rank 8, the model had 33 million parameters, exceeding available CPU memory. Reducing the rank to 2 produced a model with 578,369 parameters that could be trained, but at substantially reduced expressive capacity. This parameter explosion is inherent to the EvolveGCN-H design: the GRU hidden state dimension scales as rank multiplied by the sum of input and output GCN dimensions, growing quadratically with the GCN layer dimensions. The reduced-rank model was used for the results reported in Chapter 4. This limitation is discussed in Section 4.3.

**3.4.4 Continuous-Time TGN**

The Temporal Graph Network (TGN; Rossi et al., 2020) was implemented with a custom EMA-based memory module, rather than the GRU-based TGNMemory provided by PyTorch Geometric. This decision and several others emerged from a development process in which four implementation issues were discovered and resolved. The resolution of each issue produced a methodological insight that is documented here.

**Architecture.** The TGN model comprises five components. First, a TimeEncoder maps raw timestamp differences to a 16-dimensional sinusoidal encoding, using the sine-cosine transformation from the Transformer positional encoding (Vaswani et al., 2017) applied at 16 logarithmically spaced frequencies. This encoding captures temporal patterns at multiple scales, from sub-minute to multi-day intervals. Second, a NodeProjection MLP maps each node's raw features to an initial 128-dimensional memory state. Third, a MessageProjection MLP transforms the concatenation of source memory, destination memory, edge features, and time encoding into a 128-dimensional message vector representing the information content of the current interaction. Fourth, an EdgeClassifier MLP concatenates source memory, destination memory, projected edge features, time encoding, and the projected message, and passes the result through a two-layer MLP with dropout (p=0.3) to produce a scalar logit. Fifth, an EMAMemory module maintains and updates per-node memory states. The total parameter count is 289,217 (using the default memory_dim=128, time_dim=16), or 119,000 in the more compact configuration used for the final results (memory_dim=64, time_dim=8).

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

Hyperparameters were set based on architectural defaults from the original papers, with manual adjustment where needed for training stability on this dataset. No automated hyperparameter optimisation (grid search, random search, or Bayesian optimisation) was performed, which is acknowledged as a limitation. The full hyperparameter configurations for all models are provided in Appendix E.

**Training duration.** All models were trained on CPU (Intel Core i7, 8 threads). Approximate training times were: Logistic Regression 2 minutes, Random Forest 10 minutes, XGBoost 3 minutes, GCN 102 minutes, GAT 154 minutes (1 head), GraphSAGE 55 minutes, TemporalGCN 65 minutes, EvolveGCN-H 50 minutes, TGN 114 minutes (100 epochs, early stopped at epoch 38). The total computational investment across all models was approximately 9 CPU-hours.

**3.5.3 Evaluation Metrics and Threshold Calibration**

All models were evaluated using five metrics: AUC-ROC, AUC-PR, Precision, Recall, and F1-score. As established in Section 2.5, these metrics are appropriate for heavily class-imbalanced data because they are not inflated by the majority class, unlike accuracy.

Precision, Recall, and F1-score are threshold-dependent: they are computed at a specific classification threshold. Reporting these metrics at a single default threshold (0.5) is standard practice but can be misleading when the optimal threshold for the minority class differs substantially from 0.5, as is typical under extreme class imbalance. To address this, each model's classification threshold was calibrated on the validation set by selecting the threshold that maximised validation F1-score. Both default-threshold (0.5) and calibrated-threshold metrics are reported in Chapter 4. The calibrated threshold was then applied to the test set for the final evaluation. This calibration procedure ensures that threshold-dependent metrics reflect each model's best achievable performance rather than an arbitrary cutoff.

For TGN, an additional evaluation was performed: per-time-slice analysis. The chronologically ordered test set was divided into 12 equal slices, and metrics were computed independently for each slice. This analysis tests whether model performance improves as more interaction history accumulates in per-node memory, providing evidence for or against temporal generalisation. A model whose performance is flat across slices shows no benefit from memory accumulation; a model whose performance improves monotonically across slices demonstrates that per-node memory captures useful behavioural signal over time.

**3.6 Ethical Considerations and Reproducibility**

**Ethical considerations.** This research uses a synthetic dataset (IBM AML HI-Small) that does not contain real personal or financial information. No primary data collection from human subjects was conducted. The dataset is publicly available and was created specifically for academic research purposes (Altman et al., 2023). The laundering labels are synthetic and do not represent accusations against real individuals or institutions.

Two ethical considerations nonetheless apply. First, the AML detection tool developed in this research could, if deployed, contribute to automated decision-making with significant consequences for individuals whose accounts are flagged as suspicious. The tool is an analytical prototype, not a production system, and its outputs should be understood as decision support for human compliance analysts, not as automated determinations of criminal activity. This distinction is important: the models presented here detect patterns statistically associated with laundering, not laundering itself. Second, the model's performance characteristics have fairness implications. If the underlying transaction data reflects biases in which accounts or transaction patterns are flagged as suspicious, the model may amplify those biases. The IBM AML dataset's laundering patterns are derived from FATF typologies rather than from real-world enforcement data, which mitigates but does not eliminate this concern.

**Reproducibility.** All experiments used a fixed random seed (42) across NumPy, PyTorch, and Python's random module. Data splits are deterministic: chronological sort followed by index-based partitioning at 70/15/15 ratios. Model initialisation is controlled by the fixed seed. Training procedures do not involve stochastic data augmentation. Under these conditions, re-running any experiment with the same arguments produces identical results.

Complete reproducibility requires Python 3.11, PyTorch 2.x, PyTorch Geometric 2.5.x, scikit-learn 1.x, XGBoost 2.x, NumPy, and Pandas. The full dependency list with version pinning, reproduction commands, and data split documentation are provided in Appendix B.

**Tool documentation.** The tool's architecture, module structure, and development process are documented within this chapter (Sections 3.3-3.5) and in the reproducibility guide in Appendix B. Appendix A provides the complete feature specification (12 node features, 28 edge features) referenced in Section 3.3.1. Appendix D contains the full results tables for every model. Together, these materials are intended to enable an independent researcher or practitioner to understand, reproduce, and adapt the tool.

---

# Chapter 4: Results, Analyses and Tool Performance

This chapter presents the empirical results of the three-tier comparative evaluation described in Chapter 3. Results are reported for each model tier in sequence, followed by a cross-model comparison that synthesises the findings across tiers, and an assessment of tool scalability and generalizability. All results are from the IBM AML HI-Small dataset.

**4.1 Baseline Results: Conventional Machine Learning (Tier 1)**

Table 4.1 presents the test set performance of the three conventional supervised classifiers. These models operate on flat edge feature vectors without access to graph structure or temporal information, establishing the performance floor against which GNN-based models are compared. Training and validation set results are provided in Appendix F.

**Table 4.1: Conventional ML baseline results on the test set (random 70/15/15 split, threshold 0.50).**

| Model               | AUC-ROC | AUC-PR | Precision | Recall | F1     |
| ------------------- | ------- | ------ | --------- | ------ | ------ |
| XGBoost             | 0.9381  | 0.1511 | 0.0265    | 0.8610 | 0.0514 |
| Random Forest       | 0.8603  | 0.0619 | 0.0035    | 0.9148 | 0.0070 |
| Logistic Regression | 0.9378  | 0.0376 | 0.0135    | 0.9295 | 0.0267 |

XGBoost is the strongest conventional classifier, achieving AUC-ROC 0.9381 and AUC-PR 0.1511. Logistic Regression matches XGBoost on AUC-ROC (0.9378) but achieves substantially lower AUC-PR (0.0376), indicating that its strong ranking performance does not translate to effective identification of the minority class. At the default 0.5 threshold, Logistic Regression achieves very high recall (0.9295) but near-zero precision (0.0135): it flags nearly all laundering transactions, but at the cost of an overwhelming false positive rate that would be operationally unworkable.

Random Forest underperforms both alternatives across all metrics (AUC-ROC 0.8603, F1 0.0070). The depth limit of 10, applied to prevent overfitting to the majority class, appears to have been too restrictive for the complex decision boundary required under extreme class imbalance. Without the depth constraint, preliminary experiments showed severe overfitting; with it, the model lacks sufficient capacity.

The key insight from the baseline tier is that even the best conventional classifier (XGBoost) achieves precision of only 0.0265 at the default threshold. For every 1,000 transactions flagged as suspicious, approximately 974 are false positives. This reflects the fundamental limitation identified in Section 2.2: without access to relational information, individual transaction features carry limited signal for distinguishing laundering from legitimate activity.

**4.2 Static GNN Results: Graph Structure Without Time (Tier 2)**

Table 4.2 presents the test set performance of the three static GNN architectures. These models incorporate graph structure through message passing but treat all transactions as simultaneously present, without temporal ordering.

**Table 4.2: Static GNN results on the test set (random 70/15/15 split, calibrated thresholds).**

| Model        | Params | AUC-ROC | AUC-PR | Precision | Recall | F1     | Thresh |
| ------------ | ------ | ------- | ------ | --------- | ------ | ------ | ------ |
| GCN          | 63K    | 0.9705  | 0.1882 | 0.1846    | 0.3933 | 0.2513 | 0.7029 |
| GAT (1 head) | 64K    | 0.9581  | 0.0958 | 0.0539    | 0.5317 | 0.0979 | 0.5544 |
| GraphSAGE    | 81K    | 0.9459  | 0.0420 | 0.0563    | 0.2953 | 0.0946 | 0.4852 |

GCN is the strongest static GNN, achieving AUC-ROC 0.9705 and AUC-PR 0.1882 with only 63,489 parameters. At its calibrated threshold of 0.7029, GCN detects 39.3% of laundering transactions at 18.5% precision. Compared to the best baseline (XGBoost, AUC-PR 0.1511), GCN adds 0.0371 AUC-PR, confirming that graph structural information contributes measurable detection value beyond what flat features provide.

GAT underperforms GCN (AUC-ROC 0.9581, AUC-PR 0.0958) despite its theoretically more expressive attention mechanism. The single-head configuration, necessitated by CPU memory constraints (4 heads caused OOM on the 5-million-edge graph), likely limits the model's capacity to learn multiple relational patterns in parallel. The original GAT paper (Veličković et al., 2018) reported that multi-head attention was important for stable training and performance; the single-head result here is consistent with that finding. This is a hardware constraint rather than an architectural limitation of GAT, and is discussed as a limitation in Section 5.4.

GraphSAGE achieves the lowest static GNN performance (AUC-ROC 0.9459, AUC-PR 0.0420). Mean aggregation with neighbourhood sampling, while computationally efficient, appears to lose discriminative signal. In a graph where laundering accounts are structurally distinctive (high out-degree, unusual counterparty patterns), averaging neighbour features may dilute the very signal the model needs to detect. LSTM or max aggregation might preserve more of this signal, at increased computational cost.

Comparing these results to the original IBM AML dataset paper (Altman et al., 2023), the GCN performance reported here (AUC-ROC 0.9705) is broadly consistent with their findings, though direct numeric comparison is complicated by differences in data split strategy and evaluation protocol.

**4.3 Temporal GNN Results: Graph Structure With Time (Tier 3)**

This section presents results for the three temporal GNN architectures. Unlike the static GNNs in Section 4.2, which were evaluated on a random 70/15/15 split, the temporal models were evaluated on a chronological split: trained on the earliest 70% of transactions, validated on the next 15%, and tested on the latest 15%. This protocol is deployment-realistic (Section 3.3.3) but also inherently harder: the model must detect laundering in a future time period using patterns learned from the past.

**4.3.1 Snapshot-Based Temporal Models**

Table 4.3 presents the test set results for the two snapshot-based temporal architectures.

**Table 4.3: Snapshot temporal GNN results on the test set (chronological split, calibrated thresholds).**

| Model       | Params | AUC-ROC | AUC-PR | Precision | Recall | F1     | Thresh |
| ----------- | ------ | ------- | ------ | --------- | ------ | ------ | ------ |
| TemporalGCN | 162K   | 0.9570  | 0.0637 | 0.1177    | 0.1563 | 0.1343 | 0.7326 |
| EvolveGCN-H | 578K   | 0.8972  | 0.0275 | 0.0465    | 0.0982 | 0.0631 | 0.7029 |

TemporalGCN achieves AUC-ROC 0.9570 with 161,793 parameters. Despite incorporating temporal information through GRU-evolved node states across 12 snapshots, it underperforms the static GCN (AUC-ROC 0.9705, AUC-PR 0.1882). The difference is partly attributable to the harder chronological evaluation protocol, but the magnitude of the gap (0.1245 AUC-PR) suggests that the 12-snapshot temporal resolution is too coarse to capture laundering patterns. Structuring and layering schemes that unfold across individual transactions within a single snapshot window are invisible to the model.

EvolveGCN-H is the weakest GNN across all three tiers (AUC-ROC 0.8972, AUC-PR 0.0275). Two implementations were attempted. With the rank parameter set to 8, the model had 33 million parameters, exceeding available CPU memory. Reducing the rank to 2 produced a 578,369-parameter model that could be trained, but at substantially reduced expressive capacity. The parameter explosion is inherent to the EvolveGCN-H design: the GRU hidden state dimension scales as rank multiplied by (input dimension plus output dimension), growing quadratically with GCN layer dimensions. Even at rank 2, EvolveGCN-H has more parameters than GCN and GAT combined (578K vs 63K + 64K) yet achieves the worst performance, indicating that weight-space evolution is not merely expensive but architecturally unstable for this task.

**4.3.2 Continuous-Time TGN**

Table 4.4 presents the test set results for the continuous-time TGN.

**Table 4.4: TGN results on the test set (chronological split, calibrated thresholds).**

| Model | Params | AUC-ROC | AUC-PR | Precision | Recall | F1     | Thresh |
| ----- | ------ | ------- | ------ | --------- | ------ | ------ | ------ |
| TGN   | 119K   | 0.9684  | 0.3195 | 0.4257    | 0.3011 | 0.3527 | 0.4159 |

TGN achieves AUC-ROC 0.9684 and AUC-PR 0.3195 with 119,000 parameters. This is the best overall result across all three tiers. At its calibrated threshold of 0.4159, TGN detects 30.1% of laundering transactions at 42.6% precision: for every 100 alerts generated, approximately 43 are genuine laundering cases.

Two comparisons are essential for interpreting these results. First, TGN's AUC-ROC (0.9684) matches GCN's (0.9705) despite being evaluated on a harder protocol (chronological vs random split). Chronological evaluation prevents the model from seeing future transactions during training, making the task more representative of real deployment. That TGN matches GCN's AUC-ROC under these stricter conditions is a strong result. Second, TGN's AUC-PR (0.3195) is 5.0 times higher than TemporalGCN's (0.0637) under the same chronological evaluation protocol. Since both models are evaluated identically, this gap can be attributed to architectural differences: continuous-time processing with individual timestamps versus coarse snapshot bucketing.

The development of the TGN implementation involved resolving four methodological issues, as documented in Section 3.4.4: a PyG TGNMemory dtype incompatibility, a GRU gradient disconnection, a data leakage bug in which training used updated memory while evaluation used old memory, and the interaction between gradient clipping and class-weighted loss. Each issue was identified, resolved, and documented as a methodological finding. The final configuration, with EMA memory (beta=0.85), disabled gradient clipping, and predictions always computed from old memory state, produced the results reported here.

**4.3.3 TGN Temporal Generalisation: Per-Slice Analysis**

Table 4.5 presents TGN performance across individual time slices of the chronologically ordered test set. The test set (15% of all transactions, approximately 760,000 edges) was divided into 12 equal slices by edge count after chronological sorting, matching the 12-window configuration used for the snapshot-based temporal models to enable a like-for-like temporal resolution comparison. Metrics were computed independently for each slice. This analysis tests whether model performance improves as per-node memory accumulates interaction history.

**Table 4.5: TGN per-slice performance (threshold 0.50, selected slices).**

| Slice             | AUC-ROC | AUC-PR |
| ----------------- | ------- | ------ |
| 0 (earliest test) | 0.9205  | 0.0502 |
| 3                 | 0.9280  | 0.0853 |
| 6                 | 0.9714  | 0.0712 |
| 9                 | 0.9591  | 0.0769 |
| 10                | 0.9563  | 0.1875 |
| 11 (latest test)  | 0.9732  | 0.4518 |

AUC-PR improves from 0.0502 in the earliest test slice to 0.4518 in the latest, a factor of 9.0. AUC-ROC improves from 0.9205 to 0.9732. The upward trend is not monotonic across all slices (slice 6 achieves higher AUC-ROC than slice 9, for example), reflecting natural variation in laundering prevalence and difficulty across time windows. The overall trajectory, however, is unambiguously positive.

This improvement is not a training effect. Model weights are frozen at test time. What changes across slices is the content of per-node memory: each transaction processed by the model updates the source and destination accounts' memory states via the EMA mechanism. An account active in slice 0 only has memory accumulated during the training period. By slice 11, the same account's memory encodes information from the training period plus all preceding test-period transactions.

The practical interpretation is that TGN gets better the longer it runs. An account that begins exhibiting laundering behaviour mid-way through the test period cannot be detected in earlier slices (the behaviour has not occurred yet), but by later slices the model has observed the account's full behavioural arc and can recognise the laundering pattern. A static GCN would show a flat performance line across slices, since it has no memory and classifies each edge independently. The rising curve is direct evidence that per-node memory accumulates behaviourally useful signal over time.

**4.4 Cross-Model Comparison**

Table 4.6 presents the test set performance of all nine models in a unified leaderboard, ordered by AUC-PR. The evaluation protocol column is essential for fair comparison: models evaluated on random splits are being tested on an easier task than those evaluated on chronological splits.

**Table 4.6: Complete model leaderboard on the test set, ordered by AUC-PR.**

| Tier     | Model       | Params | AUC-ROC | AUC-PR | F1     | Eval Split    |
| -------- | ----------- | ------ | ------- | ------ | ------ | ------------- |
| Temporal | TGN         | 119K   | 0.9684  | 0.3195 | 0.3527 | Chronological |
| Static   | GCN         | 63K    | 0.9705  | 0.1882 | 0.2513 | Random        |
| Conv     | XGBoost     | N/A    | 0.9381  | 0.1511 | 0.0514 | Random        |
| Static   | GAT         | 64K    | 0.9581  | 0.0958 | 0.0979 | Random        |
| Temporal | TemporalGCN | 162K   | 0.9570  | 0.0637 | 0.1343 | Chronological |
| Conv     | RF          | N/A    | 0.8603  | 0.0619 | 0.0070 | Random        |
| Static   | GraphSAGE   | 81K    | 0.9459  | 0.0420 | 0.0946 | Random        |
| Conv     | LR          | N/A    | 0.9378  | 0.0376 | 0.0267 | Random        |
| Temporal | EvolveGCN-H | 578K   | 0.8972  | 0.0275 | 0.0631 | Chronological |

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

---

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

Continuous-time temporal GNNs with per-node memory (TGN) decisively outperform both static GNNs and conventional machine learning classifiers for AML detection under deployment-realistic chronological evaluation. The performance hierarchy is: **TGN > GCN > XGBoost > GAT > TemporalGCN > GraphSAGE > EvolveGCN-H**, measured by AUC-PR, the metric most sensitive to minority class detection quality.

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

| Factor                            | Tier 1: XGBoost           | Tier 2: GCN                | Tier 3: TGN                |
| --------------------------------- | ------------------------- | -------------------------- | -------------------------- |
| AUC-PR                            | 0.151                     | 0.188                      | 0.320                      |
| Precision at calibrated threshold | ~0.03                     | ~0.18                      | ~0.43                      |
| Infrastructure requirements       | Low                       | Moderate                   | Substantial                |
| Interpretability                  | High (feature importance) | Moderate (node embeddings) | Moderate (memory states)   |
| Temporal dynamics                 | Not modelled              | Not modelled               | Modelled (continuous-time) |
| Deployment complexity             | Low                       | Moderate                   | High                       |

**5.3.2 Precision-Recall Trade-offs and Operational Alert Burden**

The precision-recall trade-off has direct operational consequences for compliance team workload. At the calibrated threshold, TGN generates approximately 43 true positives for every 100 alerts, with the remaining 57 being false positives. At a laundering prevalence of 0.1%, this means that for every 100,000 transactions processed, approximately 100 are genuine laundering cases. TGN at its calibrated threshold would flag approximately 70 transactions, of which roughly 30 would be genuine laundering and 40 would be false positives. This alert volume (70 per 100,000 transactions, or roughly 700 per million) is within the review capacity of a typical compliance team.

In contrast, XGBoost at its default threshold flags approximately 3,250 transactions per 100,000 (based on its 0.0265 precision and 0.8610 recall, flagging 86% of 100 laundering cases at 2.7% precision), producing approximately 3,150 false positives for every 86 true positives. An analyst reviewing 100 alerts per day would see approximately 3 genuine laundering cases from XGBoost versus approximately 43 from TGN.

The threshold is configurable. An institution prioritising recall (catching as many laundering cases as possible, accepting more false positives) can lower the threshold. An institution prioritising precision (minimising analyst time wasted on false positives) can raise it. The calibrated thresholds reported in Chapter 4 maximise F1-score but are not prescriptive; each institution should calibrate against its own cost ratio of false negatives to false positives.

**5.3.3 Deployment Considerations**

Several practical considerations arise from the development and evaluation of these models.

**Chronological retraining.** TGN's per-node memory states are a function of transaction history. When the model is retrained on new data, memory states must be reinitialised from the beginning of the training period or carried forward from the previous training run. The former is simpler but discards accumulated history; the latter preserves history but requires careful handling to avoid stale memory states from outdated model weights. A practical approach is periodic full retraining (monthly or quarterly) with memory states computed from scratch over the full historical dataset, combined with daily inference using frozen model weights and continuously updating memory.

**Feature engineering in production.** The 28 edge features and 12 node features used in this study (detailed in Appendix A) were computed from raw transaction and account data. In a production setting, these features must be computed in real time or near-real time as transactions arrive. The log-transformed amount features and one-hot encoded categorical features are straightforward to compute; the cyclic time encodings (hour of day, day of week) require timestamp parsing. The node features (degree, volume, counterparty statistics) are aggregate statistics that must be recomputed or incrementally updated as new transactions arrive.

**Memory persistence.** For deployment, TGN's per-node memory states must persist between inference batches via a key-value store mapping account composite keys to memory vectors. At 64 dimensions (256 bytes per account at float32), total storage for 500,000 accounts is approximately 128 MB, negligible by modern infrastructure standards.

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

This study set out to answer how static and temporal graph neural network architectures compare to conventional machine learning for detecting money laundering in financial transaction networks. The answer, supported by a systematic three-tier evaluation on a standardised public benchmark, is that continuous-time temporal GNNs with per-node memory achieve the strongest detection performance, but temporal modelling is not automatically beneficial: snapshot-based approaches underperformed the static GCN, demonstrating that temporal information must be at the right granularity to add value. Money laundering is both relational and temporal; the detection tools built to counter it must address both dimensions.

---

# References

Alarab, I., & Prakoonwit, S. (2023). Graph-based LSTM for anti-money laundering: Experimenting with temporal graph convolutional network for bitcoin data. *Neural Processing Letters*, *55*(1), 689-707. https://doi.org/10.1007/s11063-022-10904-8

Altman, E., Blanuša, J., von Niederhäusern, L., Egressy, B., Anghel, A., & Atasu, K. (2023). Realistic synthetic financial transactions for anti-money laundering models. In *Advances in Neural Information Processing Systems 36 (NeurIPS 2023): Datasets and Benchmarks Track*. https://proceedings.neurips.cc/paper_files/paper/2023/hash/5f38404edff6f3f642d6fa5892479c42-Abstract-Datasets_and_Benchmarks.html

Breiman, L. (2001). Random forests. *Machine Learning*, *45*(1), 5-32. https://doi.org/10.1023/A:1010933404324

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 785-794). ACM. https://doi.org/10.1145/2939672.2939785

Chen, Z., Van Khoa, L. D., Teoh, E. N., Nazir, A., Karuppiah, E. K., & Lam, K. S. (2018). Machine learning techniques for anti-money laundering (AML) solutions in suspicious transaction detection: A review. *Knowledge and Information Systems*, *57*(2), 245-285. https://doi.org/10.1007/s10115-017-1144-z

Cheng, D., Zou, Y., Xiang, S., & Jiang, C. (2024). Graph neural networks for financial fraud detection: A review. *Frontiers of Computer Science*, *19*(5), Article 19505. https://doi.org/10.1007/s11704-024-40474-y

Cho, K., Van Merrienboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning phrase representations using RNN encoder-decoder for statistical machine translation. In *Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP)* (pp. 1724-1734). ACL. https://doi.org/10.3115/v1/D14-1179

Dou, Y., Liu, Z., Sun, L., Deng, Y., Peng, H., & Yu, P. S. (2020). Enhancing graph neural network-based fraud detectors against camouflaged fraudsters. In *Proceedings of the 29th ACM International Conference on Information and Knowledge Management (CIKM)* (pp. 315-324). ACM. https://doi.org/10.1145/3340531.3411903

FATF. (2023). *International standards on combating money laundering and the financing of terrorism and proliferation: The FATF recommendations*. Financial Action Task Force. https://www.fatf-gafi.org/en/publications/Fatfrecommendations/Fatf-recommendations.html

Fey, M., & Lenssen, J. E. (2019). Fast graph representation learning with PyTorch Geometric. In *ICLR 2019 Workshop on Representation Learning on Graphs and Manifolds*. https://arxiv.org/abs/1903.02428

Grover, A., & Leskovec, J. (2016). node2vec: Scalable feature learning for networks. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 855-864). ACM. https://doi.org/10.1145/2939672.2939754

Hamilton, W. L., Ying, R., & Leskovec, J. (2017). Inductive representation learning on large graphs. In *Advances in Neural Information Processing Systems 30 (NeurIPS 2017)* (pp. 1024-1034). https://papers.nips.cc/paper/2017/hash/5dd9db5e033da9c6fb5ba83c7a7ebea9-Abstract.html

He, H., & Garcia, E. A. (2009). Learning from imbalanced data. *IEEE Transactions on Knowledge and Data Engineering*, *21*(9), 1263-1284. https://doi.org/10.1109/TKDE.2008.239

Johannessen, F., & Jullum, M. (2025). Finding money launderers using heterogeneous graph neural networks. *Journal of Finance and Data Science*, *11*, Article 100175. https://doi.org/10.1016/j.jfds.2025.100175

Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. In *International Conference on Learning Representations (ICLR 2017)*. https://arxiv.org/abs/1609.02907

Levi, M. (2002). Money laundering and its regulation. *The Annals of the American Academy of Political and Social Science*, *582*(1), 181-194. https://doi.org/10.1177/000271620258200113

Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollar, P. (2017). Focal loss for dense object detection. In *Proceedings of the IEEE International Conference on Computer Vision (ICCV)* (pp. 2980-2988). IEEE. https://doi.org/10.1109/ICCV.2017.324

Oztas, B., Cetinkaya, D., Adedoyin, F., & Budka, M. (2023). SAML-D: A synthetic anti-money laundering dataset with controlled complexity. *Data in Brief*, *51*, 109692. https://doi.org/10.1016/j.dib.2023.109692

Pareja, A., Domeniconi, G., Chen, J., Ma, T., Suzumura, T., Kanezashi, H., Kaler, T., Schardl, T. B., & Leiserson, C. E. (2020). EvolveGCN: Evolving graph convolutional networks for dynamic graphs. In *Proceedings of the 34th AAAI Conference on Artificial Intelligence (AAAI 2020)* (pp. 5363-5370). AAAI Press. https://doi.org/10.1609/aaai.v34i04.5984

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, *12*, 2825-2830. https://www.jmlr.org/papers/volume12/pedregosa11a/pedregosa11a.pdf

Rossi, E., Chamberlain, B., Frasca, F., Eynard, D., Monti, F., & Bronstein, M. (2020). Temporal graph networks for deep learning on dynamic graphs. *arXiv preprint arXiv:2006.10637*. https://arxiv.org/abs/2006.10637

Sekaran, U., & Bougie, R. (2019). *Research methods for business: A skill-building approach* (8th ed.). Wiley.

Trivedi, R., Farajtabar, M., Biswal, P., & Zha, H. (2019). DyRep: Learning representations over dynamic graphs. In *International Conference on Learning Representations (ICLR 2019)*. https://openreview.net/forum?id=HyePrhR5KX

United Nations. (1988). *United Nations Convention against Illicit Traffic in Narcotic Drugs and Psychotropic Substances*. United Nations Treaty Series, 1582, 95. https://treaties.un.org/doc/Treaties/1990/11/19901101%2006-35%20AM/Ch_VI_19p.pdf

UNODC. (2011). *Estimating illicit financial flows resulting from drug trafficking and other transnational organized crimes*. United Nations Office on Drugs and Crime. https://www.unodc.org/documents/data-and-analysis/Studies/Illicit_financial_flows_2011_web.pdf

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention is all you need. In *Advances in Neural Information Processing Systems 30 (NeurIPS 2017)* (pp. 5998-6008). https://papers.nips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html

Veličković, P., Cucurull, G., Casanova, A., Romero, A., Lio, P., & Bengio, Y. (2018). Graph attention networks. In *International Conference on Learning Representations (ICLR 2018)*. https://arxiv.org/abs/1710.10903

Weber, M., Domeniconi, G., Chen, J., Weidele, D. K. I., Bellei, C., Robinson, T., & Leiserson, C. E. (2019). Anti-money laundering in bitcoin: Experimenting with graph convolutional networks for financial forensics. In *KDD 2019 Workshop on Anomaly Detection in Finance*. https://arxiv.org/abs/1908.02591

Xu, K., Hu, W., Leskovec, J., & Jegelka, S. (2019). How powerful are graph neural networks? In *International Conference on Learning Representations (ICLR 2019)*. https://arxiv.org/abs/1810.00826

Xu, D., Ruan, C., Korpeoglu, E., Kumar, S., & Achan, K. (2020). Inductive representation learning on temporal graphs. In *International Conference on Learning Representations (ICLR 2020)*. https://arxiv.org/abs/2002.07962

---

# Appendices

**Appendix A: Complete Feature Specification**

This appendix provides the exhaustive specification of all features used by the models in this study. Section 3.3.1 provides illustrative examples; this appendix provides the complete reference.

**A.1 Node Features (12 features)**

Node features are computed per account from the accounts file and aggregated transaction statistics. All count and amount features are log1p-transformed before standardisation. Categorical features are label-encoded then standardised.

**Table A.1: Complete node feature specification.**

| Index | Feature Name           | Type        | Source           | Computation                                                                                                |
| ----- | ---------------------- | ----------- | ---------------- | ---------------------------------------------------------------------------------------------------------- |
| 0     | bank_name              | Categorical | accounts.csv     | Label-encoded, then standardised (z-score)                                                                 |
| 1     | bank_id                | Categorical | accounts.csv     | Label-encoded, then standardised (z-score)                                                                 |
| 2     | entity_type            | Categorical | accounts.csv     | Extracted from Entity Name (e.g., "Corporation #33520" becomes "Corporation"), label-encoded, standardised |
| 3     | degree_out             | Numeric     | transactions.csv | Number of transactions sent by this account (log1p)                                                        |
| 4     | total_amount_out       | Numeric     | transactions.csv | Sum of amounts sent (log1p)                                                                                |
| 5     | avg_amount_out         | Numeric     | transactions.csv | Mean amount sent (log1p)                                                                                   |
| 6     | num_counterparties_out | Numeric     | transactions.csv | Number of unique receiving accounts (log1p)                                                                |
| 7     | degree_in              | Numeric     | transactions.csv | Number of transactions received by this account (log1p)                                                    |
| 8     | total_amount_in        | Numeric     | transactions.csv | Sum of amounts received (log1p)                                                                            |
| 9     | avg_amount_in          | Numeric     | transactions.csv | Mean amount received (log1p)                                                                               |
| 10    | num_counterparties_in  | Numeric     | transactions.csv | Number of unique sending accounts (log1p)                                                                  |
| 11    | degree_total           | Numeric     | transactions.csv | degree_out + degree_in (log1p)                                                                             |

All 12 features are standardised to zero mean and unit variance using a StandardScaler fitted on the training set only. Accounts with no transaction history receive zero values for all transaction statistic features after joining.

**A.2 Edge Features (28 features)**

Edge features are computed per transaction. Amount features are log1p-transformed. Cyclic time features are encoded as sine-cosine pairs. Categorical features are one-hot encoded. The first 6 features (amount_log1p, hour_sin, hour_cos, dow_sin, dow_cos, amount_paid_log1p) are standardised; the 22 one-hot features (7 payment format + 15 currency) are left unstandardised since they are bounded to {0, 1}.

**Table A.2: Complete edge feature specification.**

| Index | Feature Name      | Type         | Source           | Computation                                                                                                                                 |
| ----- | ----------------- | ------------ | ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| 0     | amount_log1p      | Numeric      | transactions.csv | log1p(Amount Received), standardised                                                                                                        |
| 1     | hour_sin          | Numeric      | transactions.csv | sin(2 * pi * hour / 24), standardised                                                                                                       |
| 2     | hour_cos          | Numeric      | transactions.csv | cos(2 * pi * hour / 24), standardised                                                                                                       |
| 3     | dow_sin           | Numeric      | transactions.csv | sin(2 * pi * day_of_week / 7), standardised                                                                                                 |
| 4     | dow_cos           | Numeric      | transactions.csv | cos(2 * pi * day_of_week / 7), standardised                                                                                                 |
| 5     | amount_paid_log1p | Numeric      | transactions.csv | log1p(Amount Paid), standardised                                                                                                            |
| 6-12  | pmt_{category}    | One-hot (7)  | transactions.csv | Payment Format one-hot: ACH, Cheque, Credit Card, Domestic Wire, International Wire, Cash, Unknown. Exactly one column = 1 per transaction. |
| 13-27 | cur_{code}        | One-hot (15) | transactions.csv | Currency one-hot: one column per currency code (USD, EUR, GBP, etc.). Exactly one column = 1 per transaction.                               |

**A.3 Feature Engineering Design Notes**

Features are computed from the training set only. The fitted encoders (LabelEncoder for categorical node features and edge payment/currency fields, StandardScaler for numeric features) are then applied to validation and test sets without refitting. This prevents data leakage from validation and test partitions into model training.

Log1p transformation is applied to all amount and count features because transaction amounts and degree distributions are heavily long-tailed: a small number of accounts send or receive orders of magnitude more transactions and larger amounts than the typical account. Without log transformation, these extreme values would dominate the standardised feature space.

Cyclic time encoding using sine-cosine pairs ensures that temporally adjacent moments have similar feature representations. Under a linear encoding, 23:59 and 00:01 would be separated by 23.98 units; under the cyclic encoding, they are separated by the Euclidean distance between (sin(23:59), cos(23:59)) and (sin(00:01), cos(00:01)), which is small. The same principle applies to day of week, where Monday and Sunday are neighbours on the 7-day circle.

---

**Appendix B: Reproducibility Guide**

This appendix provides the complete set of commands and configuration required to reproduce all experimental results reported in this thesis.

**B.1 Environment**

All experiments were conducted with the following software versions:

- Python 3.11
- PyTorch 2.x
- PyTorch Geometric 2.5.x
- scikit-learn 1.x
- XGBoost 2.x
- NumPy 1.24+
- Pandas 2.0+
- Matplotlib 3.7+
- Seaborn 0.12+

The complete dependency list with exact version numbers is specified in `requirements.txt` at the project root. Install with:

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
python experiments/run_gnn.py --variant HI-Small --model gcn --seed 42 --epochs 200
python experiments/run_gnn.py --variant HI-Small --model gat --seed 42 --epochs 200 --heads 1
python experiments/run_gnn.py --variant HI-Small --model sage --seed 42 --epochs 200 --aggregator mean
```

Or run all three sequentially:

```
python experiments/run_gnn.py --variant HI-Small --model all --seed 42 --epochs 200
```

**Snapshot temporal GNNs (Tier 3a):**

```
python experiments/run_temporal.py --variant HI-Small --model temporal_gcn --seed 42 --epochs 200
python experiments/run_temporal.py --variant HI-Small --model evolve_gcn_h --seed 42 --epochs 200 --rank 2
```

**Continuous-time TGN (Tier 3b):**

```
python experiments/run_tgn.py --variant HI-Small --epochs 100 --lr 0.003 --pos_weight_mult 0.01 --grad_clip 0 --seed 42
```

The `--grad_clip 0` argument is critical: with `pos_weight=12.4`, gradient clipping destroys the minority class learning signal (see Section 3.4.4 for the full explanation).

**B.4 Data Splits**

All splits are chronological (time-based):

1. Transactions are sorted by their Unix timestamp.
2. The earliest 70% of edges are assigned to training.
3. The next 15% are assigned to validation.
4. The latest 15% are assigned to testing.

For snapshot temporal models, the 12 quantile-based windows are chronologically ordered: windows 0-7 = training, window 8 = validation, windows 9-11 = testing.

The chronological split ensures that models are trained on past transactions and evaluated on future transactions, mirroring deployment conditions.

**B.5 Expected Output**

Running all reproduction commands produces the following expected metrics (minor variation may occur if library versions differ):

| Model               | AUC-ROC | AUC-PR | F1     |
| ------------------- | ------- | ------ | ------ |
| Logistic Regression | 0.9378  | 0.0376 | 0.0267 |
| Random Forest       | 0.8603  | 0.0619 | 0.0070 |
| XGBoost             | 0.9381  | 0.1511 | 0.0514 |
| GCN                 | 0.9705  | 0.1882 | 0.2513 |
| GAT                 | 0.9581  | 0.0958 | 0.0979 |
| GraphSAGE           | 0.9459  | 0.0420 | 0.0946 |
| TemporalGCN         | 0.9570  | 0.0637 | 0.1343 |
| EvolveGCN-H         | 0.8972  | 0.0275 | 0.0631 |
| TGN                 | 0.9684  | 0.3195 | 0.3527 |

---

**Appendix C: Generative AI Usage Declaration**

This appendix declares the use of generative AI tools in the preparation of this thesis, in accordance with the Amsterdam University of Applied Sciences Master Project module guide requirements.

**Tool used:** ChatGPT (OpenAI).

**Nature of use:**

- Assistance with implementing and debugging Python code for GNN architectures, model training loops, based on my own architectural design decisions and research methodology. Specific examples include debugging the TGN data leakage bug (train/eval memory state mismatch), resolving the PyG TGNMemory dtype incompatibility, and implementing the EMA-based memory module.
- Reviewing chapter drafts I had written for clarity, consistency, and grammatical correctness.

**Nature of author contribution:**

- All experimental design, implementation, and execution was performed by the author.
- All research questions, methodological decisions, and conclusions were formulated by the author.
- All literature review, citation selection, and theoretical framework development was performed by the author.
- The author wrote all chapter drafts, provided all substantive content (experimental results, architectural descriptions, methodological reasoning), directed the revision process, reviewed all AI-suggested edits for accuracy and appropriateness, and takes full responsibility for the final content of this thesis.

**Verification:** All factual claims, numerical results, and citations in this thesis have been verified by the author against primary sources (experimental logs, published papers, and the assessment rubric).

---

**Appendix D: Full Results Tables**

This appendix reproduces the complete results from Chapter 4 for reference. The data is identical to that presented in the chapter body and in the project's `docs/RESULTS.md` file.

**Table D.1: Conventional ML baseline results (random 70/15/15 split, threshold 0.50).**

| Model               | AUC-ROC | AUC-PR | Precision | Recall | F1     |
| ------------------- | ------- | ------ | --------- | ------ | ------ |
| Logistic Regression | 0.9378  | 0.0376 | 0.0135    | 0.9295 | 0.0267 |
| Random Forest       | 0.8603  | 0.0619 | 0.0035    | 0.9148 | 0.0070 |
| XGBoost             | 0.9381  | 0.1511 | 0.0265    | 0.8610 | 0.0514 |

**Table D.2: Static GNN results (random 70/15/15 split, calibrated thresholds).**

| Model        | Params | AUC-ROC | AUC-PR | Precision | Recall | F1     | Thresh |
| ------------ | ------ | ------- | ------ | --------- | ------ | ------ | ------ |
| GCN          | 63,489 | 0.9705  | 0.1882 | 0.1846    | 0.3933 | 0.2513 | 0.7029 |
| GAT (1 head) | 64,001 | 0.9581  | 0.0958 | 0.0539    | 0.5317 | 0.0979 | 0.5544 |
| GraphSAGE    | 81,409 | 0.9459  | 0.0420 | 0.0563    | 0.2953 | 0.0946 | 0.4852 |

**Table D.3: Temporal GNN results (chronological split, calibrated thresholds).**

| Model                | Params  | AUC-ROC | AUC-PR | Precision | Recall | F1     | Thresh |
| -------------------- | ------- | ------- | ------ | --------- | ------ | ------ | ------ |
| TemporalGCN          | 161,793 | 0.9570  | 0.0637 | 0.1177    | 0.1563 | 0.1343 | 0.7326 |
| EvolveGCN-H          | 578,369 | 0.8972  | 0.0275 | 0.0465    | 0.0982 | 0.0631 | 0.7029 |
| EvolveGCN-H (rank=8) | 33M     | N/A     | N/A    | N/A       | N/A    | N/A    | N/A    |
| TGN                  | 119,000 | 0.9684  | 0.3195 | 0.4257    | 0.3011 | 0.3527 | 0.4159 |

**Table D.4: TGN per-slice performance (12 equal chronological test set slices, threshold 0.50).**

| Slice             | AUC-ROC | AUC-PR | Precision | Recall |
| ----------------- | ------- | ------ | --------- | ------ |
| 0 (earliest test) | 0.9205  | 0.0502 | 0.0000    | 0.0000 |
| 3                 | 0.9280  | 0.0853 | 0.6000    | 0.0405 |
| 6                 | 0.9714  | 0.0712 | 0.0000    | 0.0000 |
| 9                 | 0.9591  | 0.0769 | 0.1294    | 0.1028 |
| 10                | 0.9563  | 0.1875 | 0.1553    | 0.3617 |
| 11 (latest test)  | 0.9732  | 0.4518 | 0.5749    | 0.3300 |

---

**Appendix E: Hyperparameter Configurations**

This appendix presents the hyperparameter configurations used for all neural network models (Table 3.1 in the main text). All experiments used these settings unless otherwise noted in the methodology chapter.

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
| Epochs          | 200         | 200           | 200           | 100    |
| Patience        | 25          | 25            | 25            | 25     |
| Batch size      | Full graph  | Full snapshot | Full snapshot | 2048   |
| Memory dim      | N/A         | N/A           | N/A           | 64     |
| Time dim        | N/A         | N/A           | N/A           | 8      |
| EMA beta        | N/A         | N/A           | N/A           | 0.85   |
| Rank            | N/A         | N/A           | 2             | N/A    |
| GAT heads       | 1           | N/A           | N/A           | N/A    |
| SAGE aggregator | mean        | N/A           | N/A           | N/A    |

---

**Appendix F: Training and Validation Results**

This appendix provides the complete training, validation, and test set metrics for the conventional ML baselines (Table 4.1 in the main text). The training set metrics indicate how well each model fits the training data; the validation set metrics were used for early stopping and threshold calibration. The large gap between training and validation performance for Random Forest indicates overfitting despite the depth constraint.

**Table F.1: Conventional ML baseline results across all splits (threshold 0.50).**

| Model               | Split | AUC-ROC | AUC-PR | Precision | Recall | F1     |
| ------------------- | ----- | ------- | ------ | --------- | ------ | ------ |
| Logistic Regression | train | 0.9007  | 0.0102 | 0.0060    | 0.8344 | 0.0118 |
| Logistic Regression | val   | 0.9022  | 0.0115 | 0.0071    | 0.8566 | 0.0141 |
| Logistic Regression | test  | 0.9378  | 0.0376 | 0.0135    | 0.9295 | 0.0267 |
| Random Forest       | train | 0.9741  | 0.0615 | 0.0018    | 1.0000 | 0.0035 |
| Random Forest       | val   | 0.8087  | 0.0187 | 0.0016    | 0.8789 | 0.0031 |
| Random Forest       | test  | 0.8603  | 0.0619 | 0.0035    | 0.9148 | 0.0070 |
| XGBoost             | train | 0.9838  | 0.0848 | 0.0135    | 0.9167 | 0.0265 |
| XGBoost             | val   | 0.8891  | 0.0390 | 0.0138    | 0.7382 | 0.0272 |
| XGBoost             | test  | 0.9381  | 0.1511 | 0.0265    | 0.8610 | 0.0514 |
