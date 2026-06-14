# Graph Neural Networks Applied to Money Laundering Detection

## Master Thesis

**Draft: Chapters 1-3**

---

**Student:** Shadab Gada (500981772)

**Supervisor:** Kees van Montfort, PhD

**Second Assessor:** Debarati Bhaumik, PhD

**Program:** Master Digital Driven Business

**Institution:** Amsterdam University of Applied Sciences

**Date:** May 2026

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
- 2.4 Temporal Graph Neural Networks
- 2.5 Evaluation Under Class Imbalance
- 2.6 Research Gap Synthesis

**Chapter 3: Research Methodology and Tool Development**

- 3.1 Research Design Overview
- 3.2 Dataset: IBM AML HI-Small
- 3.3 Data Engineering and Graph Construction (SQ1)
- 3.4 Model Architectures (SQ2 and SQ3)
- 3.5 Training and Evaluation Protocol (SQ2 and SQ3)
- 3.6 Ethical Considerations and Reproducibility

---


---

# Chapter 1: Introduction

**1.1 Background: Money Laundering as a Global Challenge**

Money laundering, the process through which illegally obtained funds are moved through legitimate financial channels to obscure their criminal origin, represents one of the most persistent threats to global economic stability and security (United Nations, 1988). The United Nations Office on Drugs and Crime estimates that between 2% and 5% of global GDP is laundered annually, financing organised crime, drug trafficking, terrorism, and human trafficking (UNODC, 2011). In absolute terms, this amounts to between $800 billion and $2 trillion USD per year, making anti-money laundering (AML) enforcement a critical regulatory and operational priority for financial institutions worldwide (UNODC, 2011).

Financial institutions are legally obligated under frameworks such as the European Union's Anti-Money Laundering Directives and the Financial Action Task Force (FATF) recommendations to detect and report suspicious activity (FATF, 2023). These obligations carry substantial operational costs: institutions collectively spend tens of billions of dollars annually on compliance infrastructure, yet the effectiveness of these systems remains limited. The scale of the problem, combined with the regulatory imperative and the high cost of compliance failures, creates an acute need for more accurate and efficient detection methods.

**1.2 Current AML Detection Approaches and Their Limitations**

Traditional rule-based AML systems apply fixed thresholds and heuristics to flag suspicious transactions: transactions above a certain amount, transfers to high-risk jurisdictions, or activity patterns matching predefined typologies (FATF, 2023). While these systems are interpretable and form the backbone of current compliance operations, they suffer from three fundamental limitations. First, they are rigid: rules must be explicitly defined and cannot adapt to evolving laundering tactics without manual intervention. Second, they generate extremely high false positive rates: industry reports suggest that over 95% of AML alerts are false positives, creating severe alert fatigue among compliance analysts and diverting resources from genuinely suspicious cases (Chen et al., 2018). Third, they evaluate each transaction in isolation, blind to the relational context that reveals sophisticated laundering schemes.

Conventional machine learning approaches, including logistic regression and tree-based models such as random forests and gradient boosting, have been applied to improve upon rule-based systems (Chen et al., 2018; Altman et al., 2023). These methods analyse individual transaction features such as amount, currency, and payment format, and improve detection rates to an extent. However, they share the third limitation of rule-based systems: they fundamentally fail to capture the relational structure of money laundering. Sophisticated laundering schemes, such as layering through chains of intermediary accounts or structuring (smurfing) across multiple accounts, are only detectable when the broader network of transactions is considered (Levi, 2002). A single transaction may appear benign; its position within a network of suspicious activity reveals its true nature.

**1.3 The Graph-Structured Nature of Financial Transactions**

Financial transaction data is inherently graph-structured. Accounts are entities, transactions are directed interactions between entities, and the patterns that distinguish legitimate activity from laundering behaviour emerge from the topology and dynamics of this interaction network. A laundering operation that distributes funds across ten accounts via fifty transactions is not suspicious at the individual transaction level; it is the collective pattern, the fan-out structure, the timing and sequencing of transfers, that constitutes the laundering signal.

This observation points toward a class of machine learning models specifically designed for graph-structured data: Graph Neural Networks (GNNs). GNNs propagate and aggregate information across nodes and edges, learning representations that capture both local features and the broader relational context of each entity in the network (Kipf & Welling, 2017). Applied to financial transaction networks, where accounts are nodes and transactions are edges, GNNs can learn from the entire web of financial interactions, capturing the relational and contextual patterns that distinguish legitimate activity from laundering behaviour (Johannessen & Jullum, 2025; Weber et al., 2019).

However, money laundering is not only a relational phenomenon. It is also a temporal one. Layering chains and structuring schemes unfold across time-ordered transaction sequences. The order in which transactions occur, the rhythm of account activity, and the evolution of behavioural patterns over time all carry signal that a static graph representation cannot capture. This temporal dimension motivates the exploration of temporal GNN architectures that explicitly model the dynamics of financial transaction networks.

**1.4 Problem Statement**

Despite substantial investment in AML compliance infrastructure and a growing body of research applying machine learning to financial crime detection, a significant gap persists in the empirical evaluation of detection architectures. The IBM Transactions for Anti-Money Laundering dataset (Altman et al., 2023), a large-scale synthetic dataset specifically designed for AML research and published at NeurIPS 2023, was released with baseline results using static GNNs only: GCN, GAT, and GraphSAGE. The dataset paper did not evaluate any temporal GNN architectures. Meanwhile, published temporal GNN work in AML (Alarab & Prakoonwit, 2023) used the Elliptic Bitcoin dataset, which represents a fundamentally different transaction domain. No study has produced a systematic comparative evaluation of temporal and static GNN architectures alongside conventional supervised classifiers within a unified framework on a standardised public banking AML benchmark (Cheng et al., 2024; Johannessen & Jullum, 2025).

Two paradigms exist for capturing temporal dynamics in graph learning: snapshot-based architectures such as TemporalGCN and EvolveGCN-H (Pareja et al., 2020), which partition transactions into time windows and evolve representations across windows, and continuous-time architectures such as the Temporal Graph Network (TGN; Rossi et al., 2020), which processes each transaction individually with its exact timestamp. Neither paradigm has been evaluated on the IBM AML benchmark.

This gap has practical consequences. Without a rigorous, like-for-like comparison spanning all three tiers (conventional machine learning, static GNNs, snapshot-based temporal GNNs, and continuous-time temporal GNNs), AML compliance practitioners lack the empirical evidence needed to make informed decisions about which class of model to invest in, what performance trade-offs to expect, and under what conditions temporal modelling adds sufficient value to justify its additional complexity. The research problem is therefore both academic (an unaddressed gap in the comparative evaluation literature) and practical (insufficient evidence for practitioner model selection in AML compliance contexts).

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
- Evidence-based guidance on model selection across three tiers of detection approaches, grounded in a rigorous like-for-like comparison on the same dataset.
- Quantified analysis of the precision-recall trade-offs that operational compliance teams face when deploying graph-based detection tools, including the relationship between threshold selection and false positive burden.
- A documented, reproducible reference implementation that compliance analytics teams can adapt and evaluate against their own institutional data and requirements.

**1.8 Report Structure**

The remainder of this report is structured as follows.

**Chapter 2 (Theoretical Framework)** synthesises the relevant literature across four domains: money laundering typologies and the regulatory context, conventional machine learning for AML, static GNN architectures and their application to financial crime detection, and temporal GNN architectures spanning both snapshot-based and continuous-time approaches. The chapter concludes with a synthesis of the research gap and the theoretical basis for the architectural choices evaluated in this study.

**Chapter 3 (Research Methodology and Tool Development)** describes the dataset, the data engineering and graph construction pipeline, the implementation of each model architecture, and the training and evaluation protocol. Design decisions are explicitly justified with reference to the literature discussed in Chapter 2, and the chapter includes a comparison of alternative design options where relevant.

**Chapter 4 (Results, Analyses and Tool Performance)** presents the empirical findings. Results are reported for each model tier, followed by a cross-model comparison, a temporal generalisation analysis examining TGN performance across time slices, and an assessment of tool scalability.

**Chapter 5 (Discussion, Recommendations and Conclusions)** answers each research sub-question and the main research question, discusses the practical implications of the findings for AML compliance practice, presents the study's theoretical contributions and novelty claims, acknowledges limitations, proposes directions for future research, and provides concluding remarks.

The appendices document the tool's requirements and development process, provide a complete reproducibility guide, and include the required declaration of generative AI usage.


---

# Chapter 2: Theoretical Framework

**2.1 Money Laundering Typologies and Regulatory Context**

Money laundering is the process of disguising the criminal origin of funds by moving them through legitimate financial channels. The Financial Action Task Force (FATF), the global standard-setting body for anti-money laundering regulation, identifies three stages in the laundering process: placement, where illicit funds first enter the financial system; layering, where funds are moved through sequences of transactions to obscure their origin; and integration, where laundered funds re-enter the legitimate economy (FATF, 2023). These stages are not merely descriptive categories. They correspond to distinct behavioural patterns in transaction networks, each of which leaves a different structural signature in the graph of financial interactions.

Placement typically involves depositing large sums into accounts, creating transaction patterns with high individual amounts but relatively simple counterparty structures. Layering is the most structurally complex stage: funds are routed through chains of intermediary accounts, split into smaller amounts (structuring or smurfing), and distributed across multiple destinations (fan-out) before being reaggregated (fan-in). These patterns produce distinctive topological signatures: unusually long transaction chains, accounts with high out-degree relative to in-degree, and clusters of accounts with dense internal connectivity but weak external ties. Integration involves transactions with legitimate businesses, often characterised by amounts and frequencies that blend with normal commercial activity. The regulatory obligation to detect these patterns falls on financial institutions under frameworks such as the European Union's Anti-Money Laundering Directives and the FATF Recommendations, which require institutions to implement systems capable of identifying and reporting suspicious transactions (FATF, 2023).

The academic literature on money laundering has long identified the relational nature of laundering behaviour. Levi (2002) analysed money laundering as a criminological phenomenon, documenting how illicit funds traverse financial systems through networks of accounts and intermediaries, and argued that the practical limitations of detection arise precisely because individual transactions appear legitimate when examined in isolation. The analytical implication is clear: effective detection requires examining transactions in their relational context, not as independent observations.

Two additional considerations are relevant to this study. First, laundering patterns evolve over time in response to changes in detection methods and regulatory enforcement. A static detection system trained on historical patterns will degrade as launderers adapt their techniques, creating a structural need for models that can capture temporal dynamics. Second, the FATF-documented typologies (structuring, layering, fan-in/fan-out) are explicitly encoded in the IBM AML dataset used in this research (Altman et al., 2023), meaning the dataset's laundering patterns reflect real-world regulatory knowledge rather than arbitrary simulation choices. This connection between regulatory typologies and dataset design is methodologically significant: it means a model that learns to detect these patterns in the dataset is learning to detect patterns that the global regulatory framework identifies as suspicious.

**2.2 Conventional Machine Learning for AML Detection**

Conventional machine learning approaches to AML detection treat each transaction as an independent feature vector and apply supervised classification methods to distinguish laundering from legitimate activity. Chen et al. (2018) provided a comprehensive review of machine learning techniques applied to suspicious transaction detection, covering logistic regression, decision trees, support vector machines, and ensemble methods. Their review identified two persistent limitations: first, the extreme class imbalance inherent in AML data, where laundering transactions constitute a tiny fraction of total volume, makes standard classifiers prone to high false positive rates; second, treating transactions as independent observations discards the relational structure that characterises laundering behaviour.

Logistic regression serves as the simplest baseline, modelling the log-odds of a transaction being suspicious as a linear function of its features. Its interpretability is an advantage in compliance contexts where regulatory requirements demand explainable decisions, but its linear decision boundary cannot capture the nonlinear interactions that characterise complex laundering schemes. Random forest classifiers (Breiman, 2001) address this by ensembling multiple decision trees trained on random subsets of features and samples, producing nonlinear decision boundaries while maintaining reasonable interpretability through feature importance scores. XGBoost (Chen & Guestrin, 2016) extends gradient boosting with regularization and optimised computation, and has become a standard benchmark in tabular classification tasks across domains including financial crime detection.

Several studies have applied these methods to AML detection with varying success. Chen et al. (2018) reported that ensemble methods outperformed linear classifiers on synthetic AML data, but noted that all tabular methods suffered from the same structural limitation: they cannot model relationships between transactions. A model can learn that transactions above a certain amount are more likely to be suspicious, but it cannot learn that a transaction is suspicious because it is the third in a chain of five small transfers between the same two accounts. This limitation is not an implementation detail; it is a fundamental consequence of the independence assumption underlying tabular machine learning. The next section discusses model architectures that explicitly relax this assumption.

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

Two structural concerns arise with the snapshot-based paradigm when applied to AML detection. First, the granularity of snapshot partitioning determines the temporal resolution of the model. With N snapshots, the model can observe at most N state transitions per node. Laundering operations such as structuring, which involves splitting a large sum into many small transactions executed in rapid succession, may unfold entirely within a single snapshot window. If the snapshot granularity is too coarse, these transaction-level patterns are invisible to the model. Second, both TemporalGCN and EvolveGCN process snapshots in fixed chronological order, meaning the model sees each edge exactly once in the context of its assigned snapshot. There is no mechanism to revisit a specific transaction in light of information from later snapshots, even though the significance of a transaction often becomes apparent only retrospectively.

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

The research design comprises five stages. First, a structured literature study established the theoretical foundation for GNN-based AML detection, identified the appropriate architectures, and confirmed the research gap motivating the comparative evaluation. Second, the IBM AML HI-Small dataset was selected through a systematic comparison against an alternative candidate and subjected to data engineering to construct graph representations suitable for both static and temporal GNN analysis. Third, seven model architectures spanning three tiers were implemented: three conventional supervised classifiers (Logistic Regression, Random Forest, XGBoost), three static GNNs (GCN, GAT, GraphSAGE), and three temporal GNNs (TemporalGCN, EvolveGCN-H, TGN). Fourth, all models were trained and evaluated under identical experimental conditions using metrics appropriate for heavily class-imbalanced data. Fifth, the empirical findings were analysed comparatively and translated into practitioner guidance.

The research design intentionally spans three tiers, rather than comparing GNN variants alone, to isolate the contribution of graph structure and temporal modelling to detection performance. Tier 1 (conventional ML) establishes the performance achievable without access to graph structure. Tier 2 (static GNNs) measures the gain from relational modelling. Tier 3 (temporal GNNs) measures the additional gain from temporal modelling and compares snapshot-based against continuous-time approaches. This design allows the study to answer not only which model performs best, but why.

During implementation, the scope of the temporal modelling tier expanded beyond the single architecture (EvolveGCN) specified in the research plan. Preliminary experiments revealed that snapshot-based temporal models underperformed the static GCN, which prompted the addition of TemporalGCN to verify whether the limitation was specific to EvolveGCN's weight-space evolution or inherent to the snapshot paradigm, and subsequently the addition of continuous-time TGN to test whether finer temporal granularity could overcome the limitation. This expansion was a data-driven methodological decision, grounded in empirical observations made during the research process. It is documented here transparently as part of the methodological narrative.

**3.2 Dataset: IBM AML HI-Small**

**3.2.1 Dataset Selection and Justification**

The dataset used in this study is the IBM Transactions for Anti-Money Laundering dataset (Altman et al., 2023), publicly available on Kaggle (https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml). The IBM AML dataset was chosen over the main alternative, the Synthetic AML Dataset (SAML-D; Oztas et al., 2023), for two reasons. First, the IBM AML dataset is structured natively as a graph: a dedicated accounts file defines each unique account as a persistent entity, and a transactions file captures directed interactions between accounts. This maps directly onto the node-and-edge representation required for GNN-based analysis. SAML-D, by contrast, is a flat tabular dataset without an explicit account-level structure, making the construction of stable node identities (a prerequisite for temporal GNNs) a non-trivial and ambiguous preprocessing step. Second, the IBM AML dataset's laundering patterns are derived from FATF-documented AML typologies, including structuring, layering, and fan-in/fan-out schemes (FATF, 2023; Altman et al., 2023), ensuring the synthetic patterns reflect real-world regulatory knowledge. The dataset was published at NeurIPS 2023 specifically as a public benchmark for GNN-based AML research (Altman et al., 2023).

The dataset is available in four variants (HI/LI combined with Small/Medium). This study uses the HI-Small variant (518,581 accounts, 5,078,345 transactions, 5,177 laundering, 0.102% prevalence). The HI variants contain a higher laundering ratio, providing more positive cases for training. The Small variant was chosen for computational feasibility: all models were trained on CPU, and the Medium variants (tens of millions of transactions) would have made full training runs for all seven architectures infeasible within the project timeline. The four variants share an identical data-generating process, so architectural findings from HI-Small are expected to generalise, though empirical verification on larger variants is noted as future work.

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

1. Bank and entity identifiers: the account's bank name and bank ID are label-encoded (each unique bank assigned an integer, then standardized to zero mean and unit variance), and the entity type is extracted from the entity name field (for example, "Corporation #33520" becomes "Corporation") and label-encoded. This captures whether an account belongs to a corporation, an individual, or another entity category.
2. Transaction statistics: ten aggregated statistics computed from the account's transaction history within the training set, including out-degree and in-degree (number of transactions sent and received), total and average amounts sent and received, and number of unique counterparties. All count and amount features are log1p-transformed to compress their long-tailed distributions.
3. All features are standardized (z-scored) so that zero represents the mean across all accounts.

To make this concrete, consider an account with the following profile after feature engineering: high out-degree (+1.89, roughly 47 outgoing transactions, well above the mean), low in-degree (-0.22, roughly 3 incoming transactions), high total amount sent (+0.67, approximately $234,000), and low amount received (-0.15, approximately $8,200). This account sends far more money than it receives, to many more counterparties than it receives from: a fan-out pattern characteristic of structuring behaviour. The node features encode this behavioural signature without the model needing to traverse the graph.

**Edge features.** Twenty-eight edge-level features were constructed for each transaction:

1. Amount: the log1p-transformed amount received and amount paid (2 features). Log transformation compresses the long-tailed amount distribution, preventing a small number of very large transactions from dominating the feature space.
2. Cyclic time: four features encoding the hour of day and day of week as sine and cosine pairs. Rather than representing 14:30 as the scalar 14.5 (where 23:59 and 00:01 appear 23 hours apart), the sine and cosine of (2 * pi * hour / 24) place all times on a circle where adjacent moments are always close. The same principle applies to day of week: Monday and Sunday are neighbours on the 7-day circle, which a linear encoding would not capture.
3. Payment format: seven one-hot columns, one per category. A transaction paid via ACH produces the column pattern [0, 1, 0, 0, 0, 0, 0]; a cheque produces [0, 0, 1, 0, 0, 0, 0]; a domestic wire produces [0, 0, 0, 0, 1, 0, 0]. Exactly one column is 1 for each transaction; all others are 0.
4. Currency: fifteen one-hot columns following the same principle, one per currency code. A USD transaction sets the USD column to 1; a EUR transaction sets the EUR column to 1.

The seven payment format columns and fifteen currency columns are left unstandardized (since one-hot values are already bounded to {0, 1}), while the amount and cyclic time features are standardized to zero mean and unit variance. This mixed encoding strategy preserves the interpretability of categorical features while normalising the scale of continuous features. The complete list of all 12 node features and 28 edge features with their types and computation methods is provided in Appendix A.

**Comparison with alternatives.** An alternative feature engineering approach would have been to use learned node embeddings (for example, Node2Vec; Grover & Leskovec, 2016) rather than hand-crafted features. The advantage of learned embeddings is that they can capture structural properties of the graph that hand-crafted features might miss, such as community membership and higher-order neighbourhood patterns. The disadvantage is that they require a separate pretraining stage, add computational overhead, and produce features that are less interpretable. Hand-crafted features were selected because they are directly interpretable, grounded in domain knowledge about what distinguishes laundering accounts (high counterparty count, unusual temporal patterns, transaction volume extremes), and computationally lightweight. The 28-dimensional edge feature vector and 12-dimensional node feature vector are compact enough to keep model parameter counts manageable while providing sufficient signal for the classification task.

**3.3.2 Graph Construction for Static and Temporal Models (SQ1)**

Two graph construction strategies were employed, corresponding to the static and temporal modelling paradigms.

**Static graph construction.** For static GNNs and conventional baselines, a single directed graph was constructed in which each unique account identifier maps to a node and each transaction maps to a directed edge from the originating account to the destination account. The graph was built using PyTorch Geometric (PyG; Fey & Lenssen, 2019). Edge indices, node feature matrices, edge feature matrices, and edge labels were assembled into a PyG Data object. Edge directions were preserved to capture the inherently directional nature of financial transactions: a transfer from account A to account B is structurally and semantically different from a transfer from B to A.

**Temporal snapshot construction.** For snapshot-based temporal GNNs (TemporalGCN and EvolveGCN-H), the transaction timeline was divided into 12 windows using a quantile-based strategy: window boundaries were placed such that each window contains approximately the same number of transactions. This strategy was chosen over fixed-width (equal time duration) windows because transaction density in the dataset is heavily skewed, with some periods containing orders of magnitude more transactions than others. Fixed-width windows would produce snapshots with highly variable edge counts, causing some snapshots to be too sparse for meaningful graph convolution and others to be too dense for efficient computation. Quantile-based windows ensure that each snapshot has sufficient and comparable edge density, which is important for stable GNN training across snapshots.

The 12-window granularity was chosen to balance temporal resolution against per-snapshot edge density. With approximately 5 million transactions total, each snapshot contains roughly 420,000 transactions, providing adequate density for GCN operations. A larger number of snapshots (for example, 24 or 48) would increase temporal resolution but reduce per-snapshot edge counts and increase training time linearly with the number of snapshots. The sensitivity of model performance to snapshot granularity was not systematically investigated, which is noted as a limitation.

**Continuous-time data construction.** For TGN, the temporal data builder processes transactions in strict chronological order without binning into windows. Each edge retains its individual timestamp as a continuous value (Unix seconds). The data is chronologically sorted and divided into training (70% of edges, earliest in time), validation (15%), and test (15%, latest in time) partitions by index. This preserves the natural temporal ordering: the model is trained on past transactions and evaluated on future ones.

**Design justification across paradigms.** The use of three different graph construction strategies (static, snapshot, continuous-time) is not an inconsistency but a reflection of the different modelling paradigms. Static GNNs require a single graph and gain no benefit from temporal information in the data structure. Snapshot temporal GNNs require a sequence of static graphs and benefit from temporal binning. TGN requires individual timestamps and would be degraded by binning. Using the appropriate data representation for each paradigm ensures that each model is evaluated under the conditions for which it was designed, enabling a fair comparison of the paradigms themselves rather than of suboptimal instantiations.

**3.3.3 Chronological Data Splitting**

All models were evaluated using a chronological (time-based) data split. For static models, transactions were sorted by timestamp and the earliest 70% were assigned to training, the next 15% to validation, and the latest 15% to testing. For snapshot temporal models, the 12 snapshots were chronologically ordered and assigned: snapshots 0 through 7 to training, snapshot 8 to validation, and snapshots 9 through 11 to testing. For TGN, the continuous temporal edge stream was partitioned at the same 70/15/15 ratios by edge index after chronological sorting.

This chronological split strategy has a specific advantage over random shuffling: it evaluates models under deployment-realistic conditions. In a production AML system, models are trained on historical data and must detect laundering in future transactions. Random splits, which mix past and future edges across train and test sets, introduce a subtle form of data leakage: the model sees edges from the future during training and edges from the past during testing, inflating performance estimates relative to real-world deployment conditions. Several published AML GNN studies have used random splits (Weber et al., 2019; Altman et al., 2023). This study's use of chronological splits provides a more honest and deployment-relevant performance estimate, though it also makes the evaluation task harder, a point discussed in the cross-model comparison in Chapter 4.

A consequence of chronological splitting is that the class distribution varies across partitions, since the laundering ratio is not constant over time. In the IBM AML HI-Small dataset, the laundering ratio increases from approximately 0.01% in the earliest time window to 0.30% in the latest. The chronological split means that the test set has a higher laundering prevalence than the training set, which is both realistic (laundering patterns may intensify over time in a real system) and challenging (the model is evaluated on a distribution that differs from its training distribution). The pos_weight for loss computation was computed from the training set only, consistent with the principle that no test-set information may influence model training.

**3.4 Model Architectures (SQ2 and SQ3)**

This section addresses SQ2 by describing the implementation of each model architecture and justifying the design choices with reference to the theoretical framework established in Chapter 2.

**3.4.1 Conventional ML Baselines**

Three supervised classifiers were implemented as baselines that operate on flat feature vectors without access to graph structure: Logistic Regression, Random Forest, and XGBoost. These models were selected to represent a progression of complexity and to establish the performance floor against which GNN-based models are compared, directly addressing SQ3.

**Logistic Regression** was implemented using scikit-learn (Pedregosa et al., 2011) with L2 regularization and the liblinear solver. Class imbalance was addressed via class_weight="balanced", which automatically weights the minority class inversely proportional to its frequency in the training set. No sample_weight was applied, avoiding the double-weighting issue in which class_weight and sample_weight simultaneously scale the minority class loss, effectively squaring the intended penalty.

**Random Forest** (Breiman, 2001) was implemented with 100 estimators, a maximum depth of 10, and class_weight="balanced". The depth limit was applied to mitigate overfitting to the majority class, which preliminary experiments showed was severe without regularization: unconstrained trees would grow deep enough to memorise individual legitimate transactions, producing near-perfect training scores but generalising poorly.

**XGBoost** (Chen & Guestrin, 2016) was implemented with default hyperparameters and early stopping configured to monitor validation set log loss with a patience of 20 rounds. Early stopping on the validation set, rather than on training data as in an earlier implementation, is methodologically important: monitoring training data provides no signal about generalisation and can lead to overfitting.

All three baselines receive the identical 28-dimensional edge feature vectors used by the GNN models. The key difference is that the baselines treat each edge independently, while the GNNs additionally receive node features and the graph adjacency structure. Any performance difference between the baselines and the GNNs can therefore be attributed to the graph structural information, since the edge-level input features are held constant.

**3.4.2 Static GNNs**

The three static GNN architectures described in Section 2.3.1 were implemented using PyTorch Geometric (Fey & Lenssen, 2019). All three share a common architectural template: an edge classification model consisting of node encoding layers, edge feature projection, and a classifier head.

**GCN (Kipf & Welling, 2017).** The implementation uses two GCN convolutional layers with hidden dimension 128 and ReLU activation. Each GCN layer applies the symmetric normalised graph Laplacian convolution to propagate node features across edges. After convolution, the final-layer node embeddings for the source and destination nodes of each edge are concatenated with the projected edge features and passed through a two-layer MLP classifier with dropout (p=0.3) to produce a scalar logit per edge. The total parameter count is 63,489.

**GAT (Velickovic et al., 2018).** The implementation uses two GAT convolutional layers with hidden dimension 128 and a single attention head. The original GAT paper reported that multi-head attention (typically 4 or 8 heads) was important for stable training. However, preliminary experiments with 4 heads on the full HI-Small graph (5 million edges) caused memory exhaustion on CPU. With a single head, the model has 64,001 parameters and completed training successfully. The use of a single head likely reduces the expressiveness of the attention mechanism, since the model cannot attend to different relational patterns in parallel, but trade-offs of this kind are unavoidable when training large-graph models on CPU-constrained hardware.

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

The selection of seven architectures across three tiers is justified by the research objective of isolating the contribution of graph structure and temporal modelling to detection performance. A narrower comparison, for example, comparing only GCN against EvolveGCN, would identify which temporal model performs better but could not determine whether either outperforms non-graph baselines. A broader comparison, adding architectures such as Graph Isomorphism Networks (Xu et al., 2019) or temporal attention-based models, would provide more comprehensive coverage but at the cost of computational feasibility: each additional architecture requires a full training cycle on 5 million edges.

The specific static architectures (GCN, GAT, GraphSAGE) were chosen because they represent the three dominant design philosophies in static GNN research: spectral convolution, attention-based aggregation, and sampling-based inductive learning. They are also the architectures for which the IBM AML dataset paper (Altman et al., 2023) reported baseline results, enabling direct comparison. The temporal architectures (TemporalGCN, EvolveGCN-H, TGN) were chosen to span the two temporal modelling paradigms: snapshot-based (with both state-space and weight-space evolution) and continuous-time. This coverage ensures that the study's findings about temporal granularity are not specific to a single architecture or paradigm.

**3.5 Training and Evaluation Protocol (SQ2 and SQ3)**

This section describes the training and evaluation protocol shared across all models, addressing SQ3 by establishing the conditions under which the comparative evaluation is conducted.

**3.5.1 Loss Functions and Class Weighting**

All models were trained to minimise weighted binary cross-entropy loss. For a training set with N\_neg legitimate transactions and N\_pos laundering transactions, the pos_weight is computed as N\_neg / N\_pos. For the HI-Small training partition, this yields a value of approximately 1244.

For static GNNs and snapshot temporal models, a pos_weight_multiplier of 0.1 was applied, yielding an effective pos_weight of approximately 124. For TGN, a lower multiplier of 0.01 was used (effective pos_weight approximately 12.4), following empirical observation that the larger multiplier produced unstable training in combination with the per-batch memory updates. The difference arises because TGN processes edges in batches with online memory updates, creating a noisier gradient environment than the full-graph training of static GNNs. A lower pos_weight reduces the variance of minority class gradients, stabilising training.

The use of pos_weight rather than alternative class imbalance handling techniques (oversampling the minority class, undersampling the majority class, or using focal loss) was chosen for two reasons. Oversampling and undersampling modify the effective training distribution and can distort the temporal structure of the data: oversampling repeats transactions, creating artificial temporal dependencies, while undersampling discards potentially informative legitimate transactions. Focal loss (Lin et al., 2017) down-weights easy examples to focus training on hard ones, which is conceptually appealing for AML but introduces an additional focusing hyperparameter. Weighted cross-entropy is the simplest and most transparent approach, and its single parameter (pos_weight) has a clear interpretation.

**3.5.2 Hyperparameter Configuration**

Hyperparameters were set based on architectural defaults from the original papers, with manual adjustment where needed for training stability on this dataset. No automated hyperparameter optimisation (grid search, random search, or Bayesian optimisation) was performed, which is acknowledged as a limitation. The configurations used for each model are summarised in Table 3.1.

**Table 3.1: Hyperparameter configurations.**

| Parameter | Static GNNs | TemporalGCN | EvolveGCN-H | TGN |
|-----------|-------------|-------------|-------------|-----|
| Hidden dim | 128 | 128 | 128 | 128 |
| Num layers | 2 | 2 | 2 | |
| Dropout | 0.3 | 0.3 | 0.3 | 0.3 |
| Learning rate | 0.001 | 0.001 | 0.001 | 0.003 |
| Weight decay | 0.0005 | 0.0005 | 0.0005 | 0.0005 |
| Grad clip | 1.0 | 1.0 | 1.0 | 0 |
| Pos weight mult | 0.1 | 0.1 | 0.1 | 0.01 |
| Epochs | 200 | 200 | 200 | 100 |
| Patience | 25 | 25 | 25 | 25 |
| Batch size | Full graph | Full snapshot | Full snapshot | 2048 |
| Memory dim | | | | 64 |
| Time dim | | | | 8 |
| EMA beta | | | | 0.85 |
| Rank | | | 2 | |
| GAT heads | 1 | | | |
| SAGE aggregator | mean | | | |

For the conventional ML baselines, Logistic Regression used class_weight="balanced" and L2 regularization (C=1.0). Random Forest used 100 estimators, max_depth=10, and class_weight="balanced". XGBoost used default hyperparameters with early_stopping_rounds=20 monitored on validation log loss.

**Training duration.** All models were trained on CPU (Intel Core i7, 8 threads). Approximate training times were: Logistic Regression 2 minutes, Random Forest 10 minutes, XGBoost 3 minutes, GCN 102 minutes, GAT 154 minutes (1 head), GraphSAGE 55 minutes, TemporalGCN 65 minutes, EvolveGCN-H 50 minutes, TGN 114 minutes (100 epochs, early stopped at epoch 38). The total computational investment across all models was approximately 9 CPU-hours.

**3.5.3 Evaluation Metrics and Threshold Calibration**

All models were evaluated using five metrics: AUC-ROC, AUC-PR, Precision, Recall, and F1-score. As established in Section 2.5, these metrics are appropriate for heavily class-imbalanced data because they are not inflated by the majority class, unlike accuracy.

Precision, Recall, and F1-score are threshold-dependent: they are computed at a specific classification threshold. Reporting these metrics at a single default threshold (0.5) is standard practice but can be misleading when the optimal threshold for the minority class differs substantially from 0.5, as is typical under extreme class imbalance. To address this, each model's classification threshold was calibrated on the validation set by selecting the threshold that maximised validation F1-score. Both default-threshold (0.5) and calibrated-threshold metrics are reported in Chapter 4. The calibrated threshold was then applied to the test set for the final evaluation. This calibration procedure ensures that threshold-dependent metrics reflect each model's best achievable performance rather than an arbitrary cutoff.

For TGN, an additional evaluation was performed: per-time-slice analysis. The chronologically ordered test set was divided into 12 equal slices, and metrics were computed independently for each slice. This analysis tests whether model performance improves as more interaction history accumulates in per-node memory, providing evidence for or against temporal generalisation. A model whose performance is flat across slices shows no benefit from memory accumulation; a model whose performance improves monotonically across slices demonstrates that per-node memory captures useful behavioural signal over time.

**3.6 Ethical Considerations and Reproducibility**

**Ethical considerations.** This research uses a synthetic dataset (IBM AML HI-Small) that does not contain real personal or financial information. No primary data collection from human subjects was conducted. The dataset is publicly available and was created specifically for academic research purposes (Altman et al., 2023). The laundering labels are synthetic and do not represent accusations against real individuals or institutions.

Two ethical considerations nonetheless apply. First, the AML detection tool developed in this research could, if deployed, contribute to automated decision-making with significant consequences for individuals whose accounts are flagged as suspicious. The tool is an analytical prototype, not a production system, and its outputs should be understood as decision support for human compliance analysts, not as automated determinations of criminal activity. This distinction is important: the models presented here detect patterns statistically associated with laundering, not laundering itself. Second, the model's performance characteristics have fairness implications. If the underlying transaction data reflects biases in which accounts or transaction patterns are flagged as suspicious, the model may amplify those biases. The IBM AML dataset's laundering patterns are derived from FATF typologies rather than from real-world enforcement data, which mitigates but does not eliminate this concern.

**Reproducibility.** All experiments used a fixed random seed (42) across NumPy, PyTorch, and Python's random module. Data splits are deterministic: chronological sort followed by index-based partitioning at 70/15/15 ratios. Model initialisation is controlled by the fixed seed. Training procedures do not involve stochastic data augmentation. Under these conditions, re-running any experiment with the same arguments produces identical results.

Complete reproducibility requires the following: Python 3.11, PyTorch 2.x, PyTorch Geometric 2.5.x, scikit-learn 1.x, XGBoost 2.x, NumPy, and Pandas. The full dependency list with version numbers is specified in the project's requirements.txt file. The complete source code, including data loading, feature engineering, graph construction, model implementations, training loops, and evaluation procedures, is available in the project repository. Reproduction commands for each experiment are documented in Appendix B.

**Tool documentation.** The tool's requirements, architecture, module structure, and development process are documented in Appendix A. The documentation covers the data pipeline (loading, feature engineering, graph construction), model implementations, training procedures, and evaluation framework. The appendix is intended to enable an independent researcher or practitioner to understand, reproduce, and adapt the tool.
