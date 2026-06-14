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
