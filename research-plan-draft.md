# Research Plan — Draft

**Project:** 3.4 GNN Money Laundering Detection
**Student:** Shadab Gada (500981772)
**Supervisor:** Kees van Montfort, PhD
**Last Updated:** 2026-04-18

## Status

| Section                   | Status |
| ------------------------- | ------ |
| II. Summary               | Done   |
| III. Problem Statement    | Done   |
| IV. MRQ + Sub-questions   | Done   |
| V. Theoretical Foundation | Done   |
| VI. Methodology           | Done   |
| VII. Planning (Gantt)     | Done   |
| VIII. References          | Done   |

## II. Summary of the Master Project Assignment

**Working title:** Graph Neural Networks Applied to Money Laundering Detection (tool development)

**Objective(s):**

The objective of this project is to develop and evaluate a GNN-based analytical tool for detecting money laundering in financial transaction networks. This is pursued through three specific objectives:

1. To perform the data engineering work required to transform raw transaction data from the IBM Transactions for Anti-Money Laundering (AML) dataset into graph structures suitable for GNN-based analysis.
2. To implement and evaluate four GNN architectures spanning static and temporal approaches: Graph Convolutional Networks (GCN), Graph Attention Networks (GAT), GraphSAGE, and EvolveGCN, for the task of money laundering detection on financial transaction graphs.
3. To compare the detection performance of static and temporal GNN-based models against conventional supervised machine learning classifiers (Logistic Regression, Random Forest, and XGBoost) using the Area Under the Receiver Operating Characteristic curve (AUC-ROC), Precision, Recall, and F1-score, and to assess the practical value of graph-based approaches for AML compliance.

## III. Problem Statement

Money laundering, the process through which illegally obtained funds are moved through legitimate financial channels to obscure their criminal origin, represents one of the most persistent threats to global economic stability and security (United Nations, 1988). The United Nations Office on Drugs and Crime estimates that between 2% and 5% of global GDP is laundered annually, financing organised crime, drug trafficking, terrorism, and human trafficking (UNODC, 2011). In absolute terms, this amounts to between \$800 billion and $2 trillion USD per year, making anti-money laundering (AML) enforcement a critical regulatory and operational priority for financial institutions worldwide (UNODC, 2011).

Financial institutions are legally obligated under frameworks such as the EU's Anti-Money Laundering Directives and the Financial Action Task Force (FATF) recommendations to detect and report suspicious activity (FATF, 2023). Despite substantial investment in compliance infrastructure, current AML systems remain significantly limited in their effectiveness. Traditional rule-based systems apply fixed thresholds and heuristics to flag suspicious transactions, for example, transactions above a certain amount or to high-risk jurisdictions. While these systems are interpretable, they are rigid, fail to adapt to evolving laundering tactics, and generate extremely high false positive rates. Industry reports suggest that over 95% of AML alerts are false positives, creating severe alert fatigue among compliance analysts and diverting resources from genuinely suspicious cases (Chen et al., 2018).

Conventional machine learning approaches, including logistic regression and tree-based models such as random forests and gradient boosting, have been applied to improve upon rule-based systems. These methods analyse individual transaction features in isolation, such as amount, currency, and payment format, and improve detection rates to an extent. However, they fundamentally fail to capture the relational structure of money laundering. Sophisticated laundering schemes, such as layering through chains of intermediary accounts or structuring (smurfing) across multiple accounts, are only detectable when the broader network of transactions is considered (Levi, 2002). A single transaction may appear benign; its position within a network of suspicious activity reveals its true nature.

This is the core research challenge: financial transaction data is inherently graph-structured, yet existing analytical approaches treat it as tabular. Graph Neural Networks (GNNs) offer a principled solution to this challenge. GNNs are a class of deep learning models designed to operate on graph-structured data by propagating and aggregating information across nodes and edges (Kipf & Welling, 2017). Applied to financial networks, where accounts are nodes and transactions are edges, GNNs can learn from the entire web of financial interactions, capturing the relational and contextual patterns that distinguish legitimate activity from laundering behaviour (Johannessen & Jullum, 2025).

However, a critical limitation persists in existing GNN-based AML research. GNN-based approaches have demonstrated promise in financial crime detection across cryptocurrency and banking AML contexts (Alarab & Prakoonwit, 2023; Johannessen & Jullum, 2025; Pareja et al., 2020; Weber et al., 2019), yet existing studies predominantly employ static graph models that treat all transactions as a single snapshot, ignoring the temporal ordering of financial interactions. Money laundering is an inherently sequential process: layering chains and structuring schemes unfold across time-ordered transaction sequences, and static models structurally cannot capture this ordering. No study has therefore produced a systematic comparative evaluation of temporal and static GNN architectures alongside conventional supervised classifiers within a unified framework on a standardised public AML benchmark (Cheng et al., 2024; Johannessen & Jullum, 2025). The research gap is therefore the absence of exactly this unified comparison, spanning all three tiers: conventional machine learning, static GNNs, and a temporal GNN.

This research fills that gap through tool development: building a GNN-based analytical system trained on the IBM Transactions for Anti-Money Laundering (AML) dataset (Altman et al., 2023), a large-scale synthetic dataset specifically designed for AML research containing millions of labelled financial transactions. The tool implements four GNN architectures spanning static and temporal approaches: Graph Convolutional Networks (GCN; Kipf & Welling, 2017), Graph Attention Networks (GAT; Veličković et al., 2018), GraphSAGE (Hamilton et al., 2017), and EvolveGCN (Pareja et al., 2020), a temporal GNN that captures time-ordered transaction dynamics. These are evaluated against conventional supervised machine learning baselines, specifically Logistic Regression, Random Forest, and XGBoost (Chen et al., 2018; Chen & Guestrin, 2016). Given the heavily class-imbalanced nature of AML transaction datasets, model performance is assessed using metrics appropriate for such conditions: the Area Under the Receiver Operating Characteristic curve (AUC-ROC), Precision, Recall, and F1-score, in preference to classification accuracy, which is an unreliable indicator under severe class imbalance (He & Garcia, 2009).

The novelty of this research lies in the comparative benchmarking framework itself. While temporal GNN approaches have been explored in cryptocurrency AML contexts (Alarab & Prakoonwit, 2023; Pareja et al., 2020), EvolveGCN has not previously been evaluated on a standardised public banking transaction benchmark. This research addresses that gap directly: EvolveGCN is evaluated alongside static GNN architectures (GCN, GAT, and GraphSAGE) and conventional supervised classifiers (Logistic Regression, Random Forest, and XGBoost), all trained and tested on the same dataset under identical experimental conditions. This produces empirical evidence on whether incorporating temporal transaction dynamics improves money laundering detection, and provides a systematic basis for practitioner model selection in AML compliance contexts. Delivering this requires an inherently multidisciplinary approach, combining financial domain knowledge to understand laundering typologies and regulatory requirements, data engineering expertise to construct graph representations from raw transaction data, and machine learning expertise to design, train, and evaluate GNN architectures.

## IV. Main Research Question and Sub-questions

**Main Research Question:**
How do static and temporal Graph Neural Network architectures compare to conventional supervised machine learning classifiers in detecting money laundering in financial transaction networks?

**Sub-questions:**

**SQ1.** What graph construction design decisions are required to represent financial transaction data as graph structures for static and temporal GNN-based AML analysis, and what is the rationale for each?

**SQ2.** How does the choice of GNN architecture affect money laundering detection performance on financial transaction networks, specifically comparing static architectures (GCN, GAT, and GraphSAGE) against a temporal architecture (EvolveGCN)?

**SQ3.** How does the performance of static and temporal GNN-based models compare to Logistic Regression, Random Forest, and XGBoost in detecting money laundering, as measured by AUC-Precision, Recall, and F1-score?

## V. Theoretical Foundation

1. Cheng, D., Zou, Y., Xiang, S., & Jiang, C. (2024). Graph neural networks for financial fraud detection: A review. *Frontiers of Computer Science*. https://doi.org/10.1007/s11704-024-40474-y
   
   Provides a comprehensive review of GNN architectures applied to financial fraud detection, covering GCN, GAT, and GraphSAGE variants. Systematically evaluates their effectiveness across different fraud types and datasets. Explicitly identifies the incorporation of temporal dynamics into GNN architectures as a key future research direction, providing direct support for this research's focus on comparative evaluation of static and temporal GNN approaches.

2. Chen, Z., Van Khoa, L. D., Teoh, E. N., Nazir, A., Karuppiah, E. K., & Lam, K. S. (2018). Machine learning techniques for anti-money laundering (AML) solutions in suspicious transaction detection: A review. *Knowledge and Information Systems*, *57*(2), 245–285.
   
   Reviews conventional machine learning techniques applied to AML, including supervised and unsupervised approaches. Identifies key limitations of existing methods, particularly their inability to capture relational transaction patterns, providing the theoretical basis for this research's shift towards graph-based detection approaches.

3. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794.
   
   Introduces XGBoost, a scalable gradient boosting framework that has become a standard benchmark in tabular classification tasks. As one of three conventional supervised classifiers used as baselines in this research, this paper provides the algorithmic foundation for the comparative performance evaluation against GNN-based models.

4. Dou, Y., Liu, Z., Sun, L., Deng, S., Peng, H., & Yu, P. S. (2020). Enhancing graph neural network-based fraud detection via imbalanced graph classification. *Proceedings of the 29th ACM International Conference on Information and Knowledge Management (CIKM)*.
   
   Addresses the dual challenge of applying GNNs to fraud detection under severe class imbalance, proposing techniques to improve minority fraud node detection. Directly relevant as the dataset used in this research is heavily imbalanced, and the findings inform both model design decisions and evaluation strategy.

5. Hamilton, W. L., Ying, R., & Leskovec, J. (2017). Inductive representation learning on large graphs. *Advances in Neural Information Processing Systems*, *30*.
   
   Introduces GraphSAGE, an inductive learning framework that generates node embeddings by sampling and aggregating features from local neighbourhoods. Unlike transductive methods, GraphSAGE generalises to unseen nodes, making it suitable for financial transaction networks where new accounts continuously appear.

6. He, H., & Garcia, E. A. (2009). Learning from imbalanced data. *IEEE Transactions on Knowledge and Data Engineering*, *21*(9), 1263–1284.
   
   Provides a comprehensive analysis of class imbalance challenges in machine learning, reviewing resampling techniques, cost-sensitive learning, and evaluation metric selection. Directly informs this research's choice of AUC-ROC, Precision, Recall, and F1-score as primary metrics, given the heavily imbalanced nature of AML transaction datasets.

7. Johannessen, F., & Jullum, M. (2025). Finding money launderers using heterogeneous graph neural networks. *Journal of Finance and Data Science*, *11*, Article 100175. https://doi.org/10.1016/j.jfds.2025.100175
   
   Applies heterogeneous GNNs to AML detection using real-world banking data, demonstrating that graph-based models outperform conventional classifiers by capturing multi-relational transaction patterns. Directly validates the core premise of this research and provides a methodological reference for applying GNNs to financial AML datasets.

8. Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. In *Proceedings of the International Conference on Learning Representations (ICLR)*.
   
   Introduces Graph Convolutional Networks (GCN), a foundational architecture for node classification on graph-structured data. GCN propagates and aggregates feature information across neighbouring nodes using a spectral convolution approximation, establishing the baseline GNN architecture implemented and evaluated in this research.

9. Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018). Graph attention networks. In *Proceedings of the International Conference on Learning Representations (ICLR)*.
   
   Introduces Graph Attention Networks (GAT), extending GCN by applying learnable attention coefficients to neighbour aggregation, allowing the model to weight individual connections by importance. In financial transaction networks, this enables the model to assign higher attention to interactions with potentially suspicious counterparties.

10. Weber, M., Domeniconi, G., Chen, J., Weidele, D. K. I., Bellei, C., Robinson, T., & Leiserson, C. E. (2019). Anti-money laundering in bitcoin: Experimenting with graph convolutional networks for financial forensics. In *Proceedings of the KDD Workshop on Anomaly Detection in Finance*.
    
    One of the earliest applications of GCN to financial crime detection, using the Elliptic Bitcoin dataset to classify transactions as illicit or licit. Establishes empirical precedent for graph-based AML detection and provides a benchmark for comparing GNN performance against conventional classifiers.

11. Pareja, A., Domeniconi, G., Chen, J., Ma, T., Suzumura, T., Kanezashi, H., Kaler, T., & Leiserson, C. E. (2020). EvolveGCN: Evolving graph convolutional networks for dynamic graphs. *Proceedings of the AAAI Conference on Artificial Intelligence*, *34*(04), 5363–5370. https://doi.org/10.1609/aaai.v34i04.5984
    
    Introduces EvolveGCN, a temporal GNN that adapts GCN weight matrices over time using a recurrent neural network, enabling the model to capture the evolution of graph structure across time-ordered snapshots. Directly implemented in this research as the temporal GNN architecture, allowing evaluation of whether capturing transaction ordering improves money laundering detection performance over static GNN baselines.

12. Alarab, I., & Prakoonwit, S. (2023). Graph-based LSTM for anti-money laundering: Experimenting temporal graph convolutional network with Bitcoin data. *Neural Processing Letters*, *55*, 689–707. https://doi.org/10.1007/s11063-022-10904-8
    
    Applies a temporal GCN with LSTM to money laundering detection on the Elliptic Bitcoin dataset, explicitly identifying that existing GNN-based AML studies have largely neglected temporal information. Supports the research gap motivating this study and provides methodological precedent for incorporating temporal dynamics into GNN-based AML detection.

13. Financial Action Task Force. (2023). *The FATF Recommendations: International standards on combating money laundering and the financing of terrorism and proliferation* (Updated June 2023). FATF/OECD. https://www.fatf-gafi.org/en/topics/fatf-recommendations.html
    
    Establishes the global regulatory standards that define how financial institutions are required to detect, prevent, and report money laundering. Directly grounds the compliance context motivating this research and documents the core AML typologies, including structuring, layering, and integration, that inform the graph construction design decisions and situate the study within the multidisciplinary domain of financial regulation and data science.

14. Levi, M. (2002). Money laundering and its regulation. *The ANNALS of the American Academy of Political and Social Science*, *582*(1), 181–194.
    
    Analyses money laundering as a criminological and regulatory phenomenon, examining how illicit funds are moved through financial systems and the practical limitations of detection and enforcement responses. Provides the domain-level conceptual grounding for understanding the laundering typologies and transaction patterns that GNN-based models aim to detect, establishing the multidisciplinary link between financial crime and the analytical approaches employed in this research.

## Core Theoretical Foundation *(personal reference, not for submission)*

| Paper                       | Role                                                               |
| --------------------------- | ------------------------------------------------------------------ |
| Kipf & Welling (2017)       | The method: foundational GCN architecture                          |
| Johannessen & Jullum (2025) | Domain validation: GNNs applied to AML specifically                |
| Chen et al. (2018)          | Problem justification: why conventional ML is insufficient for AML |

## VI. Methodological and Tool Development Approaches

This section describes the methodological approaches and tool development strategies employed in this research. The project is classified as applied research with a tool development orientation, combining a literature study, secondary data collection, and the design and evaluation of a GNN-based analytical tool. The overall research design follows a deductive approach, in which theoretical frameworks from the academic literature on GNNs and AML are operationalised and empirically tested on real-world-representative financial transaction data (Sekaran & Bougie, 2019).

### 6.1 Literature Study

A structured literature study forms the first methodological component of this research. Its purpose is threefold: to establish the theoretical foundation for GNN-based AML detection, to determine which GNN architectures are appropriate for financial crime detection and understand how they function, and to identify the limitations of conventional machine learning approaches that motivate this research.

The literature study follows a systematic approach. Academic databases including Google Scholar, IEEE Xplore, ACM Digital Library, and arXiv are used as primary search platforms. Search terms include combinations of "graph neural network", "anti-money laundering", "financial fraud detection", "GCN", "GAT", "GraphSAGE", "EvolveGCN", "temporal GNN", "dynamic graph", "transaction network", "class imbalance", "AML compliance", "money laundering typologies", and "financial regulation". Selection criteria require that articles are peer-reviewed or published in recognised conference proceedings and are directly relevant to the focal research domain. Articles are selected from within the last decade where possible, though foundational architectural works (e.g., Kipf & Welling, 2017; Hamilton et al., 2017; Pareja et al., 2020) are included regardless of publication date given their methodological centrality.

The literature study directly informs the tool development process by identifying the appropriate GNN architectures (GCN, GAT, GraphSAGE, EvolveGCN), the relevant evaluation metrics for imbalanced datasets (AUC-ROC, Precision, Recall, F1-score), and the conventional supervised classifiers (Logistic Regression, Random Forest, XGBoost) used for comparative evaluation. It also establishes the theoretical basis for the graph construction methodology applied to the IBM AML dataset, and grounds the research in the regulatory and criminological context of AML compliance through engagement with domain literature on money laundering typologies and financial regulation (FATF, 2023; Levi, 2002).

### 6.2 Data Collection: Secondary Research

This research relies exclusively on secondary data. No primary data collection, such as interviews, surveys, or direct data gathering from financial institutions, is conducted. The use of secondary data is appropriate here because the research objective is tool development and comparative model evaluation, for which a representative, labelled dataset is required rather than newly generated observations. Secondary data collection involves identifying, accessing, and processing existing data sources that are fit for the research purpose (Sekaran & Bougie, 2019).

The selected secondary data source is the IBM Transactions for Anti-Money Laundering (AML) dataset (Altman et al., 2023), publicly available on Kaggle. This dataset was chosen after evaluating it against an alternative candidate, the Synthetic AML Dataset (SAML-D) (Oztas et al., 2023). Both provide full ground-truth labelling of transactions as legitimate or suspicious, and both are synthetic, which is necessary given that real-world AML data from financial institutions is not publicly accessible due to privacy and regulatory constraints (Chen et al., 2018). The rationale for selecting the IBM AML dataset over SAML-D is as follows.

SAML-D is a tabular dataset containing approximately nine million transactions with international scope, including features such as country of origin and destination, currency, transaction amount, and account identifiers. While it offers geographic diversity, its structure is designed primarily for conventional tabular machine learning approaches (Oztas et al., 2023). Constructing a graph representation from SAML-D requires extracting and deduplicating unique account identities from flat transaction records, as the dataset does not provide an explicit account-level structure. This makes defining the stable node identities required for GNN-based analysis a non-trivial preprocessing step.

The IBM AML dataset, by contrast, is structured natively as a graph. It provides a dedicated accounts file in which each unique account is represented as a distinct entity, and a transactions file in which each transaction is a directed interaction between two accounts, annotated with features including amount, timestamp, and payment format, along with a binary label indicating whether the transaction is part of a laundering scheme. This structure maps directly and unambiguously onto the graph representation required for GNN-based analysis: accounts become nodes and transactions become directed, labelled edges (Weber et al., 2019; Johannessen & Jullum, 2025). The dataset contains millions of labelled transactions and was explicitly designed and published to support GNN-based AML research (Altman et al., 2023). Critically, the laundering patterns encoded in the dataset are derived from FATF-documented AML typologies, including structuring, layering, and fan-in/fan-out schemes, ensuring that the synthetic transaction patterns reflect the structure of real-world laundering behaviour as codified in international regulatory standards (FATF, 2023; Altman et al., 2023).

A further advantage specific to this research is that the IBM AML dataset's explicit account-level structure enables stable node identities to be maintained across time-ordered graph snapshots, which is a prerequisite for the temporal GNN architecture implemented in this study (Pareja et al., 2020). Because each account is defined as a persistent entity independent of individual transactions, the graph can be partitioned into time windows without ambiguity about node identity. This directly supports the temporal modelling component of this research (Alarab & Prakoonwit, 2023).

One limitation of the IBM AML dataset is that it models transactions within a single financial system and does not include cross-border features such as country of origin or destination. SAML-D's international scope is an advantage in terms of geographic realism. However, for the purposes of this research, which focuses on the comparative evaluation of GNN architectures and conventional classifiers rather than on geographic generalisability, this limitation does not affect the validity of the experimental findings. The IBM AML dataset is selected on the basis that its native graph structure, stable account-level entities, and explicit design for GNN-based AML research make it the most appropriate data source for the objectives of this study (Altman et al., 2023; Cheng et al., 2024; Johannessen & Jullum, 2025).

A further consideration concerns ecological validity: models trained on synthetic data may to some degree reflect statistical properties of the simulation algorithm rather than authentic laundering behaviour. This is acknowledged as a study limitation, but it must be understood in the context of how the AML research field is structured. Real-world AML transaction data held by financial institutions is non-public by regulatory design, not merely by practical inconvenience. Financial institutions are legally prohibited from releasing customer transaction data under AML confidentiality obligations and data protection regulation, and suspicious activity reports are protected by law across jurisdictions (FATF, 2023). Where studies have used real institutional data, such as Johannessen & Jullum (2025), who accessed proprietary transaction records from DNB, this reflects a closed institutional partnership that is not replicable by independent researchers and does not constitute a publicly available benchmark against which external research can be evaluated or extended. The IBM AML dataset was created precisely to fill this gap in the research ecosystem. Published at NeurIPS 2023, it was designed as a replicable public benchmark enabling the research community to develop and evaluate AML detection models without requiring restricted institutional access (Altman et al., 2023). Its laundering typologies are grounded in FATF regulatory documentation (FATF, 2023; Altman et al., 2023), ensuring the synthetic patterns reflect real-world laundering behaviour as codified in international standards rather than arbitrary simulation choices. The use of this dataset is therefore not a methodological compromise but the appropriate and standard foundation for academic AML research, consistent with the broader published literature in this domain (Weber et al., 2019; Alarab & Prakoonwit, 2023).

### 6.3 Data Engineering and Graph Construction (SQ1)

Before model training can begin, the raw tabular transaction data from the IBM AML dataset must be transformed into a graph structure suitable for GNN-based analysis. This constitutes the data engineering component of the project and directly addresses SQ1 by establishing and justifying the graph construction design decisions required for both static and temporal GNN-based analysis.

The data transformation proceeds in the following stages. First, raw CSV transaction files are ingested and validated, with data cleaning steps applied to handle missing values, outliers, and format inconsistencies. Second, a directed graph is constructed in which each unique account identifier becomes a node and each transaction becomes a directed edge from the originator node to the beneficiary node. Third, node-level and edge-level features are engineered. Edge features include transaction amount, payment format, and timestamp-derived temporal features (e.g., hour of day, day of week). Node features are aggregated from transaction history and include total transaction volume, average transaction size, number of unique counterparties, and temporal activity patterns. The resulting graph, together with associated feature matrices and binary edge labels (laundering or non-laundering), serves as the unified input to all subsequent modelling stages.

Three design decisions are central to this construction and directly answer SQ1. First, representing accounts as nodes and transactions as directed edges is justified by the native graph structure of the IBM AML dataset and established precedent in GNN-based AML research (Weber et al., 2019; Johannessen & Jullum, 2025). Second, maintaining stable account-level node identities across time is a prerequisite for EvolveGCN, which requires consistent node representation across time-ordered snapshots (Pareja et al., 2020). Third, including timestamp-derived temporal features enables both static models, which consume them as edge attributes, and the temporal model, which uses them to partition the graph into time windows, ensuring a single unified graph construction serves all four architectures.

### 6.4 Model Development and Training (SQ2)

Four GNN architectures are implemented and evaluated to address SQ2, comprising three static architectures and one temporal architecture. The static architectures are GCN, GAT, and GraphSAGE, which represent a progression of GNN design philosophy. GCN (Kipf & Welling, 2017) applies symmetric neighbourhood aggregation using a spectral convolution approximation, establishing the baseline for graph-based classification. GAT (Veličković et al., 2018) extends this by introducing learnable attention coefficients that allow the model to weight neighbour contributions by relevance, which is useful in financial networks where certain counterparty relationships carry stronger signals of suspicious activity. GraphSAGE (Hamilton et al., 2017) introduces inductive learning through neighbourhood sampling, enabling scalability to financial transaction graphs and generalisation to previously unseen nodes, reflecting the real-world condition in which new accounts are continuously added to a financial network.

The temporal architecture, EvolveGCN (Pareja et al., 2020), addresses the sequential nature of money laundering by treating the financial transaction graph as a series of time-ordered snapshots. Rather than processing all transactions statically, EvolveGCN evolves the GCN weight matrices over time using a gated recurrent unit (GRU), allowing the model to capture how transaction patterns change across time. This is particularly relevant for AML detection, where laundering typologies such as layering and structuring unfold across ordered transaction sequences rather than appearing in isolated snapshots.

The task is framed as binary edge classification: each transaction edge is labelled as laundering (1) or non-laundering (0), in line with how ground-truth labels are assigned in the IBM AML dataset. Training employs weighted binary cross-entropy loss to account for severe class imbalance, with the minority class weight set proportional to its inverse frequency in the training set. Model selection is performed on the basis of validation set AUC-ROC. Hyperparameter tuning is conducted for each architecture across the number of layers, hidden layer dimensionality, dropout rate, and (for GAT) the number of attention heads. A fixed random seed is applied throughout to ensure reproducibility, and all models are trained and evaluated on the same data splits (train, validation, test).

### 6.5 Baseline Comparison and Evaluation (SQ3)

To address SQ3, the four GNN models are compared against three conventional supervised machine learning classifiers: Logistic Regression, Random Forest, and XGBoost (Chen & Guestrin, 2016). Unlike GNN models, the baseline classifiers operate on flat feature vectors derived from the same transaction data, without access to graph-structural information. This controlled comparison isolates the contribution of graph structure and temporal modelling to detection performance, allowing the research to produce an empirical answer to whether relational and temporal modelling adds measurable value in the AML domain.

All models are evaluated on the same held-out test set using four metrics: AUC-ROC, Precision, Recall, and F1-score. These metrics are selected in preference to classification accuracy because the IBM AML dataset is heavily class-imbalanced, and accuracy is an unreliable performance indicator under such conditions (He & Garcia, 2009). AUC-ROC measures overall discriminative power across classification thresholds (He & Garcia, 2009). Precision and Recall quantify the trade-off between false positives and false negatives, which is operationally significant in AML compliance given the costs of both over-alerting and under-alerting (Chen et al., 2018). F1-score provides a harmonic mean that balances the two (He & Garcia, 2009).

### 6.6 Summary: Techniques per Sub-question

In summary, SQ1 is addressed through the data engineering and graph construction work described in Section 6.3, which establishes and justifies the design decisions required to represent financial transaction data for static and temporal GNN-based analysis. SQ2 is addressed through the comparative implementation and evaluation of GCN, GAT, GraphSAGE, and EvolveGCN as described in Section 6.4. SQ3 is addressed through the baseline classifier comparison and held-out test set evaluation described in Section 6.5, measuring performance across all seven models on AUC-ROC, Precision, Recall, and F1-score.

## VII. Planning

The project runs full-time from 13 April 2026 to final submission on 22 June 2026 (ten working weeks), followed by the presentation and defence in the week of 29 June 2026. See the accompanying Gantt chart: `gantt.xlsx`.

## VIII. References

Alarab, I., & Prakoonwit, S. (2023). Graph-based LSTM for anti-money laundering: Experimenting temporal graph convolutional network with Bitcoin data. *Neural Processing Letters*, *55*, 689–707. https://doi.org/10.1007/s11063-022-10904-8

Altman, E., Blanuša, J., von Niederhäusern, L., Egressy, B., Anghel, A., & Atasu, K. (2023). Realistic synthetic financial transactions for anti-money laundering models. In *Advances in Neural Information Processing Systems 36* (NeurIPS 2023, Datasets and Benchmarks Track). https://arxiv.org/abs/2306.16424

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794. https://doi.org/10.1145/2939672.2939785

Chen, Z., Van Khoa, L. D., Teoh, E. N., Nazir, A., Karuppiah, E. K., & Lam, K. S. (2018). Machine learning techniques for anti-money laundering (AML) solutions in suspicious transaction detection: A review. *Knowledge and Information Systems*, *57*(2), 245–285.

Cheng, D., Zou, Y., Xiang, S., & Jiang, C. (2024). Graph neural networks for financial fraud detection: A review. *Frontiers of Computer Science*. https://doi.org/10.1007/s11704-024-40474-y

Dou, Y., Liu, Z., Sun, L., Deng, S., Peng, H., & Yu, P. S. (2020). Enhancing graph neural network-based fraud detection via imbalanced graph classification. *Proceedings of the 29th ACM International Conference on Information and Knowledge Management (CIKM)*, 315–324. https://doi.org/10.1145/3340531.3411903

Financial Action Task Force. (2023). *The FATF Recommendations: International standards on combating money laundering and the financing of terrorism and proliferation* (Updated June 2023). FATF/OECD. https://www.fatf-gafi.org/en/topics/fatf-recommendations.html

Hamilton, W. L., Ying, R., & Leskovec, J. (2017). Inductive representation learning on large graphs. *Advances in Neural Information Processing Systems*, *30*.

He, H., & Garcia, E. A. (2009). Learning from imbalanced data. *IEEE Transactions on Knowledge and Data Engineering*, *21*(9), 1263–1284.

Johannessen, F., & Jullum, M. (2025). Finding money launderers using heterogeneous graph neural networks. *Journal of Finance and Data Science*, *11*, Article 100175. https://doi.org/10.1016/j.jfds.2025.100175

Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. In *Proceedings of the International Conference on Learning Representations (ICLR)*.

Levi, M. (2002). Money laundering and its regulation. *The ANNALS of the American Academy of Political and Social Science*, *582*(1), 181–194.

Oztas, B., Cetinkaya, D., Adedoyin, F., Budka, M., Dogan, H., & Aksu, G. (2023). Enhancing anti-money laundering: Development of a synthetic transaction monitoring dataset. *2023 IEEE International Conference on e-Business Engineering (ICEBE)*, 47–54. https://doi.org/10.1109/ICEBE59045.2023.00028

Pareja, A., Domeniconi, G., Chen, J., Ma, T., Suzumura, T., Kanezashi, H., Kaler, T., & Leiserson, C. E. (2020). EvolveGCN: Evolving graph convolutional networks for dynamic graphs. *Proceedings of the AAAI Conference on Artificial Intelligence*, *34*(04), 5363–5370. https://doi.org/10.1609/aaai.v34i04.5984

Sekaran, U., & Bougie, R. (2019). *Research methods for business: A skill-building approach* (8th ed.). Wiley.

United Nations. (1988). *United Nations convention against illicit traffic in narcotic drugs and psychotropic substances*. United Nations Treaty Series.

United Nations Office on Drugs and Crime. (2011). *Estimating illicit financial flows resulting from drug trafficking and other transnational organized crimes*. UNODC.

Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018). Graph attention networks. In *Proceedings of the International Conference on Learning Representations (ICLR)*.

Weber, M., Domeniconi, G., Chen, J., Weidele, D. K. I., Bellei, C., Robinson, T., & Leiserson, C. E. (2019). Anti-money laundering in bitcoin: Experimenting with graph convolutional networks for financial forensics. In *Proceedings of the KDD Workshop on Anomaly Detection in Finance*.
