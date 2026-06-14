# Table of Contents

## Chapter 1: Introduction
1.1 Background: Money Laundering as a Global Challenge
1.2 Current AML Detection Approaches and Their Limitations
1.3 The Graph-Structured Nature of Financial Transactions
1.4 Problem Statement
1.5 Research Objectives
1.6 Main Research Question and Sub-Questions
1.7 Contributions
1.8 Report Structure

## Chapter 2: Theoretical Framework (Literature Review)
2.1 Money Laundering Typologies and Regulatory Context
2.2 Conventional Machine Learning for AML Detection
2.3 Graph Neural Networks: Foundations
    2.3.1 GCN, GAT, and GraphSAGE
    2.3.2 GNNs for Financial Crime Detection
2.4 Temporal Graph Neural Networks
    2.4.1 Snapshot-Based Approaches (TemporalGCN, EvolveGCN)
    2.4.2 Continuous-Time Approaches (TGN)
2.5 Evaluation Under Class Imbalance
2.6 Research Gap Synthesis

## Chapter 3: Research Methodology and Tool Development
3.1 Research Design Overview
3.2 Dataset: IBM AML HI-Small
    3.2.1 Dataset Selection and Justification
    3.2.2 Dataset Characteristics
3.3 Data Engineering and Graph Construction (SQ1)
    3.3.1 Feature Engineering
    3.3.2 Graph Construction for Static and Temporal Models
    3.3.3 Chronological Data Splitting
3.4 Model Architectures (SQ2)
    3.4.1 Conventional ML Baselines (LR, RF, XGBoost)
    3.4.2 Static GNNs (GCN, GAT, GraphSAGE)
    3.4.3 Snapshot Temporal GNNs (TemporalGCN, EvolveGCN-H)
    3.4.4 Continuous-Time TGN
    3.4.5 Design Justification
3.5 Training and Evaluation Protocol (SQ3)
    3.5.1 Loss Functions and Class Weighting
    3.5.2 Hyperparameter Configuration
    3.5.3 Evaluation Metrics and Threshold Calibration
3.6 Ethical Considerations and Reproducibility

## Chapter 4: Results, Analyses and Tool Performance
4.1 Conventional ML Baseline Results
4.2 Static GNN Results
4.3 Snapshot Temporal GNN Results
4.4 Continuous-Time TGN Results
4.5 Cross-Model Comparison
4.6 Temporal Generalisation Analysis
4.7 Scalability and Generalizability Assessment

## Chapter 5: Discussion, Recommendations and Conclusions
5.1 Answering the Research Sub-Questions
    5.1.1 SQ1: Graph Construction Design Decisions
    5.1.2 SQ2: Architecture Choice and Detection Performance
    5.1.3 SQ3: GNNs vs Conventional Classifiers
5.2 Answering the Main Research Question
5.3 Practical Implications for AML Compliance Practice
    5.3.1 Model Selection Guidance for Practitioners
    5.3.2 Operational Trade-Offs
    5.3.3 Deployment Considerations
5.4 Theoretical Implications and Novelty
5.5 Limitations and Future Research
5.6 Final Conclusions

## Appendices
A. Tool Requirements and Development Documentation
B. Reproducibility Guide
C. GenAI Usage Declaration
D. Full Results Tables

## References
