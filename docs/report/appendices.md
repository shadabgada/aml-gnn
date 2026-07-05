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
