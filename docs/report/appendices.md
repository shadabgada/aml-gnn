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
| 6-12 | pmt_{category} | One-hot (7) | transactions.csv | Payment Format one-hot: ACH, Cheque, Credit Card, Domestic Wire, International Wire, Cash, Unknown. Exactly one column = 1 per transaction. |
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

| Model | AUC-ROC | AUC-PR | F1 |
|-------|---------|--------|-----|
| Logistic Regression | 0.9378 | 0.0376 | 0.0267 |
| Random Forest | 0.8603 | 0.0619 | 0.0070 |
| XGBoost | 0.9381 | 0.1511 | 0.0514 |
| GCN | 0.9705 | 0.1882 | 0.2513 |
| GAT | 0.9581 | 0.0958 | 0.0979 |
| GraphSAGE | 0.9459 | 0.0420 | 0.0946 |
| TemporalGCN | 0.9570 | 0.0637 | 0.1343 |
| EvolveGCN-H | 0.8972 | 0.0275 | 0.0631 |
| TGN | 0.9684 | 0.3195 | 0.3527 |

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

This appendix declares the use of generative AI tools in the preparation of this thesis, in accordance with the Amsterdam University of Applied Sciences Master Project module guide requirements.

**Tool used:** Claude Code (Anthropic), powered by Claude Opus 4.7.

**Nature of use:**
- Assistance with drafting and revising report chapter text based on experimental results, methodological documentation, and supervisor feedback provided by the author.
- Formatting of tables and structural organisation of report content.
- Code review and documentation of the software tool developed for this research.

**Nature of author contribution:**
- All experimental design, implementation, and execution was performed by the author.
- All research questions, methodological decisions, and conclusions were formulated by the author.
- All literature review, citation selection, and theoretical framework development was performed by the author.
- The author directed the drafting process, provided all substantive content (experimental results, architectural descriptions, methodological reasoning), reviewed all AI-generated text for accuracy and appropriateness, and takes full responsibility for the final content of this thesis.

**Verification:** All factual claims, numerical results, and citations in this thesis have been verified by the author against primary sources (experimental logs, published papers, and the assessment rubric).

---

**Appendix D: Full Results Tables**

This appendix reproduces the complete results from Chapter 4 for reference. The data is identical to that presented in the chapter body and in the project's `docs/RESULTS.md` file.

**Table D.1: Conventional ML baseline results (random 70/15/15 split, threshold 0.50).**

| Model | AUC-ROC | AUC-PR | Precision | Recall | F1 |
|-------|---------|--------|-----------|--------|-----|
| Logistic Regression | 0.9378 | 0.0376 | 0.0135 | 0.9295 | 0.0267 |
| Random Forest | 0.8603 | 0.0619 | 0.0035 | 0.9148 | 0.0070 |
| XGBoost | 0.9381 | 0.1511 | 0.0265 | 0.8610 | 0.0514 |

**Table D.2: Static GNN results (random 70/15/15 split, calibrated thresholds).**

| Model | Params | AUC-ROC | AUC-PR | Precision | Recall | F1 | Thresh |
|-------|--------|---------|--------|-----------|--------|-----|--------|
| GCN | 63,489 | 0.9705 | 0.1882 | 0.1846 | 0.3933 | 0.2513 | 0.7029 |
| GAT (1 head) | 64,001 | 0.9581 | 0.0958 | 0.0539 | 0.5317 | 0.0979 | 0.5544 |
| GraphSAGE | 81,409 | 0.9459 | 0.0420 | 0.0563 | 0.2953 | 0.0946 | 0.4852 |

**Table D.3: Temporal GNN results (chronological split, calibrated thresholds).**

| Model | Params | AUC-ROC | AUC-PR | Precision | Recall | F1 | Thresh |
|-------|--------|---------|--------|-----------|--------|-----|--------|
| TemporalGCN | 161,793 | 0.9570 | 0.0637 | 0.1177 | 0.1563 | 0.1343 | 0.7326 |
| EvolveGCN-H | 578,369 | 0.8972 | 0.0275 | 0.0465 | 0.0982 | 0.0631 | 0.7029 |
| EvolveGCN-H (rank=8) | 33M | N/A | N/A | N/A | N/A | N/A | N/A |
| TGN | 119,000 | 0.9684 | 0.3195 | 0.4257 | 0.3011 | 0.3527 | 0.4159 |

**Table D.4: TGN per-slice performance (12 equal chronological test set slices, threshold 0.50).**

| Slice | AUC-ROC | AUC-PR | Precision | Recall |
|-------|---------|--------|-----------|--------|
| 0 (earliest test) | 0.9205 | 0.0502 | 0.0000 | 0.0000 |
| 3 | 0.9280 | 0.0853 | 0.6000 | 0.0405 |
| 6 | 0.9714 | 0.0712 | 0.0000 | 0.0000 |
| 9 | 0.9591 | 0.0769 | 0.1294 | 0.1028 |
| 10 | 0.9563 | 0.1875 | 0.1553 | 0.3617 |
| 11 (latest test) | 0.9732 | 0.4518 | 0.5749 | 0.3300 |
