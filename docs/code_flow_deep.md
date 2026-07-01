# Deep Code Flow — What Happens Inside Each Function

## loader.py → load_raw_data(cfg)

4 steps:

1. **Locate CSV files** — `_find_file()` scans `data/raw/` for filenames containing both the variant name (e.g. "HI-Small") and a keyword ("account" or "Trans"). Returns the matching Path.

2. **Read CSVs** — `pd.read_csv()` loads both files into DataFrames.

3. **Normalise schema** — `_normalise_accounts()` lowercases column names, builds composite `account_id = "{bank_id}_{account_number}"`. `_normalise_transactions()` lowercases, handles pandas deduplication of the second "Account" column (pandas renames it to "Account.1" → mapped to "account_to"), builds composite `from_account` and `to_account`, renames columns to canonical names (timestamp, amount, currency, payment_format, is_laundering, etc.), casts label to int.

4. **Validate** — `_validate()` checks required columns exist, finds orphan accounts (accounts appearing in transactions but not in accounts.csv), adds them with NaN features, and logs the class imbalance ratio.

Returns: `(accounts_df, transactions_df)` with canonical column names.

---

## feature_engineering.py → engineer_node_features(accounts, transactions)

5 steps:

1. **Set index** — Sort accounts by `account_id`, drop pure-ID columns (account_number, entity_id).

2. **Extract entity type** — From "Corporation #33520", splits on "#" and takes the first part ("Corporation"). Becomes a categorical feature.

3. **Label-encode categoricals** — bank_name, bank_id, entity_type get LabelEncoder treatment. Low-cardinality integer columns (nunique ≤ 20) also get label-encoded.

4. **Compute transaction stats** — `_compute_transaction_stats()` groups transactions by `from_account` and `to_account` separately, computes per-account: degree_out, degree_in, degree_total, total_amount_out, total_amount_in, avg_amount_out, avg_amount_in, num_counterparties_out, num_counterparties_in. All counts and amounts get log1p'd later (implicitly — actually the raw values are used and log1p is applied via StandardScaler on the concatenated result... 

Wait — checking the code: the stats are computed as raw values. The log1p is NOT applied in feature_engineering.py. The StandardScaler standardizes the raw values. But the appendix says log1p is applied. Let me check...

Actually looking at the code more carefully: `_compute_transaction_stats` returns raw counts/amounts, and then `engineer_node_features` just does `StandardScaler().fit_transform()`. There's no explicit log1p in the node feature code. The appendix says log1p-transformed — this might be a documentation discrepancy, or the log1p happens elsewhere. Regardless, the code path is: raw stats → join → StandardScaler.

5. **Join + scale** — Left-join transaction stats onto accounts (accounts with no transactions get zeros). Fit StandardScaler on the result (if fit=True), return (N, 12) float32 array.

---

## feature_engineering.py → engineer_edge_features(transactions)

5 steps:

1. **Amount log1p** — `np.log1p(amount)` → (E, 1) float32 column.

2. **Cyclic time encoding** — Parse timestamp → extract hour (0-23) and day-of-week (0-6). Compute sin/cos pairs: `sin(2π * hour/24)`, `cos(2π * hour/24)`, `sin(2π * dow/7)`, `cos(2π * dow/7)`. This ensures 23:59 and 00:01 are close in feature space (unlike linear encoding).

3. **Payment format one-hot** — Label-encode the payment_format strings, then `np.eye(n_categories)[encoded]` → 7 one-hot columns (ACH, Cheque, Credit Card, etc.).

4. **Currency one-hot** — Same pattern → 15 one-hot columns (USD, EUR, GBP, etc.).

5. **Amount paid log1p** — Same as step 1 but for amount_paid column (may differ from amount received due to currency conversion).

6. **Scale numeric columns only** — The first 6 columns (amount_log1p, hour_sin, hour_cos, dow_sin, dow_cos, amount_paid_log1p) get StandardScaler. The 22 one-hot columns (7 payment + 15 currency) are left as-is (already bounded to {0,1}).

Returns: `(edge_features, scaler, payment_encoder, currency_encoder, feature_names)` — (E, 28) float32 array.

---

## graph_constructor.py → build_static_graph(accounts, transactions, cfg)

7 steps:

1. **Global node index** — `_build_node_index()` creates a sorted list of all account_ids and a dict mapping each to 0..N-1. Includes orphan accounts from validation.

2. **Time-based split** — `_time_based_split()` sorts transactions by timestamp, assigns first 70% to train, next 15% to val, last 15% to test. Returns three boolean Series (same length as transactions).

3. **Node features** — Calls `engineer_node_features()` on training transactions ONLY (fit=True). Then `_align_node_features()` pads the result to cover all N nodes (accounts not in training get zero vectors).

4. **Edge features** — Calls `engineer_edge_features()` on ALL transactions (fit=True). The fitted encoders are stored for later use.

5. **Edge index** — Maps `from_account` and `to_account` strings to integer indices via `account_to_idx` dict. Stacks into (2, E) LongTensor.

6. **Labels + masks** — `edge_label` from `is_laundering` column (float32). Train/val/test masks as boolean tensors.

7. **Pos weight** — `_compute_pos_weight()` = n_neg / n_pos on training edges only. For HI-Small this is ~1244.

Returns: `StaticGraph` dataclass wrapping a PyG `Data` object with (x, edge_index, edge_attr, edge_label, train_mask, val_mask, test_mask).

---

## temporal_data_builder.py → build_temporal_data(accounts, transactions, cfg)

9 steps:

1. **Global node index** — Same `_build_node_index()` as static graph.

2. **Parse + sort timestamps** — Parse timestamps, argsort chronologically, compute relative time (seconds from training start).

3. **Chronological split** — First 70% of sorted edges = train, next 15% = val, last 15% = test. Records split indices (train_end_idx, val_end_idx) rather than masks.

4. **Fit edge encoders on training only** — Calls `engineer_edge_features(fit=True)` on the training portion to prevent data leakage.

5. **Transform all edges** — Calls `engineer_edge_features(fit=False)` with the fitted encoders on ALL transactions.

6. **Edge indices** — Map account strings to integer node indices.

7. **Labels** — Extract `is_laundering` as float32.

8. **Pos weight** — From training edges only.

9. **Build TemporalData** — PyG `TemporalData(src, dst, t, msg, y)` where src/dst are node indices, t is relative timestamp, msg is edge features, y is label. All edges in chronological order.

Returns: `TemporalGraphData` wrapping PyG TemporalData + train_end_idx + val_end_idx.

---

## trainer.py → GNNTrainer.train(num_epochs)

Full-batch training loop:

```
for epoch in 1..num_epochs:
    # 1. Train step
    model.train()
    optimizer.zero_grad()
    logits = model(x, edge_index, edge_attr)          # forward on FULL graph
    loss = BCEWithLogitsLoss(logits[train_mask], labels[train_mask])
    loss.backward()
    if grad_clip > 0: clip_grad_norm_(params, grad_clip)
    optimizer.step()

    # 2. Validate
    with no_grad:
        val_logits = model(x, edge_index, edge_attr)
        val_metrics = compute_all_metrics(val_labels, sigmoid(val_logits[val_mask]))

    # 3. LR schedule
    scheduler.step(val_metrics["auc_roc"])             # ReduceLROnPlateau

    # 4. Early stopping
    if val_auc > best_val_auc:
        best_state = deepcopy(model.state_dict())
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= patience: break

# After training:
model.load_state_dict(best_state)
calibrate_threshold(val_labels, val_probs, metric="f1")  # find best F1 threshold on val
```

Key details:
- Full-batch (no mini-batching) — the entire graph fits in CPU memory for HI-Small.
- `pos_weight` moderated by multiplier (e.g. 0.1) — raw pos_weight of 1244 would make the model predict almost everything positive.
- Best model selected by val AUC-ROC, restored before threshold calibration.
- Threshold calibration searches over thresholds to maximize F1 on the validation set.

---

## temporal_trainer.py → TemporalTrainer.train(num_epochs)

Trains on a sequence of snapshots (not a single graph):

**TemporalGCN training (per-snapshot backprop):**
```
for epoch in 1..num_epochs:
    optimizer.zero_grad()
    state = zeros(N, hidden_dim)
    total_loss = 0

    for each training snapshot (0..train_cutoff-1):
        state, logits = model.forward_single_snapshot(x, edge_index, edge_attr, state)
        loss = BCEWithLogitsLoss(logits, edge_label)
        loss.backward()            # per-snapshot backprop
        state = state.detach()     # truncated BPTT (k=1) — no gradient through time
        total_loss += loss

    if grad_clip > 0: clip_grad_norm_(params, grad_clip)
    optimizer.step()
```

**EvolveGCN-H training (grouped backprop via bptt_steps):**
```
for epoch in 1..num_epochs:
    optimizer.zero_grad()
    model.init_weight_states(device)     # reset GRU weight states

    for each training snapshot:
        _, weight_states, logits = model.forward_single_snapshot(x, edge_index, edge_attr, weight_states)
        accumulate loss over bptt_steps snapshots
        every bptt_steps: (loss/bptt_steps).backward(); weight_states.detach()

    if grad_clip > 0: clip_grad_norm_(params, grad_clip)
    optimizer.step()
```

Key difference from static trainer:
- TemporalGCN: GRU evolves per-node hidden states across snapshots. State detached between snapshots (truncated BPTT with k=1).
- EvolveGCN-H: GRU evolves GCN weight matrices. Gradients flow through weight GRU over `bptt_steps` snapshots (default 4) before detaching.
- Validation: runs on the first val snapshot only (for early stopping speed).
- BPTT steps trade-off: more steps = longer gradient horizon but more memory and potential instability.

---

## tgn_trainer.py → TGNTrainer.train(num_epochs)

Continuous-time training — processes edges in batches, chronologically ordered:

```
for epoch in 1..num_epochs:
    model.train()
    model.reset_memory()                    # zero all per-node memories

    # Shuffle batch ORDER (not edges within batches)
    batch_starts = [0, 2048, 4096, ...]
    perm = randperm(len(batch_starts))

    for each batch_start in permuted order:
        src, dst, t, msg, y = data[batch_start : batch_start+2048]

        optimizer.zero_grad()

        # Forward with OLD memory (before this batch)
        logits = model.compute_edge_logits(src, dst, t, msg, update_memory=True)

        loss = BCEWithLogitsLoss(logits, y)
        loss.backward()

        if grad_clip > 0: clip_grad_norm_(params, grad_clip)   # CRITICAL: must be 0 for TGN
        optimizer.step()

        model.detach_memory()               # detach for truncated BPTT

    # Validation (no memory update, no gradients)
    with no_grad:
        model.reset_memory()
        for each val batch:
            logits = model.compute_edge_logits(src, dst, t, msg, update_memory=False)
            # Manually update memory (no gradients)
            model.memory.update_and_embed(src, dst, t, msg)
        val_metrics = compute_all_metrics(y_true, y_prob)

    # Early stopping + LR schedule (same pattern as other trainers)

# After training:
model.load_state_dict(best_state)
calibrate_threshold on validation edges
```

Critical design decisions:
1. **Batch order shuffled, not edges** — edges within a batch stay chronologically ordered. Only the order in which batches are processed is shuffled. This preserves temporal causality within each batch.
2. **Old memory for predictions** — compute_edge_logits captures old memory BEFORE updating, so the model can't cheat by peeking at current-batch edge features in memory.
3. **No gradient clipping** — pos_weight=12.4 creates large minority-class gradients. Gradient clipping destroys these gradients, preventing the model from learning to detect laundering.
4. **detach_memory() after each batch** — truncated BPTT. Without this, gradients would flow through the entire EMA chain across all batches, which is computationally infeasible.

---

## tgn_model.py → EMAMemory

Custom EMA memory replacing PyG's TGNMemory (which had dtype bugs with float timestamps vs Long `last_update`).

```
class EMAMemory(nn.Module):
    # Buffers (not Parameters — no direct gradient updates):
    memory:      (N, memory_dim) float32    — per-node memory vectors
    last_update: (N,)           long        — last timestamp each node was updated

    # Trainable:
    msg_proj: Linear(edge_dim → memory_dim) → ReLU → Linear(memory_dim → memory_dim)
```

**update_and_embed(src, dst, t, raw_msg):**
1. Find unique nodes in this batch: `n_id = unique(cat(src, dst))`
2. Map global src/dst indices to local positions within n_id.
3. Project raw edge features through trainable `msg_proj`: `msgs = msg_proj(raw_msg)` — shape (E, memory_dim).
4. Aggregate messages per node (mean of all messages for that node in this batch): scatter_add msgs to each node's position, divide by count.
5. EMA update: `new_memory = beta * old_memory + (1-beta) * agg_msg`.
6. **Detach** before writing to buffer: `self.memory[n_id] = new_memory.detach()`.
7. Update last_update timestamps (max of old and current batch time).

**Why detach() matters:**
- Without detach: gradients would flow through `memory_t3 → memory_t2 → memory_t1 → ...` back through the entire EMA chain. This is truncated BPTT — we only backprop through the current batch.
- The message projector (`msg_proj`) STILL gets gradients because its output (`projected_msg`) is fed directly to the EdgeClassifier (alongside node embeddings and raw edge features). The gradient path is: `loss → edge_head → projected_msg → msg_proj`.
- The EMA path (`msg_proj → agg_msg → new_memory → next batch`) is gradient-blocked by detach(). This is intentional — training through the EMA chain would require backprop through thousands of batches.

---

## TGNModel.compute_edge_logits() — forward vs predict, and the data leakage bug

**How it works (fixed version):**

```
def compute_edge_logits(src, dst, t, msg, update_memory=True):
    n_id = unique(cat(src, dst))

    # 1. CAPTURE OLD STATE before any update
    old_memory = self.memory[n_id]
    old_last_update = self.last_update[n_id]

    # 2. Project messages (TRAINABLE — gradients flow through here)
    projected_msg = self.memory.msg_proj(msg)

    # 3. Update memory IF requested (detached)
    if update_memory:
        agg_msg = aggregate projected_msg per node (scatter_add / count)
        new_memory = beta * old_memory + (1-beta) * agg_msg
        self.memory[n_id] = new_memory.detach()    # detach blocks EMA gradient path

    # 4. Time deltas from OLD last_update
    delta_t = t - old_last_update
    t_enc = TimeEncoder(delta_t)                    # sinusoidal time features

    # 5. Node embeddings from OLD memory (no leakage)
    z_src = node_proj(cat(old_memory_src, t_enc_src))
    z_dst = node_proj(cat(old_memory_dst, t_enc_dst))

    # 6. Edge logits — projected_msg in input gives msg_proj its gradients
    return edge_head(z_src, z_dst, msg, t_enc_src, projected_msg)
```

**The data leakage bug (now fixed):**

During training, the old code called `model.forward()` which:
1. Updated memory WITH current batch messages.
2. Used the UPDATED memory to compute node embeddings.
3. Made predictions from node embeddings that had already "seen" the current batch.

This meant the model could cheat: "I just updated this account's memory with a suspicious-looking message, and now I'm being asked to predict whether this transaction is suspicious — of course it is, the memory already contains the answer."

During evaluation, `model.predict()` did NOT update memory, so memory only reflected past interactions. This created a train/eval gap — the model learned to rely on the cheat signal during training, but the signal disappeared during evaluation.

**The fix:** Always predict from OLD memory. The current batch's messages influence predictions ONLY through `projected_msg` in the edge classifier input (which is legitimate — the edge features are part of the prediction input). Memory is updated AFTER capturing old state, and the update is detached. This makes training and evaluation consistent.

---

## Why `--grad_clip 0` is critical for TGN

With `pos_weight ≈ 12.4` and class prevalence 0.1%:

- A minority-class (laundering) edge contributes `pos_weight × BCE_loss` to the gradient.
- A majority-class (legitimate) edge contributes `1.0 × BCE_loss`.
- Ratio: minority gradients are ~12.4× larger than majority gradients.

Gradient clipping at 1.0 clips the L2 norm of the entire gradient vector. Since minority-class edges produce large gradients, clipping disproportionately destroys the minority-class learning signal. The model essentially never learns to detect laundering because its gradients are clipped away.

For static GNNs, `pos_weight_mult=0.1` brings effective pos_weight down to ~124, so clipping at 1.0 is tolerable. For TGN, the effective pos_weight is ~12.4 (with multiplier 0.01), and the per-batch gradient structure is different (smaller batches, more stochastic), making clipping especially harmful.
