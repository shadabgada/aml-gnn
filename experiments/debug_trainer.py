"""Debug: compare diagnostic loop vs trainer loop on identical model/data."""
import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import numpy as np
from src.data.loader import load_raw_data
from src.data.temporal_data_builder import build_temporal_data
from src.models.tgn_model import TGNModel
from src.utils.config import DataConfig
from src.utils.metrics import compute_all_metrics
from src.training.tgn_trainer import TGNTrainer

torch.set_num_threads(8)
torch.manual_seed(42)
np.random.seed(42)

cfg = DataConfig(dataset_variant='HI-Small')
accounts, transactions = load_raw_data(cfg)
graph_data = build_temporal_data(accounts, transactions, cfg)

data = graph_data.data
train_end = graph_data.train_end_idx
val_start = graph_data.train_end_idx
val_end = graph_data.val_end_idx

# --- Test 1: Direct diagnostic loop (should work) ---
print("=== Test 1: Diagnostic-style training ===")
model1 = TGNModel(
    num_nodes=graph_data.num_nodes,
    edge_dim=graph_data.num_edge_features,
    memory_dim=128, time_dim=16, hidden_dim=128, beta=0.85,
)
pw = graph_data.pos_weight * 0.01
criterion1 = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([pw], device='cpu'))
optimizer1 = torch.optim.AdamW(model1.parameters(), lr=0.003)

model1.train()
model1.reset_memory()
total_loss = 0.0
total_edges = 0
batch_starts = list(range(0, train_end, 2048))
perm = torch.randperm(len(batch_starts))
for idx in perm:
    s = batch_starts[idx]
    e = min(s + 2048, train_end)
    optimizer1.zero_grad()
    src, dst, t, msg, y = data.src[s:e], data.dst[s:e], data.t[s:e], data.msg[s:e], data.y[s:e]
    logits = model1.compute_edge_logits(src, dst, t, msg, update_memory=True)
    loss = criterion1(logits, y)
    loss.backward()
    # torch.nn.utils.clip_grad_norm_(model1.parameters(), 1.0)
    optimizer1.step()
    model1.detach_memory()
    total_loss += loss.item() * (e - s)
    total_edges += (e - s)
print(f"  Train loss: {total_loss/total_edges:.4f}")

# Eval
model1.eval()
model1.reset_memory()
all_probs, all_labels = [], []
with torch.no_grad():
    for lo in range(val_start, val_end, 2048):
        hi = min(lo + 2048, val_end)
        src, dst, t, msg, y = data.src[lo:hi], data.dst[lo:hi], data.t[lo:hi], data.msg[lo:hi], data.y[lo:hi]
        logits = model1.compute_edge_logits(src, dst, t, msg, update_memory=False)
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(y.cpu().numpy())
        new_memory, _, n_id = model1.memory.update_and_embed(src, dst, t, msg)
m1 = compute_all_metrics(np.concatenate(all_labels), np.concatenate(all_probs))
print(f"  Val AUC-ROC: {m1['auc_roc']:.4f}, AUC-PR: {m1['auc_pr']:.4f}")

# --- Test 2: Trainer class (same seed, same model init) ---
print("\n=== Test 2: Trainer-style training ===")
model2 = TGNModel(
    num_nodes=graph_data.num_nodes,
    edge_dim=graph_data.num_edge_features,
    memory_dim=128, time_dim=16, hidden_dim=128, beta=0.85,
)
trainer = TGNTrainer(
    model=model2,
    temporal_data=data,
    train_end_idx=graph_data.train_end_idx,
    val_end_idx=graph_data.val_end_idx,
    pos_weight=graph_data.pos_weight,
    pos_weight_multiplier=0.01,
    batch_size=2048,
    lr=0.003,
    weight_decay=5e-4,
    grad_clip=1.0,
    patience=25,
    device='cpu',
    log_interval=1,
    checkpoint_dir='results/checkpoints',
    checkpoint_interval=100,
)

# Run one epoch manually to compare
train_loss = trainer._train_epoch(1)
print(f"  Train loss: {train_loss:.4f}")

val_metrics = trainer._evaluate_split(trainer.val_edges)
print(f"  Val AUC-ROC: {val_metrics['auc_roc']:.4f}, AUC-PR: {val_metrics['auc_pr']:.4f}")

# Check if data is different
print(f"\n=== Data comparison ===")
print(f"  data.src[:5] = {data.src[:5].tolist()}")
print(f"  trainer.data.src[:5] = {trainer.data.src[:5].tolist()}")
print(f"  data.src is trainer.data.src? {data.src is trainer.data.src}")
print(f"  data.src.equal(trainer.data.src)? {data.src.equal(trainer.data.src)}")

# Check model parameters match initially (before any training)
print(f"\n=== Initial weight comparison ===")
print(f"  model1.node_proj[0].weight sum: {model1.node_proj[0].weight.sum().item():.4f}")
# model2 was already trained, so weights differ now
