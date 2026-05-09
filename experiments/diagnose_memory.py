"""Diagnose whether EMA memory accumulation causes TGN overfitting.

Tests 3 beta values:
- beta=1.0: Frozen memory (no accumulation, time encoding only)
- beta=0.5: Fast memory decay
- beta=0.85: Default slow EMA

Each variant trains 5 epochs. If all betas show the same overfitting pattern,
memory is not the issue. If beta=1.0 works better, memory is harmful.
"""

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
pw = graph_data.pos_weight * 0.01
print(f"pos_weight: {graph_data.pos_weight:.1f} * 0.01 = {pw:.1f}")

for beta in [0.85]:
    print(f'\n=== beta={beta} ===')
    model = TGNModel(
        num_nodes=graph_data.num_nodes,
        edge_dim=graph_data.num_edge_features,
        memory_dim=128, time_dim=16, hidden_dim=128, beta=beta,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)

    for epoch in range(1, 6):
        # Train
        model.train()
        model.reset_memory()
        total_loss = 0.0
        total_edges = 0
        batch_starts = list(range(0, train_end, 2048))
        perm = torch.randperm(len(batch_starts))
        for idx in perm:
            s = batch_starts[idx]
            e = min(s + 2048, train_end)
            optimizer.zero_grad()
            src, dst, t, msg, y = data.src[s:e], data.dst[s:e], data.t[s:e], data.msg[s:e], data.y[s:e]
            logits = model.compute_edge_logits(src, dst, t, msg, update_memory=True)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            model.detach_memory()
            total_loss += loss.item() * (e - s)
            total_edges += (e - s)

        # Eval
        model.eval()
        model.reset_memory()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for lo in range(val_start, val_end, 2048):
                hi = min(lo + 2048, val_end)
                src, dst, t, msg, y = data.src[lo:hi], data.dst[lo:hi], data.t[lo:hi], data.msg[lo:hi], data.y[lo:hi]
                logits = model.compute_edge_logits(src, dst, t, msg, update_memory=False)
                all_probs.append(torch.sigmoid(logits).cpu().numpy())
                all_labels.append(y.cpu().numpy())
                new_memory, _, n_id = model.memory.update_and_embed(src, dst, t, msg)

        metrics = compute_all_metrics(
            np.concatenate(all_labels), np.concatenate(all_probs),
        )
        print(f'  E{epoch}: loss={total_loss/total_edges:.4f} '
              f'val_auc_roc={metrics["auc_roc"]:.4f} '
              f'val_auc_pr={metrics["auc_pr"]:.4f}')

print('\nDone')
