"""TGN-style model with EMA memory — no train/eval data leakage.

Predictions always use OLD memory (before current batch). The message
projector is trained through the edge classifier (which receives
projected messages as input), not through the memory update path.

This eliminates the key overfitting mechanism: during training the
model can no longer "cheat" by peeking at current-batch edge features
in memory, since memory reflects only past interactions.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn.models.tgn import TimeEncoder


class EMAMemory(nn.Module):
    """Per-node EMA memory with trainable message projection.

    Updates happen IN the forward pass so the projector receives gradients.
    The updated memory is detached after each forward for truncated BPTT.
    """

    def __init__(self, num_nodes: int, raw_msg_dim: int, memory_dim: int,
                 beta: float = 0.85):
        super().__init__()
        self.num_nodes = num_nodes
        self.memory_dim = memory_dim
        self.beta = beta

        # Trainable message projector: edge features → memory space
        self.msg_proj = nn.Sequential(
            nn.Linear(raw_msg_dim, memory_dim),
            nn.ReLU(inplace=True),
            nn.Linear(memory_dim, memory_dim),
        )

        # Buffers (not parameters — no direct gradient updates)
        self.register_buffer('memory', torch.zeros(num_nodes, memory_dim))
        self.register_buffer('last_update',
                             torch.zeros(num_nodes, dtype=torch.long))


    def forward(self, n_id: Tensor, t: Tensor | None = None
                ) -> tuple[Tensor, Tensor]:
        """Read current memory and last_update for nodes."""
        return self.memory[n_id], self.last_update[n_id]

    def update_and_embed(
        self,
        src: Tensor,
        dst: Tensor,
        t: Tensor,
        raw_msg: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Update memory with current interactions, return updated embeddings.

        This runs INSIDE the computation graph. Gradients flow through
        msg_proj → updated_memory → node embeddings → edge logits → loss.

        Returns:
            mem_updated: (P, memory_dim) — updated memory for all unique nodes
            last_t: (P,) — updated last_update timestamps
            n_id: (P,) — unique node indices
        """
        n_id = torch.cat([src, dst]).unique()
        device = self.memory.device

        # Map edges to local indices within n_id
        local_map = {int(n): i for i, n in enumerate(n_id.tolist())}
        src_local = torch.tensor([local_map[int(s)] for s in src], device=device)
        dst_local = torch.tensor([local_map[int(d)] for d in dst], device=device)

        # Project messages (TRAINABLE — gradients from loss reach here)
        msgs = self.msg_proj(raw_msg)  # (E, memory_dim)

        # Aggregate messages per node (mean of messages in this batch)
        count = torch.zeros(len(n_id), device=device).scatter_add(
            0, src_local, torch.ones_like(src_local, dtype=torch.float))
        count = count.scatter_add(
            0, dst_local, torch.ones_like(dst_local, dtype=torch.float))
        count = count.clamp(min=1)

        agg_msg = torch.zeros(len(n_id), self.memory_dim, device=device)
        agg_msg = agg_msg.scatter_add(0, src_local.unsqueeze(-1).expand(-1, self.memory_dim), msgs)
        agg_msg = agg_msg.scatter_add(0, dst_local.unsqueeze(-1).expand(-1, self.memory_dim), msgs)
        agg_msg = agg_msg / count.unsqueeze(-1)

        # EMA update: m_new = beta * m_old + (1-beta) * msg_agg
        old_memory = self.memory[n_id]
        new_memory = self.beta * old_memory + (1 - self.beta) * agg_msg

        # Update buffers (detach so next batch doesn't BPTT through all history)
        self.memory[n_id] = new_memory.detach()
        # Per-node max timestamp
        batch_max = t.max().long()
        self.last_update[n_id] = torch.max(
            self.last_update[n_id],
            batch_max.expand(len(n_id)),
        )

        return new_memory, self.last_update[n_id], n_id


class TGNEdgeClassifier(nn.Module):
    """MLP head for edge-level binary classification.

    Takes node embeddings, raw edge features, time encoding, and the
    projected message (from EMAMemory.msg_proj) so the message projector
    receives gradients even when predictions use old/unupdated memory.
    """

    def __init__(self, node_embed_dim: int, edge_dim: int, time_dim: int,
                 msg_proj_dim: int = 128, hidden_dim: int = 128,
                 dropout: float = 0.3):
        super().__init__()
        in_dim = node_embed_dim * 2 + edge_dim + time_dim + msg_proj_dim
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, z_src: Tensor, z_dst: Tensor, edge_attr: Tensor,
                t_enc_src: Tensor, projected_msg: Tensor) -> Tensor:
        features = torch.cat(
            [z_src, z_dst, edge_attr, t_enc_src, projected_msg], dim=-1,
        )
        return self.head(features).squeeze(-1)


class TGNModel(nn.Module):
    """Temporal Graph Network for AML edge classification.

    Predictions always use OLD per-node memory (before current batch),
    eliminating train/eval data leakage. The message projector is trained
    through the edge classifier input (projected_msg), not through memory.

    Args:
        num_nodes: Total number of accounts (nodes).
        edge_dim: Dimensionality of edge features (msg).
        memory_dim: Dimensionality of per-node memory vectors.
        time_dim: Dimensionality of time encoding.
        hidden_dim: Hidden dimension for projections and classifier.
        dropout: Dropout probability.
        beta: EMA decay rate for memory updates (0=no memory, 1=frozen).
    """

    def __init__(
        self,
        num_nodes: int,
        edge_dim: int,
        memory_dim: int = 128,
        time_dim: int = 16,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        beta: float = 0.85,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.memory_dim = memory_dim
        self.time_dim = time_dim

        # Time encoder: maps scalar Δt → time_dim vector
        self.time_enc = TimeEncoder(time_dim)

        # Per-node EMA memory (replaces PyG TGNMemory)
        self.memory = EMAMemory(num_nodes, edge_dim, memory_dim, beta)

        # Node embedding: project [memory || time_enc(Δt)] → hidden_dim
        self.node_proj = nn.Sequential(
            nn.Linear(memory_dim + time_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Edge classifier: [z_src || z_dst || edge_attr || t_enc_src || proj_msg] → logit
        self.edge_head = TGNEdgeClassifier(
            node_embed_dim=hidden_dim,
            edge_dim=edge_dim,
            time_dim=time_dim,
            msg_proj_dim=memory_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def reset_memory(self):
        """Reset all node memories to zero and last_update to zero."""
        self.memory.memory.zero_()
        self.memory.last_update.zero_()

    def detach_memory(self):
        """Detach memory buffer from computation graph."""
        self.memory.memory.detach_()

    def compute_messages(self, edge_attr: Tensor) -> Tensor:
        """Project raw edge features to message vectors (for update_state)."""
        return self.memory.msg_proj(edge_attr)

    def compute_edge_logits(
        self,
        src: Tensor,
        dst: Tensor,
        t: Tensor,
        msg: Tensor,
        update_memory: bool = True,
    ) -> Tensor:
        """Compute laundering logits for a batch of edges.

        Predictions ALWAYS use OLD memory (before current batch). This
        eliminates train/eval data leakage — during training the model
        can't cheat by peeking at its own edge features in memory.

        The message projector (msg_proj) stays trainable because its
        output is fed directly to the edge classifier alongside node
        embeddings and raw edge features.

        Memory is updated AFTER capturing old state for use in the next
        batch (detached for truncated BPTT).
        """
        n_id = torch.cat([src, dst]).unique()
        device = src.device

        # Map edges to local indices within n_id
        local_map = {int(n): i for i, n in enumerate(n_id.tolist())}
        src_local = torch.tensor(
            [local_map[int(s)] for s in src], device=device)
        dst_local = torch.tensor(
            [local_map[int(d)] for d in dst], device=device)

        # --- Capture OLD state BEFORE any update ---
        old_memory_all, old_last_update = self.memory(n_id)
        old_memory_src = old_memory_all[src_local]
        old_memory_dst = old_memory_all[dst_local]
        old_last_update_src = old_last_update[src_local]
        old_last_update_dst = old_last_update[dst_local]

        # --- Project messages (always, for both prediction and memory) ---
        projected_msg = self.memory.msg_proj(msg)  # (E, memory_dim) — TRAINABLE

        # --- Update memory if requested (detached for truncated BPTT) ---
        if update_memory:
            # Aggregate messages per node
            count = torch.zeros(len(n_id), device=device).scatter_add(
                0, src_local, torch.ones_like(src_local, dtype=torch.float))
            count = count.scatter_add(
                0, dst_local, torch.ones_like(dst_local, dtype=torch.float))
            count = count.clamp(min=1)

            agg_msg = torch.zeros(len(n_id), self.memory_dim, device=device)
            agg_msg = agg_msg.scatter_add(
                0, src_local.unsqueeze(-1).expand(-1, self.memory_dim),
                projected_msg)
            agg_msg = agg_msg.scatter_add(
                0, dst_local.unsqueeze(-1).expand(-1, self.memory_dim),
                projected_msg)
            agg_msg = agg_msg / count.unsqueeze(-1)

            # EMA update (detached so next batch doesn't BPTT through all history)
            new_mem = (self.memory.beta * old_memory_all
                       + (1 - self.memory.beta) * agg_msg)
            self.memory.memory[n_id] = new_mem.detach()
            batch_max = t.max().long()
            self.memory.last_update[n_id] = torch.max(
                self.memory.last_update[n_id],
                batch_max.expand(len(n_id)),
            )

        # --- Time deltas from OLD last_update ---
        delta_t_src = t - old_last_update_src.float()
        delta_t_dst = t - old_last_update_dst.float()

        t_enc_src = self.time_enc(delta_t_src)
        t_enc_dst = self.time_enc(delta_t_dst)

        # --- Node embeddings from OLD memory (no leakage) ---
        z_src = self.node_proj(
            torch.cat([old_memory_src, t_enc_src], dim=-1))
        z_dst = self.node_proj(
            torch.cat([old_memory_dst, t_enc_dst], dim=-1))

        # projected_msg in classifier input → gradients flow to msg_proj
        return self.edge_head(z_src, z_dst, msg, t_enc_src, projected_msg)
