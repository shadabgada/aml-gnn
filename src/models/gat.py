"""Graph Attention Network (Veličković et al., 2018) for AML edge classification.

GAT extends GCN by learning attention weights over neighbours, so the model
can focus on the most relevant transactions for each account. This is
especially relevant for AML: a laundering account's behaviour is better
understood through its most suspicious counterparties rather than a uniform
average of all transactions.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv


class GATEdgeClassifier(nn.Module):
    """Multi-layer GAT with an edge-level MLP classifier head.

    Uses multi-head attention (concat for intermediate layers, average for
    the final layer) and the same edge classifier architecture as GCN for
    fair comparison.

    Args:
        node_dim: Dimensionality of input node features.
        edge_dim: Dimensionality of input edge features.
        hidden_dim: Hidden dimension per attention head.
        num_layers: Number of GATConv layers.
        heads: Number of attention heads per layer.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.heads = heads
        self.dropout_rate = dropout

        act = nn.ELU(inplace=True)

        # ---- GAT layers --------------------------------------------------
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        in_dim = node_dim
        for i in range(num_layers):
            is_last = (i == num_layers - 1)
            if is_last:
                # Final layer: average heads → hidden_dim output
                self.convs.append(
                    GATConv(in_dim, hidden_dim, heads=heads, concat=False,
                            dropout=dropout, add_self_loops=False)
                )
                out_dim = hidden_dim
            else:
                # Intermediate: concat heads → hidden_dim * heads output
                self.convs.append(
                    GATConv(in_dim, hidden_dim, heads=heads, concat=True,
                            dropout=dropout, add_self_loops=False)
                )
                out_dim = hidden_dim * heads
            self.bns.append(nn.BatchNorm1d(out_dim))
            in_dim = out_dim

        self.dropout = nn.Dropout(dropout)
        self.act = act

        # ---- Edge classifier head (same architecture as GCN) --------------
        head_in = in_dim * 2 + edge_dim
        self.edge_head = nn.Sequential(
            nn.Linear(head_in, hidden_dim),
            act,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            act,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        chunk_size: int = 500_000,
    ) -> torch.Tensor:
        """Return edge logits for all edges, computed in chunks to limit memory."""
        h = x
        for i in range(self.num_layers):
            h = self.convs[i](h, edge_index)
            h = self.bns[i](h)
            h = self.act(h)
            h = self.dropout(h)

        src, dst = edge_index[0], edge_index[1]
        num_edges = edge_index.shape[1]

        if num_edges <= chunk_size:
            h_src = h[src]
            h_dst = h[dst]
            edge_input = torch.cat([h_src, h_dst, edge_attr], dim=-1)
            return self.edge_head(edge_input).squeeze(-1)

        # Chunked to avoid OOM on large graphs
        logits_list = []
        for start in range(0, num_edges, chunk_size):
            end = min(start + chunk_size, num_edges)
            idx = torch.arange(start, end, device=x.device)
            e_in = torch.cat([
                h[src[idx]], h[dst[idx]], edge_attr[idx],
            ], dim=-1)
            logits_list.append(self.edge_head(e_in).squeeze(-1))

        return torch.cat(logits_list, dim=0)
