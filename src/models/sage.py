"""GraphSAGE (Hamilton et al., 2017) for AML edge classification.

GraphSAGE learns inductive node embeddings by sampling and aggregating
features from a node's local neighbourhood. Unlike GCN (which uses
symmetric normalised aggregation) and GAT (which learns attention weights),
GraphSAGE uses configurable aggregation functions (mean, max, pool) and
was originally designed for inductive settings where the graph structure
may change at test time.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv


class GraphSAGEEdgeClassifier(nn.Module):
    """Multi-layer GraphSAGE with an edge-level MLP classifier head.

    Uses concatenation of self-representation and aggregated neighbour
    representation (the default SAGE behaviour), with the same edge
    classifier as GCN and GAT for fair comparison.

    Args:
        node_dim: Dimensionality of input node features.
        edge_dim: Dimensionality of input edge features.
        hidden_dim: Hidden dimension.
        num_layers: Number of SAGEConv layers.
        aggregator: Aggregation function — "mean", "max", "pool", or "lstm".
        dropout: Dropout probability.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        aggregator: str = "mean",
        dropout: float = 0.3,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.aggregator = aggregator

        act = nn.ReLU(inplace=True)

        # ---- SAGE layers -------------------------------------------------
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        in_dim = node_dim
        for i in range(num_layers):
            self.convs.append(
                SAGEConv(in_dim, hidden_dim, aggr=aggregator, normalize=True)
            )
            self.bns.append(nn.BatchNorm1d(hidden_dim))
            in_dim = hidden_dim

        self.dropout = nn.Dropout(dropout)
        self.act = act

        # ---- Edge classifier head (same architecture as GCN/GAT) ----------
        head_in = hidden_dim * 2 + edge_dim
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

        logits_list = []
        for start in range(0, num_edges, chunk_size):
            end = min(start + chunk_size, num_edges)
            idx = torch.arange(start, end, device=x.device)
            e_in = torch.cat([
                h[src[idx]], h[dst[idx]], edge_attr[idx],
            ], dim=-1)
            logits_list.append(self.edge_head(e_in).squeeze(-1))

        return torch.cat(logits_list, dim=0)
