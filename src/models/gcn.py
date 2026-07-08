"""Graph Convolutional Network (Kipf & Welling, 2017) for AML edge classification.

Architecture:
  1. GCNConv layers compute node embeddings from node features + graph structure.
  2. An edge classifier MLP takes [h_src || h_dst || edge_attr] → logit.

This is the foundational static GNN baseline — symmetric neighbourhood
aggregation without learned edge weighting (cf. GAT) or inductive sampling
(cf. GraphSAGE).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCNEdgeClassifier(nn.Module):
    """Two-layer GCN with an edge-level MLP classifier head.

    Args:
        node_dim: Dimensionality of input node features.
        edge_dim: Dimensionality of input edge features.
        hidden_dim: Hidden dimension throughout the network.
        dropout: Dropout probability applied after each GCN layer.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        activation: str = "relu",
    ):
        """Construct the GCN edge classifier (GCNConv layers plus an edge-MLP head); see the class docstring for arguments."""
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout

        act = {"relu": nn.ReLU(inplace=True), "elu": nn.ELU(inplace=True),
               "leaky_relu": nn.LeakyReLU(inplace=True)}[activation]

        # ---- GCN layers -------------------------------------------------
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        in_dim = node_dim
        for i in range(num_layers):
            self.convs.append(GCNConv(in_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
            in_dim = hidden_dim

        self.dropout = nn.Dropout(dropout)
        self.act = act

        # ---- Edge classifier head ---------------------------------------
        # Input: [h_src || h_dst || edge_attr]
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
        """Xavier-initialise all Linear layers and zero their biases."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Compute node embeddings via graph convolution over the full graph."""
        h = x
        for i in range(self.num_layers):
            h = self.convs[i](h, edge_index)
            h = self.bns[i](h)
            h = self.act(h)
            h = self.dropout(h)
        return h

    def decode(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        idx: torch.Tensor | None = None,
        chunk_size: int = 500_000,
    ) -> torch.Tensor:
        """Score edges from node embeddings.

        If idx is given, only those edges are scored (used for memory-bounded
        mini-batch training). Otherwise all edges are scored, chunked to bound
        peak memory during evaluation.
        """
        src, dst = edge_index[0], edge_index[1]
        if idx is not None:
            e_in = torch.cat([h[src[idx]], h[dst[idx]], edge_attr[idx]], dim=-1)
            return self.edge_head(e_in).squeeze(-1)
        num_edges = edge_index.shape[1]
        if num_edges <= chunk_size:
            e_in = torch.cat([h[src], h[dst], edge_attr], dim=-1)
            return self.edge_head(e_in).squeeze(-1)
        out = []
        for start in range(0, num_edges, chunk_size):
            end = min(start + chunk_size, num_edges)
            sl = slice(start, end)
            e_in = torch.cat([h[src[sl]], h[dst[sl]], edge_attr[sl]], dim=-1)
            out.append(self.edge_head(e_in).squeeze(-1))
        return torch.cat(out, dim=0)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """Return edge logits for all edges (encode then decode)."""
        return self.decode(self.encode(x, edge_index), edge_index, edge_attr)
