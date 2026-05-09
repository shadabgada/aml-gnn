"""Temporal GNN architectures for AML edge classification.

Two variants are implemented:

  TemporalGCN — GRU-evolved per-node hidden states across snapshots.
      A GCN backbone processes each snapshot; between snapshots, a GRU
      evolves the latent representation of each account. This directly
      models how account behaviour changes over time (e.g., an account
      transitioning from legitimate to layering behaviour).

  EvolveGCN-H — Weight-space evolution via GRU (Pareja et al., 2020).
      The GCN weight matrices themselves are evolved across time using
      a GRU. A low-rank bottleneck keeps the recurrent state tractable.

Both share the same edge classifier head as the static GNNs for fair
comparison.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


# ---------------------------------------------------------------------------
# TemporalGCN — GRU-evolved node states
# ---------------------------------------------------------------------------


class TemporalGCN(nn.Module):
    """GCN with per-node GRU state evolution across temporal snapshots.

    For each snapshot t:
      1. GCN computes structural node embeddings h_t from (x_t, edge_index_t).
      2. Per-node hidden state is updated: s_t = GRU(h_t, s_{t-1}).
      3. Edge logits are computed from [s_t[src] || s_t[dst] || edge_attr_t].

    The GRU allows the model to accumulate behavioural history: an account
    that was legitimate in early snapshots but starts receiving structured
    deposits in later snapshots will have its state updated accordingly.

    Args:
        node_dim: Input node feature dimensionality.
        edge_dim: Input edge feature dimensionality.
        hidden_dim: Hidden dimension for GCN and GRU.
        num_layers: Number of GCNConv layers per snapshot.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout

        act = nn.ReLU(inplace=True)

        # --- GCN backbone (shared across snapshots) -----------------------
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        in_dim = node_dim
        for i in range(num_layers):
            self.convs.append(GCNConv(in_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
            in_dim = hidden_dim

        self.dropout = nn.Dropout(dropout)
        self.act = act

        # --- GRU for temporal state evolution ------------------------------
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

        # --- Edge classifier head (same as static GNNs) --------------------
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

    def forward_single_snapshot(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        node_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process one snapshot, updating node states.

        Args:
            x: Node features for this snapshot (N, node_dim).
            edge_index: Edge indices (2, E).
            edge_attr: Edge features (E, edge_dim).
            node_state: Previous node hidden states (N, hidden_dim).

        Returns:
            (new_node_state, edge_logits)
        """
        # GCN forward
        h = x
        for i in range(self.num_layers):
            h = self.convs[i](h, edge_index)
            h = self.bns[i](h)
            h = self.act(h)
            h = self.dropout(h)

        # GRU update
        new_state = self.gru(h, node_state)

        # Edge logits
        src, dst = edge_index[0], edge_index[1]
        h_src = new_state[src]
        h_dst = new_state[dst]
        edge_input = torch.cat([h_src, h_dst, edge_attr], dim=-1)
        logits = self.edge_head(edge_input).squeeze(-1)

        return new_state, logits

    def forward_sequence(
        self,
        snapshots: list,
        initial_state: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """Process a full sequence of snapshots.

        Args:
            snapshots: List of PyG Data objects with (x, edge_index, edge_attr).
            initial_state: Initial node states (N, hidden_dim). Zeros if None.

        Returns:
            List of edge logits, one tensor per snapshot.
        """
        num_nodes = snapshots[0].x.shape[0]
        device = snapshots[0].x.device

        if initial_state is None:
            state = torch.zeros(num_nodes, self.hidden_dim, device=device)
        else:
            state = initial_state

        all_logits = []
        for snap in snapshots:
            state, logits = self.forward_single_snapshot(
                snap.x, snap.edge_index, snap.edge_attr, state,
            )
            all_logits.append(logits)

        return all_logits

    def forward(self, snapshots: list) -> list[torch.Tensor]:
        """Convenience wrapper — processes all snapshots."""
        return self.forward_sequence(snapshots)


# ---------------------------------------------------------------------------
# EvolveGCN-H — Weight-space evolution (Pareja et al., 2020)
# ---------------------------------------------------------------------------


class EvolveGCNH(nn.Module):
    """EvolveGCN-H: GCN weight matrices evolved by a GRU across time.

    Implements the approach from Pareja et al. (2020) with a low-rank
    bottleneck for tractability. Instead of evolving the full weight
    matrix (which would require a GRU hidden state of size in_dim × out_dim),
    we evolve a low-rank decomposition:

        W_t = W_base + A_t @ B_t

    where A_t ∈ R^{out_dim × rank} and B_t ∈ R^{rank × in_dim} are the
    low-rank factors evolved by the GRU.

    Args:
        node_dim: Input node feature dimensionality.
        edge_dim: Input edge feature dimensionality.
        hidden_dim: Hidden dimension throughout.
        num_layers: Number of GCNConv layers.
        rank: Rank of the low-rank weight adaptation (4–16).
        dropout: Dropout probability.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        rank: int = 8,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rank = rank
        self.dropout_rate = dropout

        act = nn.ReLU(inplace=True)

        # --- Base GCN weights (static, learned) ---------------------------
        self.base_weights = nn.ParameterList()
        self.base_biases = nn.ParameterList()
        self.bns = nn.ModuleList()

        layer_dims = [(node_dim, hidden_dim)] + [
            (hidden_dim, hidden_dim) for _ in range(num_layers - 1)
        ]

        for in_d, out_d in layer_dims:
            self.base_weights.append(nn.Parameter(torch.empty(in_d, out_d)))
            self.base_biases.append(nn.Parameter(torch.zeros(out_d)))
            self.bns.append(nn.BatchNorm1d(out_d))

        # --- GRU cells for weight evolution (one per layer) ---------------
        # Each GRU evolves the flattened low-rank factors A and B.
        # A ∈ R^{out×rank}, B ∈ R^{rank×in} → total params = rank*(in+out)
        self.weight_grus = nn.ModuleList()
        self.gru_projections = nn.ModuleList()  # map node summary → GRU input
        for in_d, out_d in layer_dims:
            gru_dim = rank * (in_d + out_d)
            self.weight_grus.append(nn.GRUCell(gru_dim, gru_dim))
            # Project from hidden_dim to GRU input dim
            self.gru_projections.append(nn.Linear(hidden_dim, gru_dim))

        self.dropout = nn.Dropout(dropout)
        self.act = act

        # --- Edge classifier head ------------------------------------------
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

        # Persistent weight states (initialised on first forward or explicitly)
        self._weight_states: list[torch.Tensor] = []
        self._states_initialised = False

        self._init_weights()

    def _init_weights(self):
        for p in self.base_weights:
            nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def init_weight_states(self, device: torch.device) -> None:
        """(Re-)initialise weight GRU states to zeros. Call at start of each epoch."""
        self._weight_states = []
        for l in range(self.num_layers):
            in_d = self.base_weights[l].shape[0]
            out_d = self.base_weights[l].shape[1]
            gru_dim = self.rank * (in_d + out_d)
            self._weight_states.append(torch.zeros(gru_dim, device=device))
        self._states_initialised = True

    def _low_rank_weights(
        self, layer_idx: int, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Unflatten GRU state into low-rank factors A and B for a layer."""
        in_d, out_d = (
            self.base_weights[layer_idx].shape[0],
            self.base_weights[layer_idx].shape[1],
        )
        # state is flat: [A_flat || B_flat]
        a_size = out_d * self.rank
        A_flat = state[:a_size]
        B_flat = state[a_size:]
        A = A_flat.view(out_d, self.rank)
        B = B_flat.view(self.rank, in_d)
        return A, B

    def _compute_weight(self, layer_idx: int, state: torch.Tensor) -> torch.Tensor:
        """W = W_base + A @ B (low-rank adaptation)."""
        A, B = self._low_rank_weights(layer_idx, state)
        delta = A @ B  # (out_d, in_d)
        return self.base_weights[layer_idx] + delta.T  # (in_d, out_d)

    def _manual_gcn_conv(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        """Manual GCN convolution using provided weight and bias.

        GCN: h_i = sum_{j in N(i) U {i}} (1/sqrt(deg_i*deg_j)) * x_j @ W
        """
        from torch_geometric.utils import add_self_loops, degree
        edge_index_loop, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        row, col = edge_index_loop
        deg = degree(row, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Message passing
        out = x @ weight  # (N, out_dim)
        out = out[col] * norm.unsqueeze(-1)  # (E', out_dim)
        result = torch.zeros(x.size(0), weight.size(1), device=x.device, dtype=x.dtype)
        result = result.scatter_add(0, row.unsqueeze(-1).expand_as(out), out)
        result = result + bias
        return result

    def forward_single_snapshot(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        weight_states: list[torch.Tensor],
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        """Process one snapshot, evolving weights and computing logits.

        Args:
            x: Node features (N, node_dim).
            edge_index: Edge indices (2, E).
            edge_attr: Edge features (E, edge_dim).
            weight_states: List of GRU hidden states for each layer's weights.

        Returns:
            (node_embeddings, new_weight_states, edge_logits)
        """
        h = x
        new_states = []

        for l in range(self.num_layers):
            # Current weight from evolved state
            W = self._compute_weight(l, weight_states[l])
            b = self.base_biases[l]
            h = self._manual_gcn_conv(h, edge_index, W, b)
            h = self.bns[l](h)
            h = self.act(h)
            h = self.dropout(h)

            # Evolve weight state — input is mean node embedding
            node_summary = h.mean(dim=0)  # (hidden_dim,)
            gru_in = self.gru_projections[l](node_summary)  # → gru_dim
            new_state = self.weight_grus[l](gru_in, weight_states[l])
            new_states.append(new_state)

        # Edge logits
        src, dst = edge_index[0], edge_index[1]
        h_src = h[src]
        h_dst = h[dst]
        edge_input = torch.cat([h_src, h_dst, edge_attr], dim=-1)
        logits = self.edge_head(edge_input).squeeze(-1)

        return h, new_states, logits

    def forward_sequence(
        self,
        snapshots: list,
    ) -> list[torch.Tensor]:
        """Process a full sequence of snapshots.

        Weight states persist across calls. Call init_weight_states()
        to reset at the start of each epoch.

        Args:
            snapshots: List of PyG Data objects.

        Returns:
            List of edge logits, one tensor per snapshot.
        """
        device = snapshots[0].x.device

        if not self._states_initialised:
            self.init_weight_states(device)

        all_logits = []
        for snap in snapshots:
            _, self._weight_states, logits = self.forward_single_snapshot(
                snap.x, snap.edge_index, snap.edge_attr, self._weight_states,
            )
            all_logits.append(logits)

        return all_logits

    def forward(self, snapshots: list) -> list[torch.Tensor]:
        return self.forward_sequence(snapshots)
