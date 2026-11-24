"""Graph deep-learning models. Both take the preprocessed input windows X (B, 12, N, 7) and return speed forecasts
(B, 12, N) in z-score units (the trainer turns them back into mph before the masked loss).

STGAT (ours, the brief's "Graph Attention Network"): a GRU encodes each sensor's 12-step history (temporal), two GATv2
layers let each sensor attend to its road neighbours' encodings (spatial), and a linear head outputs the 12 horizons as
a change from the last observed speed.

DCRNN (Li et al., ICLR 2018, the brief's "diffusion graph model"): diffusion convolution over the directed graph
(forward and backward random walks, K hops) inside a GRU encoder-decoder, decoded step by step with scheduled
sampling. A PyTorch port of the reference TensorFlow code (liyaguang/DCRNN); differences are listed in the docstring
of DCGRUCell.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from metroflow.models.graph_layers import GATv2Layer


# --------------------------------------------------------------------------- STGAT
class STGAT(nn.Module):
    """Temporal encoder, then graph attention, then a 12-horizon head.

    1. A GRU (weights shared by all sensors) encodes each sensor's own 12-step history into a 64-d summary.
    2. Two GATv2 layers let every sensor attend to its road neighbours' summaries (residual + LayerNorm).
       A neighbour's summary already holds its recent trajectory ("my speed fell 10 minutes ago"), which is exactly the
       lead-lag signal the exploratory analysis measured, so one graph pass per window is enough.
    3. A linear head outputs the change from the last observed speed for the 12 horizons.

    Attention at every input step costs 12 graph passes per window instead of one: 0.67 s versus
    0.085 s per training batch of 32 windows on an M3 Pro.
    """

    def __init__(self, edge_index: torch.Tensor, edge_attr: torch.Tensor, n_in: int = 7, hidden: int = 64, heads: int = 4,
                 gat_layers: int = 2, horizon: int = 12, dropout: float = 0.1):
        super().__init__()
        self.register_buffer("edge_index", edge_index)
        self.register_buffer("edge_attr", edge_attr)
        self.gru = nn.GRU(n_in, hidden, num_layers=2, batch_first=True, dropout=dropout)
        self.gats = nn.ModuleList([GATv2Layer(hidden, hidden // heads, heads, edge_dim=edge_attr.shape[1])
                                   for _ in range(gat_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(gat_layers)])
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, horizon))

    def spatial(self, h: torch.Tensor) -> torch.Tensor:
        """h (B, N, hidden): B graphs sharing the road edges."""
        for gat, norm in zip(self.gats, self.norms):
            h = norm(h + self.drop(nn.functional.elu(gat(h, self.edge_index, self.edge_attr))))
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, n, f = x.shape
        _, last = self.gru(x.permute(0, 2, 1, 3).reshape(b * n, t, f))
        own = last[-1].reshape(b, n, -1)                                        # each sensor's own history
        mixed = self.spatial(own)                                               # ... after listening to neighbours
        out = self.head(torch.cat([own, mixed], -1)).permute(0, 2, 1)           # (B, 12, N)
        return out + x[:, -1:, :, 0]                                            # forecast = last speed + change

    def attention(self) -> torch.Tensor | None:
        """Head-averaged attention of the last GAT layer from the latest forward pass: (B, E)."""
        a = self.gats[-1].last_alpha
        return None if a is None else a.mean(-1)


# --------------------------------------------------------------------------- DCRNN
class DiffusionConv(nn.Module):
    """y = sum_s sum_k T_k(S_s) x Theta_{s,k}: Chebyshev-style recursion x_k = 2 S x_{k-1} - x_{k-2}, as in the reference."""

    def __init__(self, n_supports: int, k: int, in_dim: int, out_dim: int, bias_start: float = 0.0):
        super().__init__()
        self.k = k
        self.n_mat = n_supports * k + 1
        self.lin = nn.Linear(in_dim * self.n_mat, out_dim)
        nn.init.xavier_normal_(self.lin.weight)
        nn.init.constant_(self.lin.bias, bias_start)

    def forward(self, x: torch.Tensor, supports: list[torch.Tensor]) -> torch.Tensor:   # x (B, N, F)
        mats = [x]
        for s in supports:
            x0 = x
            x1 = torch.einsum("nm,bmf->bnf", s, x0)
            mats.append(x1)
            for _ in range(2, self.k + 1):
                x2 = 2 * torch.einsum("nm,bmf->bnf", s, x1) - x0
                mats.append(x2)
                x0, x1 = x1, x2
        return self.lin(torch.cat(mats, dim=-1))


class DCGRUCell(nn.Module):
    """GRU cell whose matrix products are diffusion convolutions over the road graph.

    Differences from the TensorFlow reference (both deliberate): the Chebyshev recursion restarts for every support
    (the reference carries x0 over from the previous support), and the supports are dense 207 x 207 matrices instead
    of sparse ones (cheaper than sparse ops at this size).
    """

    def __init__(self, in_dim: int, units: int, n_supports: int, k: int):
        super().__init__()
        self.units = units
        self.gates = DiffusionConv(n_supports, k, in_dim + units, 2 * units, bias_start=1.0)
        self.cand = DiffusionConv(n_supports, k, in_dim + units, units)

    def forward(self, x, h, supports):
        ru = torch.sigmoid(self.gates(torch.cat([x, h], -1), supports))
        r, u = ru.chunk(2, dim=-1)
        c = torch.tanh(self.cand(torch.cat([x, r * h], -1), supports))
        return u * h + (1 - u) * c


class DCRNN(nn.Module):
    def __init__(self, supports: list[torch.Tensor], n_in: int = 7, units: int = 64, layers: int = 2, k: int = 2,
                 horizon: int = 12, cl_decay_steps: int = 2000):
        super().__init__()
        for i, s in enumerate(supports):
            self.register_buffer(f"support_{i}", s)
        self.n_sup, self.units, self.horizon, self.cl_decay = len(supports), units, horizon, cl_decay_steps
        self.enc = nn.ModuleList([DCGRUCell(n_in if i == 0 else units, units, self.n_sup, k) for i in range(layers)])
        self.dec = nn.ModuleList([DCGRUCell(1 if i == 0 else units, units, self.n_sup, k) for i in range(layers)])
        self.proj = nn.Linear(units, 1)

    @property
    def supports(self) -> list[torch.Tensor]:
        return [getattr(self, f"support_{i}") for i in range(self.n_sup)]

    def teacher_prob(self, batches_seen: int) -> float:
        """Probability of feeding the ground truth to the decoder (inverse-sigmoid decay, as in the reference)."""
        return self.cl_decay / (self.cl_decay + math.exp(min(batches_seen / self.cl_decay, 50)))

    def forward(self, x: torch.Tensor, y_z: torch.Tensor | None = None, batches_seen: int = 0) -> torch.Tensor:
        """x (B, 12, N, F); y_z (B, 12, N) z-scored targets (NaN where masked), used only for scheduled sampling."""
        b, t, n, _ = x.shape
        sup = self.supports
        hs = [x.new_zeros(b, n, self.units) for _ in self.enc]
        for step in range(t):
            inp = x[:, step]
            for i, cell in enumerate(self.enc):
                hs[i] = cell(inp, hs[i], sup)
                inp = hs[i]
        go = x.new_zeros(b, n, 1)
        outs = []
        p_teacher = self.teacher_prob(batches_seen) if (self.training and y_z is not None) else 0.0
        for step in range(self.horizon):
            inp = go
            for i, cell in enumerate(self.dec):
                hs[i] = cell(inp, hs[i], sup)
                inp = hs[i]
            pred = self.proj(inp)                                   # (B, N, 1)
            outs.append(pred.squeeze(-1))
            go = pred
            if p_teacher > 0 and torch.rand(()) < p_teacher:
                truth = y_z[:, step].unsqueeze(-1)
                go = torch.where(torch.isnan(truth), pred, truth)   # masked target: fall back on the prediction
        return torch.stack(outs, dim=1)                             # (B, 12, N)
