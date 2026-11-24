"""Graph building blocks in plain PyTorch so they run on CPU, CUDA and Apple-silicon GPUs alike.

* `road_graph(...)`   : turns the fixed adjacency into the edge list a model uses (and its ablation variants).
* `GATv2Layer`        : graph attention (Brody et al., 2022), the same maths and parameter names as PyTorch Geometric's
                        `GATv2Conv(share_weights=False, add_self_loops=False, edge_dim=D)`; tests/test_graph_models.py
                        checks the outputs are identical. Works on a batch of graphs that share one edge list, so no
                        graph copies are needed.
* `diffusion_supports`: the forward / backward random-walk matrices of DCRNN (Li et al., 2018).

Direction convention (same as the rest of the project): adj[i, j] > 0 means traffic drives from sensor i to sensor j,
so i is UPSTREAM of j. An edge src -> dst sends a message from src to dst.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

GRAPH_VARIANTS = ("bidirectional", "forward", "backward", "undirected", "identity", "random")


def road_graph(adj: np.ndarray, variant: str = "bidirectional", seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Edge list (2, E) and edge features (E, 2) = [kernel weight, is_reverse] for one graph variant.

    bidirectional : every road link in both directions; is_reverse = 1 marks downstream -> upstream messages.
                    The exploratory analysis found that jams usually reach the downstream sensor first, so a sensor must hear its
                    downstream neighbours, not only the upstream ones the raw matrix points from.
    forward       : only upstream -> downstream messages (what message passing on the raw matrix gives).
    backward      : only downstream -> upstream messages.
    undirected    : symmetrised links, no direction feature (ablation: does direction matter?).
    identity      : self-loops only, i.e. no graph (ablation: equal-budget no-graph control).
    random        : as many random links as the real graph, real weights shuffled (ablation: does the ROAD matter?).
    Every variant keeps one self-loop per sensor, so a sensor always sees its own history.
    """
    if variant not in GRAPH_VARIANTS:
        raise ValueError(f"unknown graph variant {variant!r}")
    n = adj.shape[0]
    off = adj.copy()
    np.fill_diagonal(off, 0)
    src, dst = np.nonzero(off)
    w = off[src, dst]
    if variant == "random":
        rng = np.random.default_rng(seed)
        pairs = set()
        while len(pairs) < len(src):
            a, b = rng.integers(0, n, 2)
            if a != b:
                pairs.add((int(a), int(b)))
        src, dst = map(np.array, zip(*sorted(pairs)))
        w = rng.permutation(w)
    if variant == "undirected":
        sym = np.maximum(off, off.T)
        src, dst = np.nonzero(sym)
        w = sym[src, dst]
    parts = []
    if variant in ("bidirectional", "forward", "undirected", "random"):
        parts.append((src, dst, w, np.zeros_like(w)))
    if variant in ("bidirectional", "backward"):
        parts.append((dst, src, w, np.ones_like(w)))
    loops = np.arange(n)
    parts.append((loops, loops, np.ones(n, np.float32), np.zeros(n, np.float32)))
    s, d, ww, rev = (np.concatenate(p) for p in zip(*parts))
    edge_index = torch.tensor(np.stack([s, d]), dtype=torch.long)
    edge_attr = torch.tensor(np.stack([ww, rev], axis=1), dtype=torch.float32)
    return edge_index, edge_attr


def scatter_softmax(e: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    """Softmax of e (G, E, H) over the edges that share a target node (index, (E,)). Numerically stable."""
    g, _, h = e.shape
    idx = index.view(1, -1, 1).expand_as(e)
    mx = torch.full((g, n, h), float("-inf"), device=e.device, dtype=e.dtype).scatter_reduce(1, idx, e, "amax")
    ex = torch.exp(e - mx.gather(1, idx).detach())
    den = torch.zeros((g, n, h), device=e.device, dtype=e.dtype).index_add(1, index, ex)
    return ex / den.gather(1, idx).clamp_min(1e-16)


class GATv2Layer(nn.Module):
    """out_i = sum_j alpha_ij * W_l x_j + b,  alpha_ij = softmax_j( a . LeakyReLU(W_r x_i + W_l x_j + W_e e_ji) ).

    x: (G, N, F) node features of G graphs with the same edges -> (G, N, heads * out) (concat) or (G, N, out) (mean).
    """

    def __init__(self, in_dim: int, out: int, heads: int = 4, edge_dim: int | None = 2, concat: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0.0):
        super().__init__()
        self.heads, self.out, self.concat, self.slope, self.dropout = heads, out, concat, negative_slope, dropout
        self.lin_l = nn.Linear(in_dim, heads * out)
        self.lin_r = nn.Linear(in_dim, heads * out)
        self.lin_edge = nn.Linear(edge_dim, heads * out, bias=False) if edge_dim else None
        self.att = nn.Parameter(torch.empty(1, heads, out))
        self.bias = nn.Parameter(torch.zeros(heads * out if concat else out))
        nn.init.xavier_uniform_(self.att)
        self.last_alpha: torch.Tensor | None = None       # (G, E, H), kept for attention inspection

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor | None = None) -> torch.Tensor:
        g, n, _ = x.shape
        h, c = self.heads, self.out
        src, dst = edge_index
        xl = self.lin_l(x).view(g, n, h, c)
        xr = self.lin_r(x).view(g, n, h, c)
        xj = xl[:, src]                                    # (G, E, H, C) message from the source node
        z = xj + xr[:, dst]
        if self.lin_edge is not None:
            z = z + self.lin_edge(edge_attr).view(1, -1, h, c)
        e = (nn.functional.leaky_relu(z, self.slope) * self.att).sum(-1)       # (G, E, H)
        alpha = scatter_softmax(e, dst, n)
        self.last_alpha = alpha.detach()
        alpha = nn.functional.dropout(alpha, self.dropout, self.training)
        out = torch.zeros((g, n, h, c), device=x.device, dtype=x.dtype).index_add(1, dst, alpha.unsqueeze(-1) * xj)
        out = out.reshape(g, n, h * c) if self.concat else out.mean(2)
        return out + self.bias


def diffusion_supports(adj: np.ndarray, variant: str = "dual") -> list[torch.Tensor]:
    """Diffusion supports exactly as the DCRNN reference builds them: S = (D^-1 M)^T, applied as S @ x.

    forward  S_f = (D_O^-1 W)^T   : (S_f @ x)[i] mixes the sensors UPSTREAM of i   (traffic flowing in)
    backward S_b = (D_I^-1 W^T)^T : (S_b @ x)[i] mixes the sensors DOWNSTREAM of i (where jams come from)
    variant 'dual' -> [S_f, S_b] (the reference 'dual_random_walk'); 'forward' -> [S_f]; 'undirected' -> one
    support from max(W, W^T).
    """
    w = adj.astype(np.float64)

    def rw_t(m):
        d = m.sum(1, keepdims=True)
        return np.divide(m, d, out=np.zeros_like(m), where=d > 0).T

    if variant == "dual":
        mats = [rw_t(w), rw_t(w.T)]
    elif variant == "forward":
        mats = [rw_t(w)]
    elif variant == "undirected":
        mats = [rw_t(np.maximum(w, w.T))]
    else:
        raise ValueError(variant)
    return [torch.tensor(m, dtype=torch.float32) for m in mats]
