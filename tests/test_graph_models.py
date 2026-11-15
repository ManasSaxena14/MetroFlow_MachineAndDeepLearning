"""Phase 5 checks: the attention layer matches PyTorch Geometric, graph variants and diffusion directions are right,
and information flows between sensors only along the edges a model is given."""
import numpy as np
import pytest
import torch

from src.features.build import flatten_targets
from src.models.graph_layers import GATv2Layer, diffusion_supports, road_graph, scatter_softmax
from src.models.graph_models import DCRNN, STGAT
from src.train.graph import RunConfig, predict_rows

# toy road 0 -> 1 -> 2 (adj[i, j] > 0: j is DOWNSTREAM of i); sensor 3 isolated; self-loops on the diagonal
ADJ = np.eye(4, dtype=np.float32)
ADJ[0, 1], ADJ[1, 2] = 0.9, 0.5


def edges(ei):
    return set(map(tuple, ei.T.tolist()))


def test_gatv2_matches_pytorch_geometric():
    pyg = pytest.importorskip("torch_geometric.nn")
    torch.manual_seed(0)
    ei, ea = road_graph(ADJ, "bidirectional")
    ref = pyg.GATv2Conv(8, 5, heads=3, edge_dim=2, add_self_loops=False, share_weights=False)
    ours = GATv2Layer(8, 5, heads=3, edge_dim=2)
    with torch.no_grad():
        for name in ("lin_l", "lin_r"):
            getattr(ours, name).weight.copy_(getattr(ref, name).weight)
            getattr(ours, name).bias.copy_(getattr(ref, name).bias)
        ours.lin_edge.weight.copy_(ref.lin_edge.weight)
        ours.att.copy_(ref.att)
        ours.bias.copy_(ref.bias)
    x = torch.randn(4, 8)
    assert torch.allclose(ours(x[None], ei, ea)[0], ref(x, ei, ea), atol=1e-5)


def test_scatter_softmax_sums_to_one_per_target():
    ei, _ = road_graph(ADJ, "bidirectional")
    a = scatter_softmax(torch.randn(2, ei.shape[1], 3), ei[1], 4)
    tot = torch.zeros(2, 4, 3).index_add(1, ei[1], a)
    assert torch.allclose(tot, torch.ones_like(tot), atol=1e-6)


def test_graph_variants():
    loops = {(i, i) for i in range(4)}
    fwd, _ = road_graph(ADJ, "forward")
    assert edges(fwd) == {(0, 1), (1, 2)} | loops                       # messages flow upstream -> downstream
    bwd, _ = road_graph(ADJ, "backward")
    assert edges(bwd) == {(1, 0), (2, 1)} | loops
    bi, attr = road_graph(ADJ, "bidirectional")
    assert edges(bi) == {(0, 1), (1, 2), (1, 0), (2, 1)} | loops
    rev = {tuple(e) for e, r in zip(bi.T.tolist(), attr[:, 1].tolist()) if r == 1}
    assert rev == {(1, 0), (2, 1)}                                       # is_reverse marks downstream -> upstream
    ident, _ = road_graph(ADJ, "identity")
    assert edges(ident) == loops
    und, _ = road_graph(ADJ, "undirected")
    assert edges(und) == {(0, 1), (1, 0), (1, 2), (2, 1)} | loops
    rnd, _ = road_graph(ADJ, "random", seed=3)
    assert rnd.shape[1] == fwd.shape[1] and len(edges(rnd)) == rnd.shape[1]


def test_diffusion_directions():
    s_f, s_b = diffusion_supports(ADJ, "dual")
    x = torch.tensor([10.0, 20.0, 30.0, 40.0]).view(1, 4, 1)
    up = torch.einsum("nm,bmf->bnf", s_f, x)[0, :, 0]
    down = torch.einsum("nm,bmf->bnf", s_b, x)[0, :, 0]
    # forward walk: sensor 1 mixes itself with its UPSTREAM sensor 0; backward walk: with its DOWNSTREAM sensor 2
    assert 10 < up[1] < 20 and 20 < down[1] < 30
    assert up[3] == 40 and down[3] == 40                                # isolated sensor sees only itself


def _influence(model, src: int, dst: int) -> float:
    """How much sensor dst's forecast moves when only sensor src's input changes."""
    model.eval()
    torch.manual_seed(0)
    x = torch.randn(1, 12, 4, 7)
    x2 = x.clone()
    x2[:, :, src, 0] += 3.0
    with torch.no_grad():
        return float((model(x2)[0, :, dst] - model(x)[0, :, dst]).abs().max())


def test_stgat_information_flows_only_along_edges():
    torch.manual_seed(0)
    none = STGAT(*road_graph(ADJ, "identity"), hidden=16, heads=2)
    assert _influence(none, src=2, dst=1) == 0.0                         # no graph: sensors are independent
    fwd = STGAT(*road_graph(ADJ, "forward"), hidden=16, heads=2)
    assert _influence(fwd, src=2, dst=1) == 0.0                          # forward-only: 1 cannot hear downstream 2
    bi = STGAT(*road_graph(ADJ, "bidirectional"), hidden=16, heads=2)
    assert _influence(bi, src=2, dst=1) > 0                              # bidirectional: 1 hears the jam at 2
    assert _influence(bi, src=3, dst=1) == 0.0                           # but never the unconnected sensor 3


def test_dcrnn_shapes_receptive_field_and_teacher_forcing():
    torch.manual_seed(0)
    m = DCRNN(diffusion_supports(ADJ, "dual"), units=8, k=1)
    assert m(torch.randn(2, 12, 4, 7)).shape == (2, 12, 4)
    assert _influence(m, src=2, dst=1) > 0 and _influence(m, src=3, dst=1) == 0.0
    assert m.teacher_prob(0) > 0.99 and m.teacher_prob(40_000) < 0.01   # scheduled sampling decays


def test_predict_rows_matches_target_row_layout():
    class Echo(torch.nn.Module):                                         # forecast = the last input speed, z units
        def forward(self, x):
            return x[:, -1:, :, 0].expand(-1, 12, -1)
    X = np.random.default_rng(0).normal(size=(5, 12, 4, 7)).astype(np.float32)
    P = predict_rows(Echo(), X, {"mu": 0.0, "sd": 1.0}, torch.device("cpu"), batch=2)
    Y = np.repeat(X[:, -1:, :, 0], 12, axis=1)                           # (S, 12, N)
    Yr, _ = flatten_targets(Y, np.ones_like(Y, bool))
    assert np.allclose(P, Yr)


def test_run_names_are_unique_per_setting():
    names = {RunConfig(model="stgat", graph=g).name for g in ("bidirectional", "identity", "random")}
    names |= {RunConfig(model="dcrnn", k=k).name for k in (1, 2)} | {RunConfig(model="dcrnn", protocol="dcrnn").name}
    assert len(names) == 6
