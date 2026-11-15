"""Phase 5 analysis: ablation table, learning curves, DCRNN versus its published numbers, and attention inspection.

    python -m src.eval.phase5        # -> results/ablations.csv, results/attention.csv, results/phase5_findings.json,
                                     #    figures 12-14

Attention inspection asks the trained STGAT where it looks. Each sensor attends over itself, its upstream neighbours
(forward edges) and its downstream neighbours (reverse edges). Phase 1 found that jams usually reach the downstream
sensor first, so if the model learned that, it should shift attention downstream when a slowdown is about to arrive.
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.data.dataset import load_adjacency, load_scaler, load_split
from src.models.graph_layers import road_graph
from src.train.graph import CKPT, RESULTS, ROOT, RunConfig, build_model, device
from src.viz import style as S
from src.viz.eda_plots import save
from src.viz.maps import free_flow

PUBLISHED_DCRNN = {15: 2.67, 30: 3.08, 60: 3.56}          # liyaguang/DCRNN README, METR-LA, after the 2018 bug fix
ABLATIONS = [  # (label, run, compared with)
    ("STGAT, bidirectional edges (main)", "stgat_s0", "stgat_s0"),
    ("no graph (self-loops only)", "stgat_identity_s0", "stgat_s0"),
    ("forward edges only (hear upstream)", "stgat_forward_s0", "stgat_s0"),
    ("backward edges only (hear downstream)", "stgat_backward_s0", "stgat_s0"),
    ("undirected, no direction feature", "stgat_undirected_s0", "stgat_s0"),
    ("random graph, same edge count", "stgat_random_s0", "stgat_s0"),
    ("1 attention layer instead of 2", "stgat_1layer_s0", "stgat_s0"),
    ("DCRNN, dual random walk, K = 2 (main)", "dcrnn_s0", "dcrnn_s0"),
    ("DCRNN, K = 1 hop", "dcrnn_k1_s0", "dcrnn_s0"),
    ("DCRNN, forward walk only", "dcrnn_forward_s0", "dcrnn_s0"),
]


def load_run(run: str) -> tuple[pd.DataFrame, dict] | None:
    f = RESULTS / f"{run}.csv"
    if not f.exists():
        return None
    return pd.read_csv(f, dtype={"horizon": str}), json.loads(f.with_suffix(".json").read_text())


def test_mae(df: pd.DataFrame, minutes: int) -> float:
    return float(df[(df.split == "test") & (df.horizon == str(minutes // 5))].mae.iloc[0])


def ablation_table() -> pd.DataFrame:
    rows = []
    for label, run, ref in ABLATIONS:
        r, base = load_run(run), load_run(ref)
        if r is None or base is None:
            continue
        df, meta = r
        row = {"variant": label, "run": run, "parameters": meta["parameters"], "epochs_run": meta["epochs_run"]}
        for mins in (15, 30, 60):
            row[f"MAE_{mins}min"] = round(test_mae(df, mins), 3)
            row[f"delta_{mins}min_vs_main"] = round(test_mae(df, mins) - test_mae(base[0], mins), 3)
        rows.append(row)
    return pd.DataFrame(rows)


def reference_check() -> dict | None:
    r = load_run("dcrnn_dcrnn_s0")
    if r is None:
        return None
    out = {}
    for mins, pub in PUBLISHED_DCRNN.items():
        ours = test_mae(r[0], mins)
        out[f"{mins}min"] = {"ours": round(ours, 3), "published": pub, "diff_pct": round(100 * (ours - pub) / pub, 1)}
    return out


# --------------------------------------------------------------------------- attention
def attention_by_edge(run: str = "stgat_s0", batch: int = 64) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Mean attention of every edge over the test windows, split by whether the target sensor is about to slow down.

    'Slowdown ahead' = the sensor is at >= 0.8 of free flow now, and below 0.6 of it 30 minutes later (real readings).
    Returns (per-edge table, edge_index, per-window flags)."""
    cfg = RunConfig(**{k: v for k, v in json.loads((CKPT / run / "config.json").read_text())["config"].items()
                       if k in RunConfig.__dataclass_fields__})
    adj = load_adjacency()["adj"]
    dev = device()
    model = build_model(cfg, adj).to(dev)
    model.load_state_dict(torch.load(CKPT / run / "best.pt", map_location=dev))
    model.eval()
    scaler = load_scaler()
    test = load_split("test", keys=["X", "Y"])
    ff = free_flow()
    now = test["X"][:, -1, :, 0] * scaler["sd"] + scaler["mu"]                  # (S, N) speed at the forecast origin
    real_now = test["X"][:, -1, :, 1] > 0
    later = test["Y"][:, 5]                                                      # 30 min ahead
    ahead = real_now & (now >= 0.8 * ff) & (later < 0.6 * ff)                    # (S, N) slowdown about to arrive
    ei, ea = road_graph(adj, cfg.graph)
    dst = ei[1].numpy()
    sums = {"slowdown_ahead": np.zeros(ei.shape[1]), "other": np.zeros(ei.shape[1])}
    counts = {"slowdown_ahead": np.zeros(ei.shape[1]), "other": np.zeros(ei.shape[1])}
    with torch.no_grad():
        for a in range(0, len(test["X"]), batch):
            model(torch.from_numpy(np.ascontiguousarray(test["X"][a:a + batch])).to(dev))
            alpha = model.attention().cpu().numpy()                              # (b, E)
            flag = ahead[a:a + batch][:, dst]                                    # (b, E): is this edge's target about to slow?
            for k, sel in (("slowdown_ahead", flag), ("other", ~flag)):
                sums[k] += np.where(sel, alpha, 0).sum(0)
                counts[k] += sel.sum(0)
    kind = np.where(ei[0] == ei[1], "self", np.where(ea[:, 1].numpy() > 0, "downstream neighbour", "upstream neighbour"))
    tab = pd.DataFrame({"src": ei[0].numpy(), "dst": dst, "kind": kind, "weight": ea[:, 0].numpy(),
                        "att_slowdown_ahead": sums["slowdown_ahead"] / np.maximum(counts["slowdown_ahead"], 1),
                        "att_other": sums["other"] / np.maximum(counts["other"], 1),
                        "n_slowdown_ahead": counts["slowdown_ahead"].astype(int)})
    return tab, ei.numpy(), ahead


MIN_SLOWDOWN_WINDOWS = 20


def comparable_sensors(tab: pd.DataFrame) -> pd.Index:
    """Sensors with both upstream and downstream neighbours and at least 20 'slowdown ahead' test windows."""
    both = tab[tab.kind != "self"].groupby("dst").kind.nunique()
    n = tab[tab.kind == "self"].set_index("dst").n_slowdown_ahead
    return both[(both == 2) & (n.reindex(both.index) >= MIN_SLOWDOWN_WINDOWS)].index


def attention_shares(tab: pd.DataFrame) -> pd.DataFrame:
    """Average share of a sensor's attention going to itself / upstream / downstream, by situation.

    Shares are summed per target sensor first (each sensor's attention sums to 1), then averaged over comparable
    sensors (both kinds of neighbours, >= 20 slowdown-ahead windows), so sensors are weighted equally."""
    keep = comparable_sensors(tab)
    rows = []
    for col, name in (("att_slowdown_ahead", "slowdown ahead (30 min)"), ("att_other", "all other windows")):
        per = tab[tab.dst.isin(keep)].groupby(["dst", "kind"])[col].sum().unstack(fill_value=0)
        for kind in ("self", "upstream neighbour", "downstream neighbour"):
            rows.append({"situation": name, "attends_to": kind, "mean_share_pct": round(100 * float(per[kind].mean()), 2),
                         "sensors": int(len(per))})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- figures
def fig_learning_curves(runs: list[str]) -> plt.Figure:
    S.apply()
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for run, c in zip(runs, (S.ORANGE, S.AQUA, S.BLUE)):
        r = load_run(run)
        if r is None:
            continue
        h = pd.DataFrame(r[1]["history"])
        ax.plot(h.epoch, h.val_mae, color=c, marker="o", ms=3, label=f"{run}: validation")
        ax.plot(h.epoch, h.train_mae, color=c, ls="--", lw=1.4, label=f"{run}: training (half-epoch sample)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("masked MAE, all horizons (mph)")
    ax.set_title("Learning curves: validation error flattens; early stopping keeps the best epoch")
    ax.legend(fontsize=8)
    return fig


def fig_ablations(ab: pd.DataFrame, refs: dict[str, float]) -> plt.Figure:
    S.apply()
    d = ab.iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 0.42 * len(d) + 1.6))
    colors = [S.ORANGE if "(main)" in v else (S.AQUA if v.startswith("DCRNN") else S.BLUE) for v in d.variant]
    ax.barh(range(len(d)), d.MAE_60min, color=colors, height=0.6)
    for i, (v, dl) in enumerate(zip(d.MAE_60min, d.delta_60min_vs_main)):
        ax.text(v + 0.01, i, f"{v:.2f}" + ("" if dl == 0 else f"  ({dl:+.2f})"), va="center", fontsize=9, color=S.INK_2)
    for name, val in refs.items():
        ax.axvline(val, color=S.MUTED, ls="--", lw=1)
        ax.text(val, len(d) - 0.4, f" {name}", fontsize=8, color=S.INK_2, rotation=0, va="bottom")
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(d.variant, fontsize=9)
    ax.set_xlim(d.MAE_60min.min() - 0.3, max(d.MAE_60min.max(), *refs.values()) + 0.35)
    ax.set_xlabel("test MAE at 60 min (mph); in brackets: change vs the main model")
    ax.set_title("Ablations (seed 0): which parts of the graph matter?")
    ax.grid(axis="y", visible=False)
    return fig


def fig_attention(tab: pd.DataFrame, shares: pd.DataFrame) -> plt.Figure:
    S.apply()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.4), width_ratios=[1.3, 1])
    sh = shares.pivot(index="attends_to", columns="situation", values="mean_share_pct")
    kinds = ["self", "upstream neighbour", "downstream neighbour"]
    x = np.arange(3)
    for k, (col, c) in enumerate([("all other windows", S.AXIS), ("slowdown ahead (30 min)", S.ORANGE)]):
        a1.bar(x + (k - 0.5) * 0.38, sh.loc[kinds, col], 0.36, color=c, label=col)
        for i, v in enumerate(sh.loc[kinds, col]):
            a1.text(i + (k - 0.5) * 0.38, v + 0.5, f"{v:.1f}%", ha="center", fontsize=9)
    a1.set_xticks(x)
    a1.set_xticklabels(kinds)
    a1.set_ylabel("share of a sensor's attention (%)")
    a1.set_ylim(0, sh.values.max() * 1.25)
    a1.legend(loc="upper left")
    a1.set_title("Where STGAT looks, averaged over sensors")

    # every sensor: downstream share in normal windows vs when a slowdown is coming
    keep = comparable_sensors(tab)
    per = {c: tab[tab.dst.isin(keep)].groupby(["dst", "kind"])[c].sum().unstack(fill_value=0)["downstream neighbour"] * 100
           for c in ("att_other", "att_slowdown_ahead")}
    x, y = per["att_other"], per["att_slowdown_ahead"]
    up = int((y > x).sum())
    a2.plot([0, 100], [0, 100], color=S.AXIS, lw=1, zorder=1)
    a2.scatter(x, y, s=22, c=np.where(y > x, S.ORANGE, S.BLUE), edgecolor=S.SURFACE, linewidth=0.6, zorder=2)
    a2.set_xlim(0, 100)
    a2.set_ylim(0, 100)
    a2.set_aspect("equal")
    a2.set_xlabel("attention to downstream neighbours, all other windows (%)")
    a2.set_ylabel("... when a slowdown is 30 min away (%)")
    a2.text(4, 92, f"{up} of {len(x)} sensors (orange) look further\ndownstream when a slowdown is coming",
            fontsize=9, color=S.INK_2, va="top")
    a2.set_title("Every sensor: does attention shift downstream?")
    fig.tight_layout()
    return fig


def main() -> None:
    from src.train.graph import make_table

    make_table()
    ab = ablation_table()
    ab.to_csv(ROOT / "results" / "ablations.csv", index=False)
    print(ab.to_string(index=False))
    findings: dict = {"reference_check": reference_check()}
    base = pd.read_csv(ROOT / "results" / "baselines.csv").query("split == 'test'").set_index("model")
    refs = {"XGBoost": float(base.loc["XGBoost per horizon", "MAE_60min"]),
            "LSTM (no graph)": float(base.loc["LSTM (no graph)", "MAE_60min"])}
    if len(ab):
        save(fig_ablations(ab, refs), "fig13_ablations.png")
    save(fig_learning_curves(["stgat_s0", "dcrnn_s0"]), "fig12_learning_curves.png")
    if (CKPT / "stgat_s0" / "best.pt").exists():
        tab, _, ahead = attention_by_edge()
        tab.to_csv(ROOT / "results" / "attention.csv", index=False, float_format="%.5f")
        shares = attention_shares(tab)
        shares.to_csv(ROOT / "results" / "attention_shares.csv", index=False)
        print(shares.to_string(index=False))
        findings["attention_shares"] = shares.to_dict("records")
        findings["slowdown_ahead_windows"] = int(ahead.sum())
        keep = comparable_sensors(tab)
        per = {c: tab[tab.dst.isin(keep)].groupby(["dst", "kind"])[c].sum().unstack(fill_value=0)["downstream neighbour"]
               for c in ("att_other", "att_slowdown_ahead")}
        findings["sensors_shifting_downstream"] = int((per["att_slowdown_ahead"] > per["att_other"]).sum())
        save(fig_attention(tab, shares), "fig14_attention.png")
    plt.close("all")
    (ROOT / "results" / "phase5_findings.json").write_text(json.dumps(findings, indent=2, default=float))
    print(json.dumps(findings, indent=2, default=float))


if __name__ == "__main__":
    main()
