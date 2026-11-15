"""Phase 6 figures: error vs horizon for every model, peak / off-peak bars, and where the graph models gain."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.viz import style as S
from src.viz.eda_plots import save

# Colour follows the model, never its rank. Grey = reference baselines, colour = the models being compared.
LOOK = {
    "Persistence": (S.MUTED, "--"),
    "Historical average": (S.AXIS, "--"),
    "LSTM (no graph)": (S.YELLOW, "-"),
    "XGBoost per horizon": (S.BLUE, "-"),
    "XGBoost + neighbours": (S.BLUE, ":"),
    "STGAT (ours)": (S.ORANGE, "-"),
    "DCRNN": (S.AQUA, "-"),
}


def plot_horizon(long: pd.DataFrame) -> plt.Figure:
    S.apply()
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, (color, ls) in LOOK.items():
        d = long[long.model == label]
        if d.empty:
            continue
        g = d.assign(h=d.horizon.astype(int)).groupby("h").mae
        mean, std = g.mean(), g.std().fillna(0)
        name = label + (" (15/30/60 min only)" if len(mean) < 12 else "")
        ax.plot(mean.index * 5, mean.values, color=color, ls=ls, marker="o", ms=3.5, label=name)
        if d.run.nunique() > 1:
            ax.fill_between(mean.index * 5, mean - std, mean + std, color=color, alpha=0.15, lw=0)
    ax.set_xticks(range(5, 61, 5))
    ax.set_xlabel("forecast horizon (min)")
    ax.set_ylabel("masked MAE (mph)")
    ax.set_title("Test error by forecast horizon (graph models: mean of their seeds, band = ±1 std)")
    ax.legend(ncol=2, loc="upper left")
    return fig


def plot_periods(final: pd.DataFrame) -> plt.Figure:
    S.apply()
    names = [m for m in LOOK if m in set(final.model)]
    f = final.set_index("model").loc[names]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    x = np.arange(len(names))
    for k, (col, lab, c) in enumerate([("peak_MAE_60min", "weekday peak", S.BLUE),
                                       ("offpeak_MAE_60min", "weekday off-peak", S.ORANGE),
                                       ("weekend_MAE_60min", "weekend", S.AQUA)]):
        ax.bar(x + (k - 1) * 0.26, f[col], width=0.24, color=c, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(" + ", "\n+ ").replace(" (", "\n(") for n in names], fontsize=9)
    ax.set_ylabel("MAE at 60 min (mph)")
    ax.legend(loc="upper right")
    ax.set_title("60-minute error by period: rush hours are where every model struggles most")
    return fig


def plot_gain(deg: pd.DataFrame, ons: pd.DataFrame, graph: str, base: str = "XGBoost per horizon") -> plt.Figure:
    S.apply()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6))
    d = deg.pivot(index="degree", columns="model", values="mae_60min").loc[["0-4", "5-8", "9-12", "13+"]]
    n = deg.drop_duplicates("degree").set_index("degree").sensors.loc[d.index]
    rel = 100 * (d[base] - d[graph]) / d[base]
    a1.bar(range(len(rel)), rel.values, color=LOOK[graph][0], width=0.6)
    a1.set_xticks(range(len(rel)))
    a1.set_xticklabels([f"{i}\n({k} sensors)" for i, k in zip(rel.index, n.values)])
    for i, v in enumerate(rel.values):
        a1.text(i, v + (0.3 if v >= 0 else -0.3), f"{v:+.1f}%", ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
    a1.axhline(0, color=S.AXIS, lw=1)
    a1.set_xlabel("road links per sensor (in + out)")
    a1.set_ylabel(f"60-min MAE reduction vs {base} (%)")
    a1.set_title(f"{graph}: gain by sensor connectivity")

    order = [m for m in ("XGBoost per horizon", "XGBoost + neighbours", "STGAT (ours)", "DCRNN") if m in set(ons.model)]
    o = ons[ons.group == "onset"].pivot(index="model", columns="minutes", values="mae").loc[order]
    x = np.arange(3)
    for k, m in enumerate(order):
        color, _ = LOOK[m]
        a2.bar(x + (k - (len(order) - 1) / 2) * 0.2, o.loc[m, [15, 30, 60]], 0.18, color=color,
               hatch="//" if m == "XGBoost + neighbours" else None, edgecolor=S.SURFACE, label=m)
    a2.set_xticks(x)
    a2.set_xticklabels(["15 min ahead", "30 min ahead", "60 min ahead"])
    a2.set_ylabel("MAE in the 30 min after a congestion onset (mph)")
    a2.legend(loc="upper left", fontsize=8)
    a2.set_title("Error right after congestion starts, by forecast horizon")
    fig.tight_layout()
    return fig


def make_all(final, long, deg, ons, sig) -> None:
    save(plot_horizon(long), "fig15_final_error_by_horizon.png")
    save(plot_periods(final), "fig16_final_peak_offpeak.png")
    graphs = [g for g in ("STGAT (ours)", "DCRNN") if g in set(final.model)]
    if graphs and "XGBoost per horizon" in set(final.model):
        best = min(graphs, key=lambda g: final.set_index("model").loc[g, "MAE_60min"])
        save(plot_gain(deg, ons, best), "fig17_where_the_graph_helps.png")
    plt.close("all")
