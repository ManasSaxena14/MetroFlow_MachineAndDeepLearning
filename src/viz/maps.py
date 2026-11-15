"""Phase 7: congestion maps from the saved test predictions.

    python -m src.viz.maps [--model auto|stgat_s0|dcrnn_s0] [--baseline xgb]

Congestion level of a sensor = speed / its own free-flow speed (85th percentile of TRAIN speeds):
free >= 0.8, slow 0.5-0.8, congested < 0.5, grey = sensor down (masked). Colours always come with a text label.

Figures (docs/figures/):
  fig18 side-by-side map: actual | XGBoost | graph model, 30 min ahead, at the PM-peak test moment with the most
        congested sensors (chosen by that rule, never by which model looks better)
  fig19 error map: per-sensor 60-min MAE of XGBoost minus the graph model (blue = graph model better)
  fig20 time-space diagram of one freeway corridor over a test evening: actual vs both models, 30 min ahead
  fig21 interactive animated map (HTML): actual vs graph model over a test evening, 15-min frames
"""
from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

from src.data.dataset import load_adjacency, load_split
from src.data.load import load_distances, load_locations, load_speed
from src.data.preprocess import load_config, split_bounds
from src.train.baselines import PRED_DIR, ROOT
from src.viz import style as S
from src.viz.eda_plots import FIG_DIR, save

LEVELS = [("congested", 0.0, 0.5), ("slow", 0.5, 0.8), ("free", 0.8, 9.0)]
SPEED_CMAP = LinearSegmentedColormap.from_list("slow_is_dark", ["#7a1414", "#d03b3b", "#ec835a", "#f6c9b0", "#f6f3ee"])
SPEED_CMAP.set_bad("#b9b8b2")                                 # sensor down: grey, never mistaken for free flow


def free_flow() -> np.ndarray:
    speed = load_speed()
    end = split_bounds(len(speed), load_config()["split"])["train"][1]
    v = speed.values[:end].astype(float)
    return np.nanpercentile(np.where(v > 0, v, np.nan), 85, axis=0)


def level(v: np.ndarray, ff: np.ndarray) -> np.ndarray:
    """Speeds (..., N) -> 0 congested / 1 slow / 2 free / -1 masked."""
    r = v / ff
    out = np.where(r < 0.5, 0, np.where(r < 0.8, 1, 2))
    return np.where(np.isnan(v), -1, out)


def windows(P_rows: np.ndarray, n: int) -> np.ndarray:
    """Row layout (S*N, 12) -> (S, 12, N)."""
    return np.asarray(P_rows).reshape(-1, n, 12).transpose(0, 2, 1)


def pick_moment(Y: np.ndarray, times: np.ndarray, ff: np.ndarray, h: int) -> int:
    """Test window whose target (h steps ahead) is a weekday 16:00-19:59 moment with the most congested sensors."""
    t = pd.DatetimeIndex(times[:, h - 1])
    ok = (t.dayofweek < 5) & (t.hour >= 16) & (t.hour < 20)
    n_cong = (level(Y[:, h - 1], ff) == 0).sum(1)
    return int(np.flatnonzero(ok)[np.argmax(n_cong[ok])])


def _status_scatter(ax, lon, lat, lev, title):
    for code, (name, color) in zip((-1, 2, 1, 0), [("sensor down", S.STATUS["down"]), ("free", S.STATUS["free"]),
                                                   ("slow", S.STATUS["slow"]), ("congested", S.STATUS["congested"])]):
        sel = lev == code
        ax.scatter(lon[sel], lat[sel], s=34 if code == 0 else 22, c=color, edgecolor=S.SURFACE, linewidth=0.8,
                   label=f"{name} ({sel.sum()})", zorder=3 if code == 0 else 2)
    ax.set_title(title, fontsize=13)
    ax.set_aspect(1 / np.cos(np.deg2rad(lat.mean())))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.legend(loc="lower left", fontsize=11, markerscale=1.5)


def fig_side_by_side(Y, preds: dict[str, np.ndarray], times, ff, lon, lat, edges, h: int = 6) -> tuple[plt.Figure, dict]:
    w = pick_moment(Y, times, ff, h)
    fig, axes = plt.subplots(1, 1 + len(preds), figsize=(4.6 * (1 + len(preds)), 4.9))
    when = pd.Timestamp(times[w, h - 1])
    actual = level(Y[w, h - 1], ff)
    info = {"window": w, "target_time": str(when), "horizon_min": 5 * h, "actual_congested": int((actual == 0).sum())}
    panels = [("Actual", actual)] + [(k, level(np.where(np.isnan(Y[w, h - 1]), np.nan, v[w, h - 1]), ff)) for k, v in preds.items()]
    for ax, (name, lev) in zip(axes, panels):
        for u, v in edges:
            ax.plot([lon[u], lon[v]], [lat[u], lat[v]], color=S.GRID, lw=0.6, zorder=1)
        title = f"Actual\n{(actual == 0).sum()} sensors congested"
        if name != "Actual":
            real = actual >= 0
            hit = ((lev == 0) & (actual == 0)).sum()
            info[name] = {"congested_predicted": int((lev == 0).sum()), "congested_caught": int(hit),
                          "level_accuracy_pct": round(100 * float((lev[real] == actual[real]).mean()), 1)}
            title = (f"{name}, {5 * h} min ahead\ncaught {hit} of {(actual == 0).sum()} congested; "
                     f"level right at {info[name]['level_accuracy_pct']}%")
        _status_scatter(ax, lon, lat, lev, title)
    fig.suptitle(f"Congestion map, {when:%a %d %b %Y %H:%M} (test block): actual vs {5 * h}-minute-ahead forecasts",
                 x=0.01, y=0.99, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig, info


def fig_error_map(Y, M, P_base, P_graph, lon, lat, edges, names, h: int = 12) -> tuple[plt.Figure, dict]:
    m = M[:, h - 1]
    y = np.nan_to_num(Y[:, h - 1])

    def sensor_mae(P):
        return np.where(m, np.abs(P[:, h - 1] - y), 0).sum(0) / m.sum(0)

    diff = sensor_mae(P_base) - sensor_mae(P_graph)                           # > 0: graph model better
    lim = np.nanpercentile(np.abs(diff), 98)
    cmap = ListedColormap(S.DIVERGING(np.linspace(0, 1, 256))[::-1])          # blue = graph model better
    fig, ax = plt.subplots(figsize=(9, 6))
    for u, v in edges:
        ax.plot([lon[u], lon[v]], [lat[u], lat[v]], color=S.GRID, lw=0.6, zorder=1)
    sc = ax.scatter(lon, lat, c=diff, cmap=cmap, vmin=-lim, vmax=lim, s=40, edgecolor=S.INK_2, linewidth=0.3, zorder=2)
    cb = fig.colorbar(sc, ax=ax, fraction=0.03)
    cb.set_label(f"60-min MAE: {names[0]} minus {names[1]} (mph)\nblue = {names[1]} better")
    cb.outline.set_visible(False)
    ax.set_aspect(1 / np.cos(np.deg2rad(lat.mean())))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    better = int((diff > 0).sum())
    ax.set_title(f"Where the graph model helps (60 min ahead):\n{names[1]} beats {names[0]} at {better} of {len(diff)} sensors")
    return fig, {"sensors_graph_better": better, "sensors": int(len(diff)), "median_gain_mph": float(np.median(diff))}


def corridor(adj: np.ndarray, sensor_ids: list[str], length: int = 15) -> tuple[list[int], np.ndarray]:
    """A chain of consecutive sensors along one freeway: repeatedly follow the closest downstream sensor.

    Picks the longest such chain in the graph (ties: the start with the strongest links) and returns its sensors in
    driving order with their cumulative road distance in km."""
    d = load_distances()
    idx = {s: i for i, s in enumerate(sensor_ids)}
    dist = np.full(adj.shape, np.inf)
    sub = d[d["from"].isin(idx) & d["to"].isin(idx)]
    dist[sub["from"].map(idx), sub["to"].map(idx)] = sub["cost"]
    off = (adj > 0) & ~np.eye(len(adj), dtype=bool)
    best: list[int] = []
    for start in range(len(adj)):
        path = [start]
        while len(path) < length:
            cand = [j for j in np.flatnonzero(off[path[-1]]) if j not in path and dist[path[-1], j] > 300]
            if not cand:
                break
            path.append(min(cand, key=lambda j: dist[path[-1], j]))
        if len(path) > len(best):
            best = path
    km = np.concatenate([[0], np.cumsum([dist[a, b] for a, b in zip(best, best[1:])])]) / 1000
    return best, km


def fig_time_space(Y, preds, times, path, km, ff, h: int = 6) -> tuple[plt.Figure, dict]:
    """x = time over one weekday test evening, y = position along the corridor, colour = speed (dark = slow)."""
    t = pd.DatetimeIndex(times[:, h - 1])
    ok = (t.dayofweek < 5) & (t.hour >= 15) & (t.hour < 20)
    days = pd.Series(t[ok].normalize()).unique()
    sl = Y[:, h - 1][:, path]
    day = max(days, key=lambda d: np.nanmean(sl[ok & (t.normalize() == d)] < 0.5 * ff[path]))
    sel = np.flatnonzero(ok & (t.normalize() == day))
    sel = sel[np.argsort(t[sel])]
    panels = [("Actual", Y)] + list(preds.items())
    fig, axes = plt.subplots(len(panels), 1, figsize=(11, 2.6 * len(panels)), sharex=True)
    for ax, (name, A) in zip(axes, panels):
        z = np.where(np.isnan(Y[sel, h - 1][:, path]), np.nan, A[sel, h - 1][:, path]).T   # same outages in every panel
        im = ax.pcolormesh(t[sel], km, z, cmap=SPEED_CMAP, vmin=10, vmax=70, shading="nearest")
        ax.set_ylabel("km along\ncorridor")
        ax.set_title(name if name == "Actual" else f"{name}, forecast made {5 * h} min earlier", fontsize=11)
        ax.grid(False)
    axes[-1].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%H:%M"))
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01)
    cb.set_label("speed (mph)")
    cb.outline.set_visible(False)
    fig.suptitle(f"Time-space diagram, {len(path)} consecutive sensors on one freeway, {pd.Timestamp(day):%a %d %b %Y} "
                 "(km in the driving direction; dark = slow, grey = sensor down)", x=0.01, ha="left", fontsize=12,
                 fontweight="bold")
    return fig, {"day": str(pd.Timestamp(day).date()), "sensors": [int(i) for i in path], "km": float(km[-1])}


def animated_map(Y, P, times, ff, lon, lat, name: str, h: int = 6) -> str:
    """Plotly HTML: actual vs forecast levels over the busiest weekday test evening, one frame per 15 minutes."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    t = pd.DatetimeIndex(times[:, h - 1])
    ok = (t.dayofweek < 5) & (t.hour >= 16) & (t.hour < 20) & (t.minute % 15 == 0)
    day = pd.Series(t[ok].normalize()).value_counts().index[0]
    frames_idx = np.flatnonzero(ok & (t.normalize() == day))
    frames_idx = frames_idx[np.argsort(t[frames_idx])]
    colors = {-1: S.STATUS["down"], 0: S.STATUS["congested"], 1: S.STATUS["slow"], 2: S.STATUS["free"]}
    names = {-1: "sensor down", 0: "congested", 1: "slow", 2: "free"}

    def traces(w):
        out = []
        for arr in (Y[w, h - 1], np.where(np.isnan(Y[w, h - 1]), np.nan, P[w, h - 1])):
            lev = level(arr, ff)
            out.append(go.Scatter(x=lon, y=lat, mode="markers", marker=dict(color=[colors[v] for v in lev], size=8),
                                  text=[f"{names[v]} · {s:.0f} mph" if v >= 0 else names[v] for v, s in zip(lev, np.nan_to_num(arr))],
                                  hoverinfo="text", showlegend=False))
        return out

    fig = make_subplots(1, 2, subplot_titles=("Actual", f"{name}, {5 * h} min ahead"))
    for c, tr in enumerate(traces(frames_idx[0]), start=1):
        fig.add_trace(tr, 1, c)
    fig.frames = [go.Frame(data=traces(w), name=f"{t[w]:%H:%M}") for w in frames_idx]
    steps = [dict(method="animate", label=f"{t[w]:%H:%M}", args=[[f"{t[w]:%H:%M}"], dict(mode="immediate", frame=dict(duration=0))])
             for w in frames_idx]
    title = f"Congestion levels, {pd.Timestamp(day):%a %d %b %Y} evening (green free, amber slow, red congested, grey down)"
    fig.update_layout(title=title,
                      sliders=[dict(steps=steps, currentvalue=dict(prefix="time "))], height=520,
                      plot_bgcolor=S.SURFACE, paper_bgcolor=S.SURFACE,
                      updatemenus=[dict(type="buttons", buttons=[dict(label="play", method="animate",
                                                                      args=[None, dict(frame=dict(duration=700))])])])
    for ax in ("xaxis", "xaxis2", "yaxis", "yaxis2"):
        fig.layout[ax].update(visible=False)
    path = FIG_DIR / "fig21_congestion_animation.html"
    fig.write_html(path, include_plotlyjs="cdn")
    return str(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="auto", help="prediction file; 'auto' = seed 0 of the best graph model in Phase 6")
    ap.add_argument("--baseline", default="xgb")
    args = ap.parse_args()
    if args.model == "auto":
        best = json.loads((ROOT / "results" / "phase6_findings.json").read_text())["best_graph_model"]
        args.model = {"STGAT (ours)": "stgat_s0", "DCRNN": "dcrnn_s0"}[best]
    g = load_adjacency()
    adj, ids = g["adj"], [str(s) for s in g["sensor_ids"]]
    n = len(ids)
    loc = load_locations(sensor_ids=ids)
    lon, lat = loc.longitude.to_numpy(), loc.latitude.to_numpy()
    edges = [(u, v) for u, v in zip(*np.nonzero(adj)) if u != v]
    test = load_split("test", keys=["Y", "M_y", "target_time"])
    Y, M, times = test["Y"], test["M_y"], test["target_time"]
    ff = free_flow()
    P_base = windows(np.load(PRED_DIR / f"{args.baseline}_test.npy", mmap_mode="r"), n)
    P_graph = windows(np.load(PRED_DIR / f"{args.model}_test.npy", mmap_mode="r"), n)
    names = ("XGBoost", "STGAT" if args.model.startswith("stgat") else "DCRNN")
    S.apply()
    out = {}
    fig, out["side_by_side"] = fig_side_by_side(Y, {names[0]: P_base, names[1]: P_graph}, times, ff, lon, lat, edges)
    save(fig, "fig18_congestion_map_side_by_side.png")
    fig, out["error_map"] = fig_error_map(Y, M, P_base, P_graph, lon, lat, edges, names)
    save(fig, "fig19_error_map.png")
    path, km = corridor(adj, ids)
    fig, out["time_space"] = fig_time_space(Y, {names[0]: P_base, names[1]: P_graph}, times, path, km, ff)
    save(fig, "fig20_time_space_diagram.png")
    out["animation"] = animated_map(Y, P_graph, times, ff, lon, lat, names[1])
    plt.close("all")
    (ROOT / "results" / "phase7_maps.json").write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
