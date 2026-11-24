"""Failure testing: broken detectors, late feeds and corrupted readings, without retraining."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from dashboard import data as D
from dashboard import theme as T

KIND = {"dead": ("Sensors switched off", "share of sensors off (%)", "The sensors report nothing, and the pipeline knows it."),
        "stuck": ("Sensors stuck at free flow", "share of sensors stuck (%)", "The sensors keep reporting their free-flow speed, so the fault looks like good news."),
        "delay": ("Feed arrives late", "delay (minutes)", "Every reading arrives late, so the newest part of the window is unknown."),
        "noise": ("Noisy readings", "random error (mph, standard deviation)", "Every reading carries random error."),
        "spikes": ("Corrupted readings", "share of readings corrupted (%)", "Random readings are replaced by a random speed.")}
MODELS = ["Hybrid", "DCRNN", "STGAT", "XGBoost", "Persistence"]


def page() -> None:
    s = D.csv("robustness/summary.csv")
    st.title("What happens when the detectors fail?")
    if not D.need(s, command="make operations"):
        return
    models = [m for m in MODELS if m in set(s.model)]
    a, b = st.columns([3.4, 1], vertical_alignment="bottom")
    kind = a.segmented_control("Failure", list(KIND), default="stuck", key="kind", required=True, format_func=lambda k: KIND[k][0])
    mins = b.segmented_control("Forecast", [15, 30, 60], default=30, key="f_min", required=True, format_func=lambda m: f"{m} min")
    clean = s[(s.kind == "clean") & (s.minutes == mins)].set_index("model")
    d = s[(s.kind == kind) & (s.minutes == mins)]
    worst = d[(d.model == "Hybrid")].sort_values("level").iloc[-1]
    others = d[(d.level == worst.level) & (d.model != "Hybrid")].set_index("model")
    T.lede(f"{KIND[kind][2]} At the most severe setting the hybrid's error rises <b>{worst.increase_pct:.0f}%</b>, from {clean.mae['Hybrid']:.2f} to "
           f"{worst.mae:.2f} mph. XGBoost rises {others.increase_pct['XGBoost']:.0f}% and repeating the last reading {others.increase_pct['Persistence']:.0f}%.")
    left, right = st.columns([1.35, 1], gap="large")
    with left:
        st.subheader(f"Error {mins} minutes ahead")
        fig = go.Figure()
        for m in models:
            e = d[d.model == m].sort_values("level")
            fig.add_trace(go.Scatter(x=[0, *e.level], y=[clean.mae[m], *e.mae], mode="lines+markers", name=m, marker=dict(size=8),
                                     line=dict(color=T.colour(m), width=2.8 if m == "Hybrid" else 2, dash="dot" if m == "Persistence" else "solid"),
                                     hovertemplate=m + ": %{y:.2f} mph<extra></extra>"))
        T.style(fig, 420, x=KIND[kind][1], y="mean absolute error (mph)", hovermode="x unified")
        fig.update_xaxes(tickvals=[0, *sorted(d.level.unique())])
        fig.update_yaxes(rangemode="tozero")
        T.show(fig, "fail")
    with right:
        st.subheader("Rise in error at the worst setting")
        q = d[d.level == worst.level].set_index("model").loc[models].sort_values("increase_pct")
        T.show(T.hbar(list(q.index), list(q.increase_pct.clip(lower=0)), [T.colour(m) for m in q.index], "{:.0f}", "%"), "rise")
        T.note("Relative to the same model with healthy detectors. A small rise is not the same as a small error: compare the chart on the left.")
    if kind in ("dead", "stuck"):
        st.subheader("The broken sensors, and their healthy neighbours")
        e = d[d.level == worst.level].set_index("model").loc[models]
        fig = go.Figure()
        for lab, col, colr in (("Broken sensors", "mae_affected", T.STATE["congested"]), ("Healthy sensors", "mae_healthy", T.QUIET)):
            fig.add_trace(go.Bar(x=models, y=e[col], name=lab, marker=dict(color=colr, cornerradius=3), hovertemplate="%{x}, " + lab.lower() + ": %{y:.2f} mph<extra></extra>"))
        T.style(fig, 320, y="mean absolute error (mph)", barmode="group")
        T.show(fig, "split")
        T.note(f"With {worst.level:.0f}% of sensors broken. Averages of three random draws of which sensors break.")
    cov = s[(s.model == "Hybrid") & (s.minutes == mins) & s.kind.isin([kind, "clean"])].sort_values("level")
    st.subheader("Does the 80% range still hold?")
    T.show(T.hbar(["Healthy"] + [f"{x:.0f}" for x in cov.level[cov.kind == kind]], list(cov.coverage_80_pct), [T.QUIET] + [T.MODEL["Hybrid"]] * (len(cov) - 1), "{:.1f}", "%"), "cov_fail")
    T.note("Share of true speeds inside the hybrid's 80% range by failure level. The ranges were calibrated on healthy detectors and were not adjusted.")
    with st.expander("How the test was run"):
        st.markdown("""
- Only the inputs of the test windows are broken. Targets stay the real readings and no model is retrained.
- Every third test window is used, one forecast per 15 minutes over all 24 days. The healthy error is measured on the same windows.
- The same sensors break in the switched-off and stuck tests, and a higher share contains the lower one, so levels are comparable.
- These are the seed 0 models.
""")
