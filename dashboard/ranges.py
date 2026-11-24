"""Forecast ranges: do the stated 80% and 90% ranges hold on days the calibration never saw, and where do they fail?"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard import data as D
from dashboard import theme as T

SLICES = ["all", "weekday peak", "weekday off-peak", "weekend", "truly free", "truly slow", "truly congested", "30 min after a jam onset"]
NAMES = {"all": "All forecasts", "truly free": "Traffic was free", "truly slow": "Traffic was slow", "truly congested": "Traffic was congested",
         "30 min after a jam onset": "Just after a jam starts"}
METHOD = {"adaptive": ("Adaptive range", T.MODEL["Hybrid"], "circle"), "constant": ("One fixed width", T.GREYS["LSTM"], "diamond")}


def page() -> None:
    cov, sl, day, rel = (D.csv(f"uncertainty/{n}.csv") for n in ("coverage", "by_slice", "by_day", "reliability"))
    st.title("How sure is the forecast?")
    if not D.need(cov, sl, day, rel, command="make operations"):
        return
    a, b, _ = st.columns([1.5, 1, 1], vertical_alignment="bottom")
    mins = a.segmented_control("Minutes ahead", [15, 30, 60], default=30, key="u_min", required=True, format_func=lambda m: f"{m} min")
    target = b.segmented_control("Range", [80, 90], default=80, key="u_target", required=True, format_func=lambda m: f"{m}%")
    c = cov[(cov.minutes == mins) & (cov.target_pct == target)].set_index("method")
    d = sl[(sl.minutes == mins) & (sl.target_pct == target)]
    g = {m: d[d.method == m].set_index("slice") for m in METHOD}
    free, jam = g["adaptive"].median_width_mph["truly free"], g["adaptive"].median_width_mph["truly congested"]
    T.lede(f"The {target}% range held the true speed <b>{c.coverage_pct['adaptive']:.1f}%</b> of the time on the test days. It is <b>{free:.0f} mph</b> wide when traffic "
           f"flows freely and <b>{jam:.0f} mph</b> wide in congestion. Just after a jam starts it held only "
           f"<b>{g['adaptive'].coverage_pct['30 min after a jam onset']:.0f}%</b> of the time.")

    left, right = st.columns([1.25, 1], gap="large")
    with left:
        st.subheader("How often the range held the truth")
        fig = go.Figure()
        ys = [NAMES.get(s, s.capitalize()) for s in SLICES]
        for s, y in zip(SLICES, ys):
            v = [g[m].coverage_pct[s] for m in METHOD]
            fig.add_trace(go.Scatter(x=v, y=[y, y], mode="lines", line=dict(color=T.LINE, width=2), showlegend=False, hoverinfo="skip"))
        for m, (name, col, sym) in METHOD.items():
            fig.add_trace(go.Scatter(x=[g[m].coverage_pct[s] for s in SLICES], y=ys, mode="markers", name=name,
                                     marker=dict(size=12, color=col, symbol=sym, line=dict(color=T.PAGE, width=1.5)),
                                     customdata=[g[m].mean_width_mph[s] for s in SLICES],
                                     hovertemplate=name + ": %{x:.1f}%, mean width %{customdata:.1f} mph<extra></extra>"))
        fig.add_vline(x=target, line=dict(color=T.INK, width=1.2, dash="dash"), annotation_text=f"target {target}%", annotation_position="bottom right",
                      annotation_font=dict(color=T.INK_2, size=12))
        T.style(fig, 430, x="share of true speeds inside the range (%)")
        fig.update_yaxes(autorange="reversed", showgrid=False, tickfont=dict(color=T.INK, size=14))
        fig.update_xaxes(showgrid=True, range=[0, 102])
        T.show(fig, "slices")
        T.note("The promise is the target on average. It is not a promise for a slice chosen by what later happened, such as the half hour after a jam "
               "starts. Those rows show where the range should not be trusted.")
    with right:
        st.subheader("Is a 70% jam chance really 70%?")
        q = rel[rel.minutes == mins]
        fig = go.Figure(go.Scatter(x=[0, 100], y=[0, 100], mode="lines", line=dict(color=T.MUTED, width=1.2, dash="dash"), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=100 * q.mean_probability, y=100 * q.observed, mode="lines+markers", line=dict(color=T.MODEL["Hybrid"], width=2.4),
                                 marker=dict(size=9), customdata=q.n, hovertemplate="said %{x:.0f}%, happened %{y:.0f}% (%{customdata:,} forecasts)<extra></extra>"))
        T.style(fig, 430, x="stated chance of congestion (%)", y="how often it happened (%)", legend=False)
        fig.update_xaxes(range=[0, 100])
        fig.update_yaxes(range=[0, 100])
        T.show(fig, "reliability")
        b80 = cov[(cov.minutes == mins) & (cov.target_pct == 80) & (cov.method == "adaptive")].iloc[0]
        T.note(f"Points on the dashed line are perfectly calibrated. Brier score {b80.brier:.4f}, against {b80.brier_point:.4f} for a plain yes/no from the "
               f"forecast and {b80.brier_base_rate:.4f} for always stating the average rate (lower is better).")

    st.subheader("The same 80% range, day by day")
    e = day[(day.minutes == mins) & (day.method == "adaptive")].assign(d=lambda x: pd.to_datetime(x.day))
    fig = go.Figure(go.Scatter(x=e.d, y=e.coverage_pct, mode="lines+markers", line=dict(color=T.MODEL["Hybrid"], width=2), marker=dict(size=8),
                               hovertemplate="%{x|%a %d %b}: %{y:.1f}%<extra></extra>"))
    fig.add_hline(y=80, line=dict(color=T.INK, width=1.2, dash="dash"))
    T.style(fig, 260, y="inside the range (%)", legend=False)
    fig.update_yaxes(range=[min(60, e.coverage_pct.min() - 4), 100])
    fig.update_xaxes(tickformat="%d %b")
    T.show(fig, "byday")
    T.note(f"Daily coverage ran from {e.coverage_pct.min():.0f}% to {e.coverage_pct.max():.0f}%.")
    st.subheader("Fixed against adaptive")
    t = cov[cov.target_pct == target].replace({"method": {k: v[0] for k, v in METHOD.items()}})
    N = st.column_config.NumberColumn
    st.dataframe(t[["minutes", "method", "coverage_pct", "mean_width_mph", "median_width_mph", "interval_score"]], hide_index=True, width="stretch", column_config={
        "minutes": N("Minutes ahead"), "method": "Method", "coverage_pct": N("Held the truth", format="%.1f%%"), "mean_width_mph": N("Mean width (mph)", format="%.1f"),
        "median_width_mph": N("Median width (mph)", format="%.1f"), "interval_score": N("Interval score (lower is better)", format="%.1f")})
    with st.expander("How the range is made"):
        st.markdown("""
- The hybrid's corrector was fitted on the validation days, so its own validation errors look too small. The days are split into four interleaved folds and each fold is forecast by a corrector that never saw it, with a two-hour gap around every held-out day.
- Those errors are grouped by minutes ahead, by the traffic state the forecast predicts, and by how far DCRNN and XGBoost disagree. A wide disagreement means a wide range.
- The group's error quantiles, with the conformal small-sample correction, give the range. The full distribution gives the chance of congestion.
- The test days were scored once, after everything was fixed.
""")
