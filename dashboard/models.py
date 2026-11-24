"""Models: every forecaster on the same test days, the significance tests and the per-sensor baselines."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from dashboard import data as D
from dashboard import theme as T

SHORT = {"Hybrid: DCRNN + XGBoost (ours)": "Hybrid (ours)", "STGAT (ours)": "STGAT (ours)", "XGBoost per horizon": "XGBoost", "Multi-output Random Forest": "Random Forest",
         "LSTM (no graph)": "LSTM, no graph", "Ridge per sensor": "Ridge", "Persistence (last value)": "Persistence"}
UNIT = {"MAE": "mph", "RMSE": "mph", "MAPE": "%"}
GAIN = {"Hybrid: DCRNN + XGBoost (ours)": "Hybrid", "DCRNN": "DCRNN", "STGAT (ours)": "STGAT", "XGBoost + neighbours": "XGBoost + neighbours"}


def comparison() -> None:
    f, lng = D.csv("comparison/models.csv"), D.csv("comparison/by_horizon.csv", dtype={"horizon": str})
    if not D.need(f, lng, command="make compare"):
        return
    a, b, _ = st.columns([1, 1.7, 1])
    metric = a.segmented_control("Measure", list(UNIT), default="MAE", key="m_metric", required=True)
    mins = b.segmented_control("Minutes ahead", [15, 30, 60], default=60, key="m_min", required=True, format_func=lambda m: f"{m} min")
    col, unit = f"{metric}_{mins}min", UNIT[metric]
    d = f.sort_values(col)
    names = [SHORT.get(m, m) for m in d.model]
    left, right = st.columns([1, 1.05], gap="large")
    with left:
        st.subheader(f"{metric}, {mins} minutes ahead")
        T.show(T.hbar(names, list(d[col]), [T.colour(n) for n in names], "{:.2f}", f" {unit}", height=470), "m_rank")
        T.note("Lower is better. Graph models and the hybrid are means of three seeds.")
    with right:
        st.subheader("Error by how far ahead")
        fig, m = go.Figure(), metric.lower()
        for model in ("Persistence", "LSTM (no graph)", "XGBoost per horizon", "STGAT (ours)", "DCRNN", "Hybrid: DCRNN + XGBoost (ours)"):
            e = lng[(lng.model == model) & (lng.horizon != "avg")].assign(h=lambda x: x.horizon.astype(int) * 5).groupby("h")[m].mean()
            n = SHORT.get(model, model)
            fig.add_trace(go.Scatter(x=e.index, y=e, mode="lines", name=n, line=dict(color=T.colour(n), width=3 if n.startswith("Hybrid") else 2,
                                     dash="dot" if n in ("Persistence", "LSTM, no graph") else "solid"), hovertemplate=n + ": %{y:.2f} " + unit + "<extra></extra>"))
        T.style(fig, 470, x="minutes ahead", y=f"{metric} ({unit})", hovermode="x unified")
        fig.update_xaxes(tickvals=list(range(5, 61, 5)))
        fig.update_yaxes(rangemode="tozero")
        T.show(fig, "m_horizon")
    st.subheader("Rush hour, off-peak and weekend, 60 minutes ahead")
    d = f.sort_values("peak_MAE_60min")
    fig = go.Figure()
    for c, lab, colr in (("peak_MAE_60min", "Weekday peak", T.RED), ("offpeak_MAE_60min", "Weekday off-peak", T.MODEL["DCRNN"]), ("weekend_MAE_60min", "Weekend", T.GREYS["LSTM"])):
        fig.add_trace(go.Scatter(x=[SHORT.get(x, x) for x in d.model], y=d[c], mode="markers", name=lab, marker=dict(size=12, color=colr, line=dict(color=T.PAGE, width=1.5)),
                                 hovertemplate="%{x}, " + lab.lower() + ": %{y:.2f} mph<extra></extra>"))
    T.style(fig, 340, y="MAE (mph)")
    fig.update_yaxes(rangemode="tozero")
    T.show(fig, "m_period")
    with st.expander("Full table"):
        cols = ["model", "runs", *[f"{k}_{x}min" for x in (15, 30, 60) for k in UNIT], "peak_MAE_60min", "offpeak_MAE_60min", "weekend_MAE_60min"]
        st.dataframe(f[cols].replace({"model": SHORT}), hide_index=True, width="stretch")


def significance() -> None:
    s = D.csv("comparison/significance.csv")
    if not D.need(s, command="make compare"):
        return
    ref = st.segmented_control("Compared with", ["XGBoost per horizon", "DCRNN (same seed)"], default="XGBoost per horizon", key="sig_ref", required=True,
                               format_func=lambda x: {"XGBoost per horizon": "XGBoost"}.get(x, "DCRNN, hybrid only"))
    d = s[s.baseline == ref]
    fig = go.Figure()
    for m, n in GAIN.items():
        e = d[d.model == m].groupby("minutes").agg(g=("gain_mph", "mean"), lo=("ci95_low", "min"), hi=("ci95_high", "max"), days=("days_model_better", "min"))
        if e.empty:
            continue
        c = "#8f4c27" if n == "XGBoost + neighbours" else T.colour(n)
        fig.add_trace(go.Bar(x=[f"{k} min" for k in e.index], y=e.g, name=n, marker=dict(color=c, cornerradius=3), customdata=e.days,
                             error_y=dict(type="data", symmetric=False, array=e.hi - e.g, arrayminus=e.g - e.lo, color=T.INK, thickness=1.3, width=4),
                             hovertemplate=n + ": %{y:.3f} mph better, on at least %{customdata} of 24 days<extra></extra>"))
    T.style(fig, 380, y="error reduction (mph)", barmode="group")
    T.show(fig, "sig")
    T.note("Errors inside one day are correlated, so whole days are resampled: a paired bootstrap over 24 test days and a Diebold-Mariano test. Whiskers span the "
           "95% intervals of all seeds. The worst seed is shown for days and p-values.")
    t = d.groupby(["model", "minutes"], sort=False).agg(gain_pct=("gain_pct", "mean"), days=("days_model_better", "min"), p_boot=("p_bootstrap", "max"), p_dm=("p_dm", "max")).reset_index()
    for c in ("p_boot", "p_dm"):
        t[c] = [("< 0.0001" if v < 1e-4 else f"{v:.3f}") for v in t[c]]
    N = st.column_config.NumberColumn
    st.dataframe(t.replace({"model": SHORT}), hide_index=True, width="stretch", column_config={
        "model": "Model", "minutes": N("Minutes ahead"), "gain_pct": N("Error reduction", format="%.1f%%"), "days": N("Days better, of 24"),
        "p_boot": "p, bootstrap", "p_dm": "p, Diebold-Mariano"})


def baselines() -> None:
    b, acc, imp = D.csv("baselines/summary.csv"), D.csv("baselines/accuracy.csv"), D.csv("baselines/xgb_importance.csv", index_col="feature")
    if not D.need(b, acc, imp, command="make baselines accuracy"):
        return
    a, c, _ = st.columns([1.1, 1.1, 1.4])
    split = a.segmented_control("Block", ["test", "val"], default="test", key="b_split", required=True, format_func=lambda x: {"test": "Test", "val": "Validation"}[x])
    metric = c.segmented_control("Measure", list(UNIT), default="MAE", key="b_metric", required=True)
    d = b[b.split == split].sort_values(f"{metric}_avg")
    st.subheader("Six models that see one sensor at a time")
    names = [SHORT.get(m, m) for m in d.model]
    left, right = st.columns([1, 1.1], gap="large")
    with left:
        T.show(T.hbar(names, list(d[f"{metric}_avg"]), [T.colour(n) for n in names], "{:.2f}", f" {UNIT[metric]}"), "b_rank")
        T.note(f"Average over all 12 horizons. {names[0]} is lowest here.")
    with right:
        st.dataframe(d[["model", *[f"{metric}_{m}min" for m in (15, 30, 60)], f"{metric}_avg"]].replace({"model": SHORT}), hide_index=True, width="stretch", column_config={
            "model": "Model", **{f"{metric}_{m}min": st.column_config.NumberColumn(f"{m} min", format="%.2f") for m in (15, 30, 60)},
            f"{metric}_avg": st.column_config.NumberColumn("All horizons", format="%.2f")})
        w = acc.pivot(index="model", columns="minutes", values="within_5mph_pct").reset_index().replace({"model": SHORT})
        w.columns = ["Model", *[f"Within 5 mph, {m} min" for m in (15, 30, 60)]]
        st.dataframe(w, hide_index=True, width="stretch", column_config={x: st.column_config.NumberColumn(x, format="%.1f%%") for x in w.columns[1:]})
    st.subheader("What XGBoost looks at")
    x, n = st.columns([1.6, 1], gap="large")
    h = x.select_slider("Forecast horizon", [int(k) for k in imp.columns], value=6, format_func=lambda k: f"{5 * k} min")
    top_n = n.slider("Features shown", 5, 25, 12)
    top = 100 * imp[str(h)].sort_values(ascending=False).head(top_n)
    T.show(T.hbar(list(top.index), list(top), [T.MODEL["XGBoost"]] * len(top), "{:.1f}", "%", height=30 * len(top) + 40), "imp")
    T.note("Share of total gain. Every input is the sensor's own history, the clock or a fixed property of the sensor. None of it describes the road network.")
    with st.expander("XGBoost settings"):
        meta = D.js("baselines/xgb.json")
        st.json({"params": meta.get("params", {}), "trees per horizon": meta.get("best_rounds", {})}, expanded=True)


def page() -> None:
    f = D.csv("comparison/models.csv")
    st.title("Ten models, one set of test days")
    if len(f):
        i = f.set_index("model")
        h, x = i.loc["Hybrid: DCRNN + XGBoost (ours)"], i.loc["XGBoost per horizon"]
        gains = " / ".join(f"{100 * (1 - h[f'MAE_{m}min'] / x[f'MAE_{m}min']):.1f}%" for m in (15, 30, 60))
        T.lede(f"The hybrid is the most accurate by mean absolute error at 15, 30 and 60 minutes, <b>{gains}</b> below XGBoost, the best model that sees one sensor at a time.")
    tabs = st.tabs(["Comparison", "Is the gain real?", "Per-sensor baselines"])
    for tab, fn in zip(tabs, (comparison, significance, baselines)):
        with tab:
            fn()
