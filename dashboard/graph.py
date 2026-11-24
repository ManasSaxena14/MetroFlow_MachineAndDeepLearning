"""Why the road graph: the gap that per-sensor models leave, and where the graph model's gain comes from."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from dashboard import data as D
from dashboard import theme as T
from dashboard.overview import example


def gap() -> None:
    p1, p4 = D.js("findings/exploration.json"), D.js("findings/gap.json")
    ons, res, g = D.csv("gap/onset_vs_normal.csv"), D.csv("gap/residual_correlation.csv"), D.csv("gap/neighbour_gain.csv")
    if not D.need(p1, p4, ons, res, g, command="make eda gap"):
        return
    pr = p1["propagation"]
    st.subheader("Jams travel backwards")
    T.note(f"Among {pr['clear_edges']} road links with a clear lead or lag, the downstream sensor moved first on {pr['downstream_first']} and the upstream one on "
           f"{pr['upstream_first']}. The wave runs against traffic at about {pr['shockwave_kmh']:.0f} km/h, so the warning is at the next sensor down the road.")
    ex = D.csv("replay/propagation.csv", parse_dates=["time"])
    if len(ex):
        T.show(example(ex), "g_example")
        T.note("One event, 2 May 2012, two linked sensors. Both flow freely until the jam reaches the downstream one, then the upstream one ten minutes later.")
    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Errors peak just after a jam starts")
        fig = go.Figure()
        for model, name, c in (("xgb", "XGBoost", T.MODEL["XGBoost"]), ("persistence", "Persistence", T.GREYS["Persistence"])):
            d = ons[ons.model == model].pivot(index="horizon", columns="group", values="mae")
            fig.add_trace(go.Scatter(x=d.index * 5, y=d.onset, mode="lines", name=f"{name}, after an onset", line=dict(color=c, width=2.4), hovertemplate="%{y:.1f} mph<extra>" + name + ", onset</extra>"))
            fig.add_trace(go.Scatter(x=d.index * 5, y=d.normal, mode="lines", name=f"{name}, other times", line=dict(color=c, width=1.8, dash="dot"), hovertemplate="%{y:.1f} mph<extra>" + name + ", other</extra>"))
        T.style(fig, 380, x="minutes ahead", y="MAE (mph)")
        fig.update_yaxes(rangemode="tozero")
        T.show(fig, "g_onset")
        x = ons[(ons.model == "xgb") & ons.horizon.isin([3, 6, 12])].pivot(index="horizon", columns="group", values="mae")
        q = x.onset / x.normal
        T.note(f"At 15, 30 and 60 minutes XGBoost's error is {q.min():.1f} to {q.max():.1f} times larger in the half hour after an onset ({p4['test_onsets']:,} test onsets).")
    with right:
        st.subheader("Linked sensors make the same mistakes")
        d, ramp = res[res.model == "xgb"], {3: "#7d4022", 6: "#e0702f", 12: "#f7b48a"}
        groups = list(dict.fromkeys(d.group))
        fig = go.Figure()
        for h in sorted(d.horizon.unique()):
            v = d[d.horizon == h].set_index("group").mean_corr.loc[groups]
            fig.add_trace(go.Bar(x=groups, y=v, name=f"{5 * h} min ahead", marker=dict(color=ramp[h], cornerradius=3), hovertemplate="%{x}: %{y:.3f}<extra></extra>"))
        T.show(T.style(fig, 380, y="correlation of errors", barmode="group"), "g_res")
        T.note("Correlation between two sensors' XGBoost errors, by how many road links separate them. Errors a network-blind model shares along the road are information it missed.")
    st.subheader("Neighbour speeds help, downstream more than upstream")
    d = g[g.split == "test"].set_index("model")
    fig = go.Figure()
    for m, n, c in (("xgb + upstream", "Upstream neighbours", T.QUIET), ("xgb + downstream", "Downstream neighbours", "#9c5128"), ("xgb + both", "Both", T.MODEL["XGBoost"])):
        fig.add_trace(go.Bar(x=["15 min", "30 min", "60 min"], y=[d.loc[m, f"MAE_gain_{k}min_%"] for k in (15, 30, 60)], name=n, marker=dict(color=c, cornerradius=3),
                             hovertemplate=n + ": %{y:.1f}% lower error<extra></extra>"))
    T.show(T.style(fig, 320, y="lower error than XGBoost alone (%)", barmode="group"), "g_nb")
    with st.expander("Table"):
        st.dataframe(g, hide_index=True, width="stretch")


def source() -> None:
    ab, at, deg, ons = D.csv("comparison/ablations.csv"), D.csv("comparison/attention_shares.csv"), D.csv("comparison/gain_by_degree.csv"), D.csv("comparison/gain_at_onset.csv")
    if not D.need(ab, at, deg, ons, command="make phase5 compare"):
        return
    st.subheader("Take the graph apart, 60 minutes ahead")
    d = ab[ab.run.str.startswith("stgat") & (ab.run != "stgat_s0")].sort_values("delta_60min_vs_main", ascending=False)
    fig = go.Figure(go.Bar(x=d.delta_60min_vs_main, y=d.variant.str.capitalize(), orientation="h",
                           marker=dict(color=[T.MODEL["STGAT"] if v > 0 else T.QUIET for v in d.delta_60min_vs_main], cornerradius=3),
                           text=[f"{v:+.2f}" for v in d.delta_60min_vs_main], textposition="outside", textfont=dict(color=T.INK, size=14), cliponaxis=False,
                           hovertemplate="%{y}: %{x:+.3f} mph<extra></extra>"))
    T.style(fig, 340, x="change in MAE against the full STGAT (mph)", legend=False)
    fig.update_yaxes(autorange="reversed", showgrid=False, tickfont=dict(color=T.INK, size=14))
    T.show(fig, "g_abl")
    fw = ab.set_index("run").loc["dcrnn_forward_s0", "delta_60min_vs_main"]
    T.note(f"Removing the graph costs the most. Dropping dropout helped slightly. DCRNN without its backward walk, so that it cannot hear downstream, loses {fw:.2f} mph. Ablations use seed 0.")
    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Where STGAT looks")
        p = at.pivot(index="attends_to", columns="situation", values="mean_share_pct").loc[["self", "upstream neighbour", "downstream neighbour"]]
        fig = go.Figure()
        for c, n, colr in (("all other windows", "Other times", T.QUIET), ("slowdown ahead (30 min)", "Slowdown 30 min away", T.MODEL["STGAT"])):
            fig.add_trace(go.Bar(x=["Itself", "Upstream", "Downstream"], y=p[c], name=n, marker=dict(color=colr, cornerradius=3), text=[f"{v:.0f}%" for v in p[c]],
                                 textposition="outside", textfont=dict(color=T.INK), hovertemplate="%{x}: %{y:.1f}%<extra></extra>"))
        T.show(T.style(fig, 360, y="share of attention (%)", barmode="group"), "g_att")
        j = D.js("findings/graph_models.json")
        T.note(f"{j['sensors_shifting_downstream']} of {int(at.sensors.iloc[0])} sensors look further downstream when a slowdown is coming.")
    with right:
        st.subheader("Better-connected sensors gain more")
        q = deg.pivot(index="degree", columns="model", values="mae_60min").loc[["0-4", "5-8", "9-12", "13+"]]
        n = deg.drop_duplicates("degree").set_index("degree").sensors
        v = 100 * (1 - q["DCRNN"] / q["XGBoost per horizon"])
        fig = go.Figure(go.Bar(x=[f"{i} links<br>{n[i]} sensors" for i in q.index], y=v, marker=dict(color=T.MODEL["DCRNN"], cornerradius=3), text=[f"{k:.1f}%" for k in v],
                               textposition="outside", textfont=dict(color=T.INK), hovertemplate="%{y:.1f}% lower error<extra></extra>"))
        T.show(T.style(fig, 360, y="DCRNN against XGBoost, 60 min (%)", legend=False), "g_deg")
    st.subheader("Right after a jam starts")
    o = ons[(ons.group == "onset")].pivot(index="model", columns="minutes", values="mae")
    fig = go.Figure()
    for m, nme in (("XGBoost per horizon", "XGBoost"), ("STGAT (ours)", "STGAT"), ("DCRNN", "DCRNN"), ("Hybrid: DCRNN + XGBoost (ours)", "Hybrid")):
        fig.add_trace(go.Bar(x=["15 min", "30 min", "60 min"], y=o.loc[m, [15, 30, 60]], name=nme, marker=dict(color=T.colour(nme), cornerradius=3), hovertemplate=nme + ": %{y:.1f} mph<extra></extra>"))
    T.show(T.style(fig, 340, y="MAE in the half hour after an onset (mph)", barmode="group"), "g_ons2")
    T.note("An hour ahead the jam has usually not started anywhere yet, and XGBoost's calendar knowledge is slightly better than DCRNN. That is what the hybrid adds.")


def page() -> None:
    st.title("Why the road network matters")
    T.lede("Congestion moves from sensor to sensor, against the traffic. A model that reads one sensor cannot see it coming. A model that reads the road can.")
    tabs = st.tabs(["What per-sensor models miss", "Where the gain comes from"])
    for tab, fn in zip(tabs, (gap, source)):
        with tab:
            fn()
