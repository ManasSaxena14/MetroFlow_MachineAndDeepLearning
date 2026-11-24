"""Overview: the question, the answer and where to look next."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from dashboard import data as D
from dashboard import theme as T

HYBRID, XGB = "Hybrid: DCRNN + XGBoost (ours)", "XGBoost per horizon"
REPO = "https://github.com/ManasSaxena14/TrafficFlowControl_AIML-DL_T036"


def example(ex) -> go.Figure:
    """A jam reaching the downstream sensor first (readings behind figure 5d of the AML report)."""
    fig = go.Figure()
    for c, n, colr in (("downstream", "Downstream sensor", T.RED), ("upstream", "Upstream sensor, 1.5 km before it", T.MODEL["DCRNN"])):
        fig.add_trace(go.Scatter(x=ex.time, y=ex[c], mode="lines+markers", name=n, line=dict(color=colr, width=2.4), marker=dict(size=6),
                                 hovertemplate=n + ", %{x|%H:%M}: %{y:.0f} mph<extra></extra>"))
    for when, colr, text, anchor in (("2012-05-02 06:40", T.RED, "reaches downstream 06:40", "right"), ("2012-05-02 06:50", T.MODEL["DCRNN"], "reaches upstream 06:50", "left")):
        fig.add_vline(x=when, line=dict(color=colr, width=1.2, dash="dot"))
        fig.add_annotation(x=when, y=3, text=text, showarrow=False, xanchor=anchor, xshift=-6 if anchor == "right" else 6, font=dict(color=T.INK_2, size=13))
    T.style(fig, 340, y="mph")
    fig.update_xaxes(tickformat="%H:%M")
    fig.update_yaxes(range=[0, 80])
    return fig


def page() -> None:
    f, p, s, ex = D.csv("comparison/models.csv"), D.js("findings/exploration.json"), D.csv("comparison/significance.csv"), D.csv("replay/propagation.csv", parse_dates=["time"])
    st.title("Forecast the jam, not just the speed")
    T.lede("MetroFlow forecasts freeway speed 5 to 60 minutes ahead for the 207 detectors of METR-LA in Los Angeles. It asks one question: does a model that "
           "reads the road network beat strong models that look at one detector at a time?")
    if not D.need(f, p, s, ex, command="make compare"):
        return
    i = f.set_index("model")
    h, x, d = i.loc[HYBRID], i.loc[XGB], i.loc["DCRNN"]
    days = s[(s.model == "DCRNN") & (s.baseline == XGB)].days_model_better.min()
    left, right = st.columns([1.5, 1], gap="large")
    with left:
        st.subheader("A jam travels against the traffic")
        T.show(example(ex), "o_example")
        T.note("2 May 2012, two linked detectors. The jam reaches the downstream one first and the upstream one ten minutes later. A model that reads one detector "
               "cannot see it coming. A model that reads the road can.")
    with right:
        T.readout("Error 60 minutes ahead, 24 test days", [
            ("Hybrid (ours)", f"{h.MAE_60min:.2f} mph"), ("DCRNN, graph model", f"{d.MAE_60min:.2f} mph"), ("XGBoost, one detector", f"{x.MAE_60min:.2f} mph"),
            ("Repeat the last reading", f"{i.loc['Persistence'].MAE_60min:.2f} mph")])
        T.note(f"Mean absolute error. The hybrid is {100 * (1 - h.MAE_60min / x.MAE_60min):.1f}% below XGBoost overall and "
               f"{100 * (1 - h.peak_MAE_60min / x.peak_MAE_60min):.1f}% below it in the weekday rush.")
    st.subheader("What the project found")
    pr = p["propagation"]
    ab = D.csv("comparison/ablations.csv").set_index("run")
    st.markdown(f"""
- **The graph helps, reliably.** DCRNN beats XGBoost on {int(days)} of {int(s.test_days.max())} test days, at every horizon tested.
- **It is the real road that matters.** Without the graph, our attention model loses {ab.loc['stgat_identity_s0', 'delta_60min_vs_main']:.2f} mph at 60 minutes. A random graph of the same size loses {ab.loc['stgat_random_s0', 'delta_60min_vs_main']:.2f}.
- **Warnings come from downstream.** Among links with a clear lead, the downstream detector moved first on {pr['downstream_first']} and the upstream one on {pr['upstream_first']}.
- **Trees and graphs are complementary.** A small tree model that corrects DCRNN lowers its 60-minute error by {100 * (1 - h.MAE_60min / d.MAE_60min):.1f}%.
""")
    st.markdown(f"Code, data pipeline, reports and slides: [GitHub repository]({REPO}). Data: METR-LA, March to June 2012. Nothing here is a live feed or controls traffic.")
