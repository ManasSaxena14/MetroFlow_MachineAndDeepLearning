"""Alerts: share of jams warned about, how early, and the false alarms that costs."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard import data as D
from dashboard import theme as T

PROB = "Hybrid, calibrated probability"
ORDER = ["Hybrid", PROB, "DCRNN", "XGBoost", "Calendar (historical average)", "Trend rule (no model)"]
LOOK = {"Hybrid": (T.MODEL["Hybrid"], "solid"), PROB: (T.MODEL["Hybrid"], "dash"), "DCRNN": (T.MODEL["DCRNN"], "solid"),
        "XGBoost": (T.MODEL["XGBoost"], "solid"), "Trend rule (no model)": (T.GREYS["LSTM"], "dot"), "Calendar (historical average)": (T.QUIET, "dot")}


def page() -> None:
    ops, curve = D.csv("alerts/operating_points.csv"), D.csv("alerts/curve.csv")
    leads, daily, per, info = D.csv("alerts/lead_times.csv"), D.csv("alerts/daily.csv"), D.csv("alerts/by_period.csv"), D.js("alerts/summary.json")
    st.title("How many jams would operators be warned about?")
    if not D.need(ops, curve, leads, daily, per, command="make operations"):
        return
    MAIN = info["main_model"]
    a, b, _ = st.columns([1.2, 2.2, 0.2], vertical_alignment="bottom", gap="large")
    budget = a.select_slider("False alarms allowed per day, whole network", sorted(ops.budget_per_day.unique()), value=info["headline_budget"])
    lead = b.segmented_control("Warned at least this far ahead", [5, 10, 15, 20, 30], default=10, key="lead", required=True, format_func=lambda m: f"{m} min")
    o = ops[ops.budget_per_day == budget].set_index("model").loc[ORDER]
    best = o.loc[MAIN]
    T.lede(f"With a threshold set for {budget} false alarms a day on validation data, the hybrid forecast warned of <b>{best[f'recall_{lead}min_pct']:.0f}%</b> of "
           f"the {int(best.jams):,} test jams at least {lead} minutes ahead. It raised <b>{best.false_alarms_per_day:.0f}</b> false alarms a day across 207 sensors, "
           f"and <b>{best.precision_pct:.0f}%</b> of its alerts were followed by a jam within the hour.")

    left, right = st.columns([1, 1.1], gap="large")
    with left:
        st.subheader(f"Jams warned about at least {lead} min ahead")
        ci = lead == 10
        T.show(T.hbar(ORDER, list(o[f"recall_{lead}min_pct"]), [LOOK[m][0] for m in ORDER], "{:.0f}", "%",
                      lo=list(o.recall_lo) if ci else None, hi=list(o.recall_hi) if ci else None), "recall")
        T.note("Every model uses its own threshold, set to the same false-alarm budget on validation. " +
               ("Whiskers are 95% intervals from resampling whole test days." if ci else "Day-level intervals are shown for the 10-minute view."))
        if o.loc[PROB, f"recall_{lead}min_pct"] < best[f"recall_{lead}min_pct"]:
            T.note(f"The calibrated probability is the weaker alert here. It fires later: its median first warning is {o.loc[PROB].median_lead_min:.0f} minutes, against "
                   f"{best.median_lead_min:.0f} for the {MAIN.lower()} forecast. The main alert was picked by validation recall, not by test results.")
    with right:
        st.subheader("The trade-off on the test days")
        fig = go.Figure()
        for m in ORDER:
            d = curve[(curve.model == m) & (curve.false_alarms_per_day.between(5, 400))].sort_values("false_alarms_per_day")
            fig.add_trace(go.Scatter(x=d.false_alarms_per_day, y=d[f"recall_{lead}min_pct"], mode="lines", name=m,
                                     line=dict(color=LOOK[m][0], dash=LOOK[m][1], width=2.6 if m == MAIN else 2),
                                     hovertemplate=f"{m}: %{{y:.0f}}% of jams, %{{x:.0f}} false alarms a day<extra></extra>"))
        fig.add_trace(go.Scatter(x=[best.false_alarms_per_day], y=[best[f"recall_{lead}min_pct"]], mode="markers", showlegend=False, hoverinfo="skip",
                                 marker=dict(size=12, color=T.MODEL["Hybrid"], line=dict(color=T.PAGE, width=2))))
        T.style(fig, 400, x="false alarms per day, whole network (log scale)", y="jams warned about (%)")
        fig.update_xaxes(type="log", tickvals=[5, 10, 25, 50, 100, 200, 400])
        fig.update_yaxes(rangemode="tozero")
        T.show(fig, "curve")

    left, right = st.columns([1, 1.1], gap="large")
    with left:
        st.subheader("How early the first warning came")
        d = leads[(leads.model == MAIN)].set_index("lead_min").jams
        x = [f"{k} min" if k else "missed" for k in d.index]
        fig = go.Figure(go.Bar(x=x, y=100 * d / d.sum(), marker=dict(color=[T.QUIET_2 if k == 0 else T.MODEL["Hybrid"] for k in d.index], cornerradius=3),
                               hovertemplate="%{x}: %{y:.1f}% of jams<extra></extra>"))
        T.show(T.style(fig, 320, y="share of jams (%)", legend=False), "leads")
        T.note(f"Measured at {info['headline_budget']} false alarms a day, from the first alert in the hour before each jam.")
    with right:
        st.subheader("Day by day")
        d = daily[daily.model == MAIN].copy()
        d["pct"] = 100 * d.warned_10min / d.jams.where(d.jams > 0)
        weekend = pd.to_datetime(d.day).dt.dayofweek >= 5
        fig = go.Figure(go.Bar(x=pd.to_datetime(d.day), y=d.pct, marker=dict(color=[T.QUIET_2 if w else T.MODEL["Hybrid"] for w in weekend], cornerradius=3),
                               customdata=d[["jams", "false_alarms"]], hovertemplate="%{x|%a %d %b}: %{y:.0f}% of %{customdata[0]} jams, %{customdata[1]} false alarms<extra></extra>"))
        T.style(fig, 320, y="jams warned about (%)", legend=False)
        fig.update_xaxes(tickformat="%d %b")
        T.show(fig, "daily")
        T.note(f"At least 10 minutes ahead, at {info['headline_budget']} false alarms a day. Weekends are grey.")

    st.subheader("All models at this setting")
    t = o.reset_index()[["model", "precision_pct", "false_alarms_per_day", *[f"recall_{k}min_pct" for k in (5, 10, 20, 30)], "median_lead_min"]]
    pct = st.column_config.NumberColumn
    st.dataframe(t, hide_index=True, width="stretch", column_config={
        "model": "Model", "precision_pct": pct("Alerts that were right", format="%.0f%%"), "false_alarms_per_day": pct("False alarms a day", format="%.0f"),
        **{f"recall_{k}min_pct": pct(f"Warned {k}+ min", format="%.0f%%") for k in (5, 10, 20, 30)}, "median_lead_min": pct("Median warning", format="%.0f min")})
    p = per[per.model == MAIN].set_index("period")
    st.markdown(f"At {info['headline_budget']} false alarms a day the hybrid warned of {p.recall_10min_pct['peak']:.0f}% of weekday-peak jams 10 minutes ahead, "
                f"{p.recall_10min_pct['offpeak']:.0f}% of weekday off-peak jams and {p.recall_10min_pct['weekend']:.0f}% of weekend jams.")
    with st.expander("Definitions"):
        st.markdown(f"""
- **Jam:** three readings in a row below half of the sensor's own free-flow speed, after 30 minutes without one. The test block holds {info['test_jams']:,} of them over {info['test_days']:.1f} days.
- **Alert:** raised only while the sensor is flowing. It then stays quiet for {info['quiet_min']} minutes, so one event cannot raise a stream of alerts.
- **Right:** a jam starts at that sensor 5 to 60 minutes later. An alert at the moment the jam starts does not count as a warning.
- **False alarm:** no jam within the hour. Alerts followed by an hour that is more than half outage are not scored.
- **Trend rule:** extrapolates the last 15 minutes. **Calendar:** alerts on the usual daily pattern.
- **Thresholds** come from the validation days. The trade-off curve uses the test days and is for description only.
""")
