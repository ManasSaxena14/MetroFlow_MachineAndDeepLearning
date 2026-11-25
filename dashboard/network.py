"""Network: replay of two test days on the map, with the hybrid's forecasts, ranges and alerts."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard import data as D
from dashboard import theme as T

VIEWS = ["Measured now", "Forecast, 30 min ahead", "What happened 30 min later", "Chance of a jam within the hour"]


def states(v: np.ndarray, ff: np.ndarray) -> np.ndarray:
    r = v / ff
    return np.where(np.isnan(v), "no reading", np.where(r < 0.5, "congested", np.where(r < 0.8, "slow", "free")))


def clock(step: int) -> str:
    return f"{step // 12:02d}:{step % 12 * 5:02d}"


def road_lines(s: pd.DataFrame, links: pd.DataFrame) -> tuple[list, list]:
    lon, lat = s.longitude.to_numpy(), s.latitude.to_numpy()
    x, y = [], []
    for a, b in zip(links.src, links.dst):
        x += [lon[a], lon[b], None]
        y += [lat[a], lat[b], None]
    return x, y


def base_map(s: pd.DataFrame, links: pd.DataFrame) -> go.Figure:
    x, y = road_lines(s, links)
    fig = go.Figure(go.Scattermap(lon=x, lat=y, mode="lines", line=dict(color="#3b4246", width=2), hoverinfo="skip", showlegend=False))
    fig.update_layout(map=dict(style="carto-darkmatter-nolabels", center=dict(lat=34.137, lon=-118.348), zoom=10.35), height=520,
                      margin=dict(l=0, r=0, t=0, b=0), showlegend=False, uirevision="map", font=dict(family=T.FONT),
                      hoverlabel=dict(bgcolor=T.RAISED, bordercolor=T.LINE, font=dict(family=T.FONT, color=T.INK)))
    return fig


def network_map(s, links, r, t: int, view: str, picked: int) -> go.Figure:
    fig, ff = base_map(s, links), s.free_flow.to_numpy()
    lon, lat = s.longitude.to_numpy(), s.latitude.to_numpy()
    fig.add_trace(go.Scattermap(lon=[lon[picked]], lat=[lat[picked]], mode="markers", hoverinfo="skip",
                                marker=dict(size=32, color=T.INK, opacity=0.2)))
    if view == VIEWS[3]:
        p = r["prob"][t]
        order = np.argsort(p)
        fig.add_trace(go.Scattermap(lon=lon[order], lat=lat[order], mode="markers", customdata=order,
                                    marker=dict(size=6 + 12 * p[order], color=p[order], cmin=0, cmax=1,
                                                colorscale=T.CHANCE),
                                    text=[f"Sensor {s.sensor_id[i]}: {100 * p[i]:.0f}% chance" for i in order], hoverinfo="text"))
        return fig
    v = {VIEWS[0]: r["actual"][12 + t], VIEWS[1]: r["forecast"][t, 5], VIEWS[2]: r["actual"][12 + t + 6]}[view]
    st_ = states(v, ff)
    for name, size in (("no reading", 6), ("free", 7), ("slow", 11), ("congested", 15)):
        sel = np.flatnonzero(st_ == name)
        fig.add_trace(go.Scattermap(lon=lon[sel], lat=lat[sel], mode="markers", marker=dict(size=size, color=T.STATE[name]), customdata=sel,
                                    text=[f"Sensor {s.sensor_id[i]}: {name}" + ("" if np.isnan(v[i]) else f", {v[i]:.0f} mph") for i in sel],
                                    hoverinfo="text"))
    return fig


def fan(r, t: int, i: int, ff: float) -> go.Figure:
    """Last hour, 80% range and forecast for the next hour, and what happened."""
    past, ahead = np.arange(max(t - 12, -12), t + 1), np.arange(t + 1, t + 13)
    hm = [clock(k % 288) for k in past] + [clock(k % 288) for k in ahead]
    xp, xa = list(range(len(past))), list(range(len(past), len(past) + 12))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xa + xa[::-1], y=list(r["hi"][t, :, i]) + list(r["lo"][t, :, i][::-1]), fill="toself", line=dict(width=0),
                             fillcolor=T.alpha(T.MODEL["Hybrid"], 0.22), name="80% range", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xp, y=r["actual"][12 + past, i], mode="lines", line=dict(color=T.INK, width=2.4), name="Measured", text=hm[:len(past)],
                             hovertemplate="%{text}: %{y:.0f} mph measured<extra></extra>"))
    fig.add_trace(go.Scatter(x=xa, y=r["forecast"][t, :, i], mode="lines", line=dict(color=T.MODEL["Hybrid"], width=2.6), name="Forecast",
                             text=[f"{a}: {f:.0f} mph, likely {lo:.0f} to {hi:.0f}" for a, f, lo, hi in
                                   zip(hm[len(past):], r["forecast"][t, :, i], r["lo"][t, :, i], r["hi"][t, :, i])], hovertemplate="%{text}<extra></extra>"))
    fig.add_trace(go.Scatter(x=xa, y=r["actual"][12 + ahead, i], mode="lines", line=dict(color=T.INK, width=2, dash="dot"), name="What happened",
                             text=hm[len(past):], hovertemplate="%{text}: %{y:.0f} mph<extra></extra>"))
    fig.add_hline(y=0.5 * ff, line=dict(color=T.STATE["congested"], width=1.2, dash="dash"))
    fig.add_annotation(x=0, y=0.5 * ff, text="jam level", showarrow=False, yshift=10, xanchor="left", font=dict(color=T.STATE["congested"], size=12))
    fig.add_vline(x=len(past) - 1, line=dict(color=T.MUTED, width=1))
    T.style(fig, 300, y="mph")
    ticks = list(range(0, len(hm), 6))
    fig.update_xaxes(tickvals=ticks, ticktext=[hm[k] for k in ticks], range=[0, len(hm) - 1])
    fig.update_yaxes(range=[0, 82], tickvals=[0, 20, 40, 60, 80])
    return fig


def day_line(r, ff: np.ndarray, t: int) -> go.Figure:
    """Congested sensors through the day: measured against what the model said half an hour earlier."""
    cong = np.nansum(r["actual"][12:300] < 0.5 * ff, axis=1)
    said = np.r_[np.full(6, np.nan), np.nansum(np.where(np.isnan(r["actual"][18:300]), np.nan, r["forecast"][:282, 5]) < 0.5 * ff, axis=1)]
    x = list(range(288))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=cong, mode="lines", line=dict(color=T.INK, width=2), name="Measured", text=[clock(k) for k in x],
                             hovertemplate="%{text}: %{y:.0f} congested<extra></extra>"))
    fig.add_trace(go.Scatter(x=x, y=said, mode="lines", line=dict(color=T.MODEL["Hybrid"], width=2), name="Forecast, 30 min before",
                             text=[clock(k) for k in x], hovertemplate="%{text}: %{y:.0f} forecast<extra></extra>"))
    fig.add_vline(x=t, line=dict(color=T.STATE["congested"], width=1.6))
    T.style(fig, 230, y="congested sensors", hovermode="x unified")
    fig.update_xaxes(tickvals=list(range(0, 288, 36)), ticktext=[clock(k) for k in range(0, 288, 36)], range=[0, 287])
    return fig


def page() -> None:
    info, s, links = D.js("replay/summary.json"), D.csv("replay/sensors.csv", dtype={"sensor_id": str}), D.csv("replay/links.csv")
    st.title("See the jam before it reaches the sensor")
    T.lede("A replay of Los Angeles freeways in June 2012. Move through the day and compare the half-hour forecast with what the "
           "detectors measured. The model never saw these days.")
    if not D.need(info, s, links, command="make operations"):
        return
    days = info["days"]
    a, c = st.columns([4, 1], vertical_alignment="bottom")
    with a:
        day = st.segmented_control("Day", list(days), default="busiest", key="day", required=True,
                                   format_func=lambda d: f"{d.capitalize()} weekday, {pd.Timestamp(days[d]['date']):%d %b}")
    with c:
        playing = st.toggle("Play", key="playing", help="Advance 15 minutes every two seconds")
    r = D.replay(day)
    st.session_state.setdefault("time", "15:30")
    ff = s.free_flow.to_numpy()

    def body() -> None:
        if st.session_state.get("playing") and st.session_state.get("_tick"):
            st.session_state["time"] = clock((int(st.session_state["time"][:2]) * 12 + int(st.session_state["time"][3:]) // 5 + 3) % 288)
        st.session_state["_tick"] = True
        label = st.select_slider("Time of day", [clock(k) for k in range(0, 288, 3)], key="time")
        t = int(label[:2]) * 12 + int(label[3:]) // 5
        view = st.segmented_control("Map shows", VIEWS, default=VIEWS[1], key="view", required=True)
        left, right = st.columns([1.55, 1], gap="large")
        busy = r["jam"].sum(0)
        with right:
            now, later, said = (states(x, ff) for x in (r["actual"][12 + t], r["actual"][12 + t + 6], r["forecast"][t, 5]))
            recent = r["alert"][max(0, t - 11):t + 1].any(0)
            T.readout(f"{label}, {pd.Timestamp(days[day]['date']):%A %d %B %Y}", [
                ("Congested now", str((now == "congested").sum())),
                ("Forecast in 30 min", str(((said == "congested") & (later != "no reading")).sum())),
                ("Measured 30 min later", str((later == "congested").sum())),
                ("Alerts in the last hour", str(int(recent.sum())))], hot=0)
            options = list(np.argsort(-busy, kind="stable"))
            pick = st.selectbox("Sensor", options, key="sensor",
                                format_func=lambda i: f"{s.sensor_id[i]}: {int(busy[i])} jam{'s' if busy[i] != 1 else ''} this day")
            T.show(fan(r, t, int(pick), float(ff[pick])), "fan")
        with left:
            if view == VIEWS[3]:
                T.key([("unlikely", T.CHANCE[0][1], 8), ("about half", T.CHANCE[1][1], 12), ("near certain", T.CHANCE[3][1], 16)])
            else:
                T.key([("free", T.STATE["free"], 8), ("slow, below 80% of free flow", T.STATE["slow"], 12),
                       ("congested, below 50%", T.STATE["congested"], 15), ("no reading", T.STATE["no reading"], 7)])
            with st.container(key="mapbox"):
                st.plotly_chart(network_map(s, links, r, t, view, int(pick)), width="stretch", theme=None, key="map",
                                config={"displayModeBar": False, "scrollZoom": False})
            T.show(day_line(r, ff, t), "day_line")

    st.fragment(body, run_every=2.0 if playing else None)()
    T.note(f"Days are chosen by rule: the test weekday with the most jams ({days['busiest']['jams']}) and the one with the median number "
           f"({days['typical']['jams']}). Alerts use the threshold set on validation data for about {info['budget_per_day']} false alarms a day "
           "across 207 sensors. This is a replay of recorded data, not a live feed.")
