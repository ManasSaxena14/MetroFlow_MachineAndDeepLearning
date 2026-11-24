"""MetroFlow dashboard. Reads only the pre-computed files in results/; it does not import the metroflow package.

Run locally: make dashboard
"""
from __future__ import annotations

import streamlit as st

from dashboard import alerts, data, failures, graph, method, models, network, overview, ranges, theme

st.set_page_config(page_title="MetroFlow: Los Angeles traffic forecasts", layout="wide", initial_sidebar_state="collapsed")
theme.apply_page()

# a page joins the menu once the result files it reads exist
PAGES = [
    (overview.page, "Overview", "overview", ()),
    (network.page, "Network", "network", ("replay/summary.json", "replay/busiest.npz")),
    (alerts.page, "Alerts", "alerts", ("alerts/operating_points.csv", "alerts/summary.json")),
    (ranges.page, "Ranges", "ranges", ("uncertainty/coverage.csv",)),
    (failures.page, "Failure tests", "failures", ("robustness/summary.csv",)),
    (models.page, "Models", "models", ()),
    (graph.page, "Road network", "road-network", ()),
    (method.page, "Method", "method", ()),
]
st.navigation([st.Page(fn, title=title, url_path=path, default=path == "overview")
               for fn, title, path, files in PAGES if data.exists(*files)], position="top").run()
