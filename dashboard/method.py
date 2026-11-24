"""Method: the data, the protocol and what not to conclude."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard import data as D
from dashboard import theme as T


def page() -> None:
    p, h = D.js("findings/exploration.json"), D.js("hybrid/summary.json")
    st.title("How this was measured")
    T.lede("One protocol for every model: the same past hour in, the same next hour out, the same masked test targets.")
    left, right = st.columns([1.2, 1], gap="large")
    with right:
        if p:
            m, g = p["missingness"], p["graph"]
            T.readout("METR-LA, Los Angeles", [("Sensors", f"{p['data']['sensors']}"), ("Five-minute readings", f"{p['data']['timestamps']:,}"),
                                            ("Road links", f"{g['edges']:,}"), ("Readings lost to outages", f"{m['zero_pct']}%"), ("Median speed", f"{p['distribution']['median']:.0f} mph")])
            T.note(f"{m['network_outage_pct']} of the {m['zero_pct']} points of outage are whole-network blackouts. Outages are masked and never treated as traffic.")
    with left:
        st.subheader("Protocol")
        st.markdown("""
- **Split by time.** 70% train, 10% validation, 20% test: the last 24 days. Windows whose targets cross a boundary are dropped, so no target is in two blocks.
- **Nothing from the future.** The scaler, the imputation table and the free-flow speeds use training data only.
- **Choices on validation.** Hyper-parameters, early stopping and thresholds never see the test block.
- **Days, not rows.** Significance and the alert intervals resample whole days, because errors within a day move together.
""")
    st.subheader("Splits")
    st.dataframe(pd.DataFrame([("Train", 22933, "1 Mar 2012", "23 May 2012"), ("Validation", 3417, "23 May 2012", "4 Jun 2012"), ("Test", 6843, "4 Jun 2012", "27 Jun 2012")],
                              columns=["Block", "Windows", "First target", "Last target"]), hide_index=True, width="stretch")
    st.subheader("The models")
    st.markdown("""
- **XGBoost** sees one sensor: 46 features of its own history, the clock and fixed sensor properties. One model per horizon.
- **DCRNN** (Li et al., 2018) spreads information along the directed road graph, both ways, inside a recurrent encoder and decoder.
- **STGAT** (ours) summarises each sensor's hour and lets it attend to its road neighbours in both directions.
- **Hybrid** (ours) is DCRNN plus small, regularised trees that learn DCRNN's remaining error on the validation days.
""")
    st.subheader("What not to conclude")
    rounds = [v for r in h.get("runs", {}).values() for v in r["corrector_rounds"].values()]
    capped = f"{sum(v + 50 >= 1500 for v in rounds)} of {len(rounds)}" if rounds else "most"
    operations = ("- **Alerts, ranges and failure tests use one retrained seed.** Tables on the Models and Road network pages are the original three-seed runs.\n"
                  "- **Ranges are honest on average, not at every moment.** Coverage drops just after a jam starts. The Ranges page shows by how much.\n"
                  if D.exists("alerts/summary.json", "uncertainty/summary.json") else "")
    st.markdown(f"""
- **One city, one season, 2012.** Transfer to other roads or to current traffic is untested. Nothing here is a live feed.
{operations}- **Some trees stopped at their round limit:** XGBoost at every horizon from 20 minutes on, and the hybrid's corrector in {capped} fits. Longer training might lower both errors.
- **The hybrid is fitted on validation data** that the baselines used only for tuning.
- **No traffic control.** METR-LA has no signals and nothing here controls real traffic.
""")
