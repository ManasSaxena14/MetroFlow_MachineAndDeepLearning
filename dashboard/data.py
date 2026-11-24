"""Cached readers of the files in results/. The dashboard computes nothing a results file does not already hold."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

RESULTS = Path(__file__).resolve().parents[1] / "results"


@st.cache_data
def csv(path: str, **kw) -> pd.DataFrame:
    f = RESULTS / path
    return pd.read_csv(f, **kw) if f.exists() else pd.DataFrame()


@st.cache_data
def js(path: str) -> dict:
    f = RESULTS / path
    return json.loads(f.read_text()) if f.exists() else {}


@st.cache_data
def replay(name: str) -> dict[str, np.ndarray]:
    f = RESULTS / "replay" / f"{name}.npz"
    if not f.exists():
        return {}
    with np.load(f) as z:
        out = {k: z[k] for k in z.files}
    for k in ("actual", "forecast", "lo", "hi"):
        out[k] = np.where(out[k] == 255, np.nan, out[k] / 2.0).astype(np.float32)
    out["prob"] = out["prob"] / 100
    return out


def exists(*paths: str) -> bool:
    return all((RESULTS / p).exists() for p in paths)


def need(*tables, command: str) -> bool:
    """True when every table has rows; otherwise explains which command creates the missing files."""
    if all(len(t) for t in tables):
        return True
    st.info(f"These results have not been created yet. Run `{command}` in the project, then reload this page.")
    return False
