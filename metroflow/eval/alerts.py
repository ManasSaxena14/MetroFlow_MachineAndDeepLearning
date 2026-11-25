"""Congestion alerts: what share of jams would an operator have been warned about, how early, at what cost in false alarms?

    python -m metroflow.eval.alerts [--seed 0]

Jam start    the first of three consecutive real readings below half of the sensor's own free-flow speed, after 30 minutes
             without such a reading (and with at least four real readings in those 30 minutes).
Alert        raised for a sensor at a forecast origin when it is flowing (a real reading now, not below half of free flow)
             and the score passes a threshold; the sensor then stays quiet for 60 minutes.
Hit          a jam starts at that sensor 5 to 60 minutes after the alert. Lead time = jam start minus first such alert.
False alarm  no jam starts within 60 minutes. Alerts whose next hour is more than half outage are not scored.

Scores       forecasts: the lowest forecast speed over the next 60 minutes as a share of free flow (lower = more alarming);
             probability: the highest calibrated congestion probability over the next 60 minutes;
             trend rule: the last 15-minute change extrapolated, no model;  calendar: the historical average.

Every threshold is chosen on the validation block (out-of-fold forecasts for the hybrid), at a fixed budget of false alarms
per day over the whole network. The main score is the one with the best validation recall at the headline budget. The test
block only reports what those choices deliver.

Writes results/alerts/{operating_points,curve,daily,lead_times,by_period}.csv and summary.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from metroflow.data.dataset import load_split
from metroflow.data.load import load_speed
from metroflow.data.preprocess import load_config
from metroflow.eval import uncertainty as unc
from metroflow.eval.evaluate import period_labels
from metroflow.models.simple import HistoricalAverage
from metroflow.train.baselines import ROOT, train_end

OUT = ROOT / "results" / "alerts"
N = 207
HORIZON = 12
RUN, CLEAR, MIN_REAL, QUIET = 3, 6, 4, 12
BUDGETS = (25, 50, 100, 200)                     # false alarms per day over all 207 sensors
HEADLINE_BUDGET = 50
LEADS = (5, 10, 15, 20, 30)


# --------------------------------------------------------------------------- definitions
def jam_starts(v: np.ndarray, ff: np.ndarray, frac: float = unc.CONGESTED) -> np.ndarray:
    """(T, N) bool from real speeds v (NaN = outage). Labels only, never a model input."""
    real, slow = ~np.isnan(v), np.nan_to_num(v, nan=np.inf) < frac * ff
    out = np.zeros(v.shape, bool)
    for t in range(CLEAR, len(v) - RUN + 1):
        out[t] = slow[t:t + RUN].all(0) & ~slow[t - CLEAR:t].any(0) & (real[t - CLEAR:t].sum(0) >= MIN_REAL)
    return out


def raise_alerts(score: np.ndarray, eligible: np.ndarray, threshold: float, quiet: int = QUIET) -> np.ndarray:
    """(S, N) bool: score >= threshold at an eligible sensor that has not alerted in the last `quiet` origins."""
    hot = (score >= threshold) & eligible
    out, last = np.zeros_like(hot), np.full(hot.shape[1], -10**6)
    for t in range(len(hot)):
        fire = hot[t] & (t - last >= quiet)
        out[t] = fire
        last[fire] = t
    return out


def match(alerts: np.ndarray, starts: np.ndarray, real: np.ndarray, horizon: int = HORIZON) -> dict:
    """alerts (S, N) at origins 0..S-1; starts and real (S + horizon, N) on the same clock (row t = origin t's reading)."""
    S = len(alerts)
    cum = np.concatenate([np.zeros((1, starts.shape[1]), int), np.cumsum(starts, 0)])
    ahead = cum[np.arange(S) + horizon + 1] - cum[np.arange(S) + 1]                      # jams starting in (t, t + horizon]
    creal = np.concatenate([np.zeros((1, real.shape[1]), int), np.cumsum(real, 0)])
    seen = creal[np.arange(S) + horizon + 1] - creal[np.arange(S) + 1]
    scored = alerts & ((ahead > 0) | (seen >= horizon / 2))
    hit = scored & (ahead > 0)
    ts, ns = np.nonzero(starts[horizon:S])                                               # jams with a full hour of origins before them
    ts = ts + horizon
    back = np.stack([alerts[ts - k, ns] for k in range(1, horizon + 1)], axis=1)         # alert k steps before the jam?
    lead = np.where(back.any(1), 5 * (horizon - np.argmax(back[:, ::-1], axis=1)), 0)    # earliest alert in the hour
    return {"scored": scored, "hit": hit, "false": scored & ~hit, "jam_t": ts, "jam_n": ns, "lead": lead}


def metrics(m: dict, days: float) -> dict:
    n_alert, n_hit, n_jam = int(m["scored"].sum()), int(m["hit"].sum()), len(m["lead"])
    warned = m["lead"][m["lead"] > 0]
    return {"alerts": n_alert, "jams": n_jam, "precision_pct": 100 * n_hit / max(n_alert, 1),
            "false_alarms_per_day": (n_alert - n_hit) / days,
            **{f"recall_{L}min_pct": 100 * float((m["lead"] >= L).mean()) for L in LEADS},
            "median_lead_min": float(np.median(warned)) if len(warned) else 0.0}


def threshold_for_budget(score, eligible, starts, real, days: float, budget: float, n_grid: int = 60) -> float:
    """Lowest threshold (most sensitive) whose false alarms per day stay within the budget, by bisection on score ranks."""
    pool = np.sort(score[eligible & np.isfinite(score)])
    lo, hi = 0, len(pool) - 1                                     # hi always satisfies the budget (or is the strictest there is)
    for _ in range(n_grid):
        if hi - lo <= 1:
            break
        mid = (lo + hi) // 2
        m = match(raise_alerts(score, eligible, pool[mid]), starts, real)
        if m["false"].sum() / days <= budget:
            hi = mid
        else:
            lo = mid
    return float(pool[hi])


# --------------------------------------------------------------------------- scores
def to_windows(P: np.ndarray) -> np.ndarray:
    return np.asarray(P, dtype=np.float32).reshape(-1, N, HORIZON)                       # (S, N, 12)


def forecast_score(P: np.ndarray, ff: np.ndarray) -> np.ndarray:
    return -(to_windows(P).min(-1) / ff)


def main_model(points: pd.DataFrame) -> str:
    """The score with the best VALIDATION recall (10+ minutes ahead) at the headline budget."""
    p = points[points.budget_per_day == HEADLINE_BUDGET]
    return str(p.loc[p.val_recall_10min_pct.idxmax(), "model"])


def probability_score(conf: unc.GroupedConformal, d: dict) -> np.ndarray:
    p = np.stack([conf.prob_below(d["f"][:, h], d["gap"][:, h], d["ff"], h) for h in range(HORIZON)], axis=1)
    return p.reshape(-1, N, HORIZON).max(-1)


def trend_score(v: np.ndarray, origins: np.ndarray, ff: np.ndarray) -> np.ndarray:
    """Persistence of the trend: speed now plus its last 15-minute change, as a share of free flow."""
    now, past = v[origins], v[origins - 3]
    s = -((now + (now - past)) / ff)
    return np.where(np.isnan(s), -np.inf, s)


def prepare(split: str, data: dict, v: np.ndarray, starts: np.ndarray, ff: np.ndarray, conf) -> dict:
    o = load_split(split, keys=["origin_idx", "target_time"])
    org = o["origin_idx"]
    assert (np.diff(org) == 1).all(), "alert scoring needs contiguous origins"
    d = data[split]
    now = v[org]
    cal = HistoricalAverage().fit(load_speed(), train_end()).predict(o["target_time"])
    clock = np.arange(org[0], org[-1] + HORIZON + 1)
    return {"origins": org, "eligible": ~np.isnan(now) & (now >= unc.CONGESTED * ff), "starts": starts[clock],
            "real": ~np.isnan(v[clock]), "days": len(org) / 288, "target_time": o["target_time"],
            "scores": {"Calendar (historical average)": forecast_score(cal, ff), "Trend rule (no model)": trend_score(v, org, ff),
                       "XGBoost": forecast_score(d["xgb"], ff), "DCRNN": forecast_score(d["dcrnn"], ff),
                       "Hybrid": forecast_score(d["f"], ff), "Hybrid, calibrated probability": probability_score(conf, d)}}


def bootstrap_days(m: dict, org_days: np.ndarray, jam_days: np.ndarray, rng, n: int = 2000, lead: int = 10) -> dict:
    """95% intervals for recall and precision, resampling whole days (jams within a day are not independent)."""
    uniq = np.unique(org_days)
    hit = np.array([m["hit"][org_days == d].sum() for d in uniq])
    al = np.array([m["scored"][org_days == d].sum() for d in uniq])
    got = np.array([(m["lead"][jam_days == d] >= lead).sum() for d in uniq])
    jam = np.array([(jam_days == d).sum() for d in uniq])
    idx = rng.integers(0, len(uniq), (n, len(uniq)))
    rec, pre = 100 * got[idx].sum(1) / jam[idx].sum(1), 100 * hit[idx].sum(1) / np.maximum(al[idx].sum(1), 1)
    return {"recall_lo": float(np.percentile(rec, 2.5)), "recall_hi": float(np.percentile(rec, 97.5)),
            "precision_lo": float(np.percentile(pre, 2.5)), "precision_hi": float(np.percentile(pre, 97.5))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    seed = ap.parse_args().seed
    OUT.mkdir(parents=True, exist_ok=True)
    data, ff, speed = unc.load_val_test(seed), unc.free_flow(), load_speed()
    v = np.where(speed.values > 0, speed.values, np.nan).astype(np.float32)
    starts, conf = jam_starts(v, ff), unc.calibrate(data["val"])
    val, test = (prepare(s, data, v, starts, ff, conf) for s in ("val", "test"))
    org_days = speed.index[test["origins"]].normalize().to_numpy()
    rng = np.random.default_rng(0)

    points, curve, daily, leads, period = [], [], [], [], []
    labels = period_labels(test["target_time"], load_config()["peaks"], 1)[:, 0]           # period of the next 5 minutes
    for name in test["scores"]:
        sv, st = val["scores"][name], test["scores"][name]
        for budget in BUDGETS:
            thr = threshold_for_budget(sv, val["eligible"], val["starts"], val["real"], val["days"], budget)
            mv = metrics(match(raise_alerts(sv, val["eligible"], thr), val["starts"], val["real"]), val["days"])
            m = match(raise_alerts(st, test["eligible"], thr), test["starts"], test["real"])
            jam_days = org_days[m["jam_t"]]
            points.append({"model": name, "budget_per_day": budget, "threshold": thr, **metrics(m, test["days"]),
                           **bootstrap_days(m, org_days, jam_days, rng),
                           "val_recall_10min_pct": mv["recall_10min_pct"], "val_false_alarms_per_day": mv["false_alarms_per_day"]})
            if budget == HEADLINE_BUDGET:
                for d in np.unique(org_days):
                    sel = jam_days == d
                    daily.append({"model": name, "day": str(pd.Timestamp(d).date()), "jams": int(sel.sum()),
                                  "warned_10min": int((m["lead"][sel] >= 10).sum()), "alerts": int(m["scored"][org_days == d].sum()),
                                  "false_alarms": int(m["false"][org_days == d].sum())})
                leads += [{"model": name, "lead_min": L, "jams": int((m["lead"] == L).sum())} for L in range(0, 61, 5)]
                for p in ("peak", "offpeak", "weekend"):
                    sel, jl = labels == p, labels[m["jam_t"]] == p
                    period.append({"model": name, "period": p, "jams": int(jl.sum()),
                                   "recall_10min_pct": 100 * float((m["lead"][jl] >= 10).mean()),
                                   "precision_pct": 100 * float(m["hit"][sel].sum() / max(m["scored"][sel].sum(), 1))})
        pool = st[test["eligible"] & np.isfinite(st)]
        for q in np.unique(1 - np.geomspace(0.25, 2e-4, 40)):                              # test trade-off curve (description only)
            thr = float(np.quantile(pool, q))
            curve.append({"model": name, "threshold": thr,
                          **metrics(match(raise_alerts(st, test["eligible"], thr), test["starts"], test["real"]), test["days"])})
        print(pd.DataFrame(points[-len(BUDGETS):]).drop(columns=["threshold"]).round(1).to_string(index=False), flush=True)
    for nm, rows in (("operating_points", points), ("curve", curve), ("daily", daily), ("lead_times", leads), ("by_period", period)):
        pd.DataFrame(rows).to_csv(OUT / f"{nm}.csv", index=False, float_format="%.5g")
    main = main_model(pd.DataFrame(points))
    (OUT / "summary.json").write_text(json.dumps({
        "seed": seed, "test_days": test["days"], "val_days": val["days"], "test_jams": int(points[0]["jams"]),
        "val_jams": int(val["starts"][HORIZON:len(val["origins"])].sum()), "sensors": N, "budgets": BUDGETS, 
        "main_model": main, "headline_budget": HEADLINE_BUDGET,
        "jam": "3 readings below 50% of free flow after 30 clear minutes", "quiet_min": 5 * QUIET}, indent=2))


if __name__ == "__main__":
    main()
