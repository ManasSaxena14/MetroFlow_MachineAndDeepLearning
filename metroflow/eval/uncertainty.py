"""Calibrated forecast ranges and congestion probabilities for the hybrid (split-conformal, by group).

    python -m metroflow.eval.uncertainty [--seed 0]

Calibration uses out-of-fold validation errors (metroflow/train/hybrid_cv.py); the test block is scored once.

A forecast's error depends on its horizon, on the traffic state it predicts and on how far the two models behind the
hybrid disagree. Residuals (truth minus forecast) are therefore pooled by (horizon, predicted state, size of the
XGBoost / DCRNN disagreement). The pooled quantiles give an asymmetric range, and the pooled distribution shifted by the
forecast gives the probability that the sensor will be congested. One constant range per horizon is the reference.

What conformal calibration promises: the stated coverage on average over forecasts like the calibration ones. It does
NOT promise that coverage inside a slice chosen by the outcome, such as "the 30 minutes after a jam starts". That slice
is reported on purpose.

Writes results/uncertainty/{coverage,by_slice,by_day,reliability}.csv and summary.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from metroflow.analysis import gap
from metroflow.data.dataset import load_split
from metroflow.data.load import load_speed
from metroflow.data.preprocess import load_config, split_bounds
from metroflow.eval.evaluate import REPORT_HORIZONS, period_labels
from metroflow.features.build import load_features
from metroflow.train.baselines import PRED_DIR, ROOT

OUT = ROOT / "results" / "uncertainty"
CONGESTED, SLOW = 0.5, 0.8                   # share of a sensor's own free-flow speed; same levels as the congestion maps
MIN_GROUP = 500
GRID = np.concatenate([[0.0], (np.arange(1, 400)) / 400, [1.0]])      # stored quantiles of each group's residuals


def free_flow() -> np.ndarray:
    """(N,) 85th percentile of each sensor's real TRAIN speeds."""
    speed = load_speed()
    v = speed.values[: split_bounds(len(speed), load_config()["split"])["train"][1]].astype(float)
    return np.nanpercentile(np.where(v > 0, v, np.nan), 85, axis=0)


def state(forecast: np.ndarray, ff: np.ndarray) -> np.ndarray:
    """0 congested, 1 slow, 2 free, from a forecast and the matching free-flow speed."""
    r = forecast / ff
    return np.where(r < CONGESTED, 0, np.where(r < SLOW, 1, 2)).astype(np.int8)


def conformal_level(n: int, q: float, upper: bool) -> float:
    """Finite-sample corrected quantile level of n calibration residuals."""
    return min(1.0, np.ceil((n + 1) * q) / n) if upper else max(0.0, np.floor((n + 1) * q) / n)


class GroupedConformal:
    """Residual distributions per (horizon, predicted state, disagreement third)."""

    def fit(self, f: np.ndarray, gap_abs: np.ndarray, y: np.ndarray, m: np.ndarray, ff: np.ndarray) -> GroupedConformal:
        """f, gap_abs, y, m: (rows, H); ff: (rows,). gap_abs = |XGBoost - DCRNN|."""
        H = f.shape[1]
        self.edges = np.stack([np.quantile(gap_abs[:, h][m[:, h]], [1 / 3, 2 / 3]) for h in range(H)])
        self.table = np.empty((H, 3, 3, len(GRID)), np.float32)
        self.n = np.zeros((H, 3, 3), int)
        for h in range(H):
            ok = m[:, h]
            r, s = (y[ok, h] - f[ok, h]), state(f[ok, h], ff[ok])
            d = np.searchsorted(self.edges[h], gap_abs[ok, h])
            for i in range(3):
                for j in range(3):
                    sel = (s == i) & (d == j)
                    self.n[h, i, j] = sel.sum()
                    if sel.sum() < MIN_GROUP:                          # too few: pool the state, then the horizon
                        sel = (s == i) if (s == i).sum() >= MIN_GROUP else np.ones_like(s, bool)
                    self.table[h, i, j] = np.quantile(r[sel], GRID)
        return self

    def _idx(self, f, gap_abs, ff, h):
        return state(f, ff), np.searchsorted(self.edges[h], gap_abs)

    def interval(self, f: np.ndarray, gap_abs: np.ndarray, ff: np.ndarray, h: int, cover: float) -> tuple[np.ndarray, np.ndarray]:
        """Lower and upper speed (mph) for one horizon index h (0-based)."""
        s, d = self._idx(f, gap_abs, ff, h)
        lo_t, hi_t = np.empty((3, 3)), np.empty((3, 3))
        for i in range(3):
            for j in range(3):
                n = max(int(self.n[h, i, j]), MIN_GROUP)
                lo_t[i, j] = np.interp(conformal_level(n, (1 - cover) / 2, False), GRID, self.table[h, i, j])
                hi_t[i, j] = np.interp(conformal_level(n, (1 + cover) / 2, True), GRID, self.table[h, i, j])
        return np.maximum(f + lo_t[s, d], 0.0), f + hi_t[s, d]

    def prob_below(self, f: np.ndarray, gap_abs: np.ndarray, ff: np.ndarray, h: int, frac: float = CONGESTED) -> np.ndarray:
        """P(speed < frac * free flow) at horizon index h."""
        s, d = self._idx(f, gap_abs, ff, h)
        out = np.empty(len(f), np.float32)
        need = frac * ff - f                                           # residual below which the sensor is congested
        for i in range(3):
            for j in range(3):
                sel = (s == i) & (d == j)
                out[sel] = np.interp(need[sel], self.table[h, i, j], GRID)
        return out


class ConstantConformal:
    """Reference: one symmetric half-width per horizon."""

    def fit(self, f, y, m) -> ConstantConformal:
        self.abs = [np.sort(np.abs(y[m[:, h], h] - f[m[:, h], h])) for h in range(f.shape[1])]
        return self

    def interval(self, f, h, cover):
        a = self.abs[h]
        w = a[min(len(a) - 1, int(np.ceil((len(a) + 1) * cover)) - 1)]
        return np.maximum(f - w, 0.0), f + w


def interval_score(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, cover: float) -> np.ndarray:
    """Winkler score (mph, lower is better): width plus a penalty for every miss."""
    a = 1 - cover
    return (hi - lo) + (2 / a) * np.maximum(lo - y, 0) + (2 / a) * np.maximum(y - hi, 0)


def describe(y, lo, hi, cover) -> dict:
    return {"coverage_pct": 100 * float(((y >= lo) & (y <= hi)).mean()), "mean_width_mph": float((hi - lo).mean()),
            "median_width_mph": float(np.median(hi - lo)), "interval_score": float(interval_score(y, lo, hi, cover).mean()),
            "n": int(len(y))}


def load_val_test(seed: int = 0) -> dict:
    """Everything the calibration, alert and replay steps share: hybrid forecasts and the disagreement, val and test."""
    out = {}
    for split, hyb in (("val", f"hybrid_oof_s{seed}_val"), ("test", f"hybrid_resid_s{seed}_test")):
        _, Y, M = load_features(split)
        x, d = (np.load(PRED_DIR / f"{n}_{split}.npy") for n in ("xgb", f"dcrnn_s{seed}"))
        out[split] = {"f": np.load(PRED_DIR / f"{hyb}.npy"), "xgb": x, "dcrnn": d, "gap": np.abs(x - d),
                      "Y": np.asarray(Y), "M": np.asarray(M)}
    ff = free_flow()
    for d in out.values():
        d["ff"] = np.tile(ff, len(d["f"]) // len(ff))
    return out


def calibrate(val: dict) -> GroupedConformal:
    return GroupedConformal().fit(val["f"], val["gap"], val["Y"], val["M"], val["ff"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    seed = ap.parse_args().seed
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_val_test(seed)
    val, test = data["val"], data["test"]
    grouped, const = calibrate(val), ConstantConformal().fit(val["f"], val["Y"], val["M"])
    meta = load_split("test", keys=["target_time", "origin_idx"])
    periods = period_labels(meta["target_time"], load_config()["peaks"])
    onset = gap.onset_labels(meta["origin_idx"], gap.post_onset_grid(load_speed()))
    days = np.repeat(pd.DatetimeIndex(meta["target_time"].ravel()).normalize().to_numpy()
                     .reshape(meta["target_time"].shape)[:, None, :], 207, axis=1).reshape(-1, 12)

    cov, slices, daily, rel = [], [], [], []
    for step, mins in REPORT_HORIZONS.items():
        h = step - 1
        ok = test["M"][:, h]
        y, f, g, ff = test["Y"][ok, h], test["f"][ok, h], test["gap"][ok, h], test["ff"][ok]
        actual = state(y, ff)
        masks = {"all": np.ones(len(y), bool), "weekday peak": periods[ok, h] == "peak",
                 "weekday off-peak": periods[ok, h] == "offpeak", "weekend": periods[ok, h] == "weekend",
                 "truly free": actual == 2, "truly slow": actual == 1,
                 "truly congested": actual == 0, "30 min after a jam onset": onset[ok, h]}
        for cover in (0.8, 0.9):
            for method, (lo, hi) in (("adaptive", grouped.interval(f, g, ff, h, cover)), ("constant", const.interval(f, h, cover))):
                cov.append({"method": method, "minutes": mins, "target_pct": 100 * cover, **describe(y, lo, hi, cover)})
                for name, sel in masks.items():
                    slices.append({"method": method, "minutes": mins, "target_pct": 100 * cover, "slice": name,
                                   **describe(y[sel], lo[sel], hi[sel], cover)})
                if cover == 0.8:
                    inside = pd.Series((y >= lo) & (y <= hi)).groupby(days[ok, h]).mean()
                    daily += [{"method": method, "minutes": mins, "day": str(pd.Timestamp(k).date()), "coverage_pct": 100 * float(v)}
                              for k, v in inside.items()]
        p, hit = grouped.prob_below(f, g, ff, h), y < CONGESTED * ff
        base = float((val["Y"][val["M"][:, h], h] < CONGESTED * val["ff"][val["M"][:, h]]).mean())
        bins = np.clip((p * 10).astype(int), 0, 9)
        for b in range(10):
            sel = bins == b
            if sel.sum():
                rel.append({"minutes": mins, "bin": b, "mean_probability": float(p[sel].mean()), "observed": float(hit[sel].mean()),
                            "n": int(sel.sum())})
        cov[-4]["brier"] = float(((p - hit) ** 2).mean())
        cov[-4]["brier_point"] = float((((f < CONGESTED * ff).astype(float) - hit) ** 2).mean())
        cov[-4]["brier_base_rate"] = float(((base - hit) ** 2).mean())
    for name, rows in (("coverage", cov), ("by_slice", slices), ("by_day", daily), ("reliability", rel)):
        pd.DataFrame(rows).to_csv(OUT / f"{name}.csv", index=False, float_format="%.5g")
    c = pd.DataFrame(cov)
    head = c[(c.method == "adaptive") & (c.target_pct == 80)]
    summary = {"seed": seed, "calibration_targets": int(val["M"].sum()), "calibration_days": 12, "folds": 4,
               "groups": "horizon x predicted state x disagreement third",
               "coverage_80_adaptive": dict(zip(head.minutes.tolist(), head.coverage_pct.round(2).tolist()))}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(c.round(2).to_string(index=False))
    s = pd.DataFrame(slices).query("target_pct == 80")
    print(s.pivot_table(index="slice", columns=["minutes", "method"], values="coverage_pct").round(1).to_string())


if __name__ == "__main__":
    main()
