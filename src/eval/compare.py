"""Phase 6: the fair comparison. Every model is scored by the same functions, on the same test windows, with the same
mask, in mph, from the predictions each runner saved (data/processed/strict/preds/<name>_test.npy, row layout).

    python -m src.eval.compare            # -> results/final*.csv, results/significance.csv, results/gain_*.csv,
                                          #    results/phase6_findings.json, figures 15-17

Fairness contract: same chronological test block (Phase 2), same 12-step history, same masking, every model tuned on
the validation block only, errors in mph. Neural graph models are reported as the mean (and std) of 3 seeds.

Significance: errors inside one day are strongly correlated, so the unit of resampling is the DAY. For each horizon we
compare the best independent model with each graph model by (1) a paired bootstrap over test days (10,000 resamples)
and (2) a Diebold-Mariano test on the daily loss differential with the Harvey-Leybourne-Newbold small-sample correction.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from src.analysis import gap
from src.data.dataset import load_adjacency, load_split
from src.data.load import load_speed
from src.data.preprocess import load_config
from src.eval.evaluate import REPORT_HORIZONS, evaluate_rows, period_labels
from src.features.build import load_features
from src.train.baselines import PRED_DIR, ROOT

OUT = ROOT / "results"
HORIZONS = sorted(REPORT_HORIZONS)                       # 3, 6, 12 -> 15, 30, 60 min
SEEDS = (0, 1, 2)
# display name -> list of prediction files (several = seeds, reported as mean +- std)
MODELS = {
    "Persistence": ["persistence"],
    "Historical average": ["hist_avg"],
    "Ridge per sensor": ["ridge"],
    "Multi-output Random Forest": ["rf"],
    "LSTM (no graph)": ["lstm"],
    "XGBoost per horizon": ["xgb"],
    "XGBoost + neighbours": ["xgb_both"],
    "STGAT (ours)": [f"stgat_s{s}" for s in SEEDS],
    "DCRNN": [f"dcrnn_s{s}" for s in SEEDS],
    "Hybrid: DCRNN + XGBoost (ours)": [f"hybrid_resid_s{s}" for s in SEEDS],
}
GRAPH_MODELS = ("STGAT (ours)", "DCRNN")
HYBRID = "Hybrid: DCRNN + XGBoost (ours)"
HYBRID_VARIANTS = {  # ablation of the hybrid itself: each row is mean over the DCRNN seeds it was built on
    "DCRNN alone": "dcrnn_s{s}",
    "Blend: w_h * DCRNN + (1 - w_h) * XGBoost": "hybrid_blend_s{s}",
    "Residual corrector, all 46 features (deep trees)": "hybrid_residfull_s{s}",
    "Residual corrector, compact + regularised (main)": "hybrid_resid_s{s}",
}
BEST_INDEPENDENT = "XGBoost per horizon"                 # strongest model that never sees a neighbour (Phase 3)


def available() -> dict[str, list[str]]:
    out = {}
    for label, files in MODELS.items():
        have = [f for f in files if (PRED_DIR / f"{f}_test.npy").exists()]
        if have:
            out[label] = have
    return out


def load_pred(name: str) -> np.ndarray:
    return np.load(PRED_DIR / f"{name}_test.npy", mmap_mode="r")


# --------------------------------------------------------------------------- tables
def headline(Y, M, labels, models) -> tuple[pd.DataFrame, pd.DataFrame]:
    """final.csv (one row per model, seed mean and std) and final_by_horizon.csv (all 12 horizons, every run)."""
    rows, long = [], []
    for label, files in models.items():
        per_run = []
        for f in files:
            P = load_pred(f)
            ev = evaluate_rows(Y, P, M).set_index("horizon")
            r = {}
            for h, mins in REPORT_HORIZONS.items():
                for k in ("mae", "rmse", "mape"):
                    r[f"{k.upper()}_{mins}min"] = ev.loc[str(h), k]
            for per in ("peak", "offpeak", "weekend"):
                mm = np.asarray(M[:, 11]) & (labels[:, 11] == per)
                r[f"{per}_MAE_60min"] = float(np.abs(np.asarray(P[:, 11]) - np.nan_to_num(np.asarray(Y[:, 11])))[mm].mean())
            per_run.append(r)
            e = ev.drop(index="avg").reset_index()
            e.insert(0, "run", f)
            e.insert(0, "model", label)
            long.append(e)
        df = pd.DataFrame(per_run)
        row = {"model": label, "runs": len(files), **df.mean().round(3).to_dict()}
        if len(files) > 1:
            row.update({f"{c}_std": round(float(df[c].std()), 3) for c in ("MAE_15min", "MAE_30min", "MAE_60min")})
        rows.append(row)
    return pd.DataFrame(rows), pd.concat(long, ignore_index=True)


def daily_errors(P, Y, M, days: np.ndarray, h: int) -> tuple[np.ndarray, np.ndarray]:
    """Per test day: sum of absolute errors and number of real targets at horizon h."""
    y, p, m = np.asarray(Y[:, h - 1]), np.asarray(P[:, h - 1]), np.asarray(M[:, h - 1])
    err = np.where(m, np.abs(p - np.nan_to_num(y)), 0.0)
    d = days[:, h - 1]
    uniq = np.unique(d)
    pos = np.searchsorted(uniq, d)
    return np.bincount(pos, err, len(uniq)), np.bincount(pos, m.astype(float), len(uniq))


def dm_test(diff: np.ndarray, lag: int = 1) -> tuple[float, float]:
    """Diebold-Mariano statistic on a loss-differential series with Newey-West variance and the HLN correction.

    Returns (statistic, two-sided p-value from Student t with T-1 degrees of freedom)."""
    from scipy import stats

    t = len(diff)
    d = diff - diff.mean()
    gamma = [d @ d / t] + [d[k:] @ d[:-k] / t for k in range(1, lag + 1)]
    var = gamma[0] + 2 * sum((1 - k / (lag + 1)) * gamma[k] for k in range(1, lag + 1))
    if var <= 0:
        return 0.0, 1.0                                     # identical forecasts: no evidence either way
    stat = diff.mean() / math.sqrt(var / t)
    h = lag + 1
    stat *= math.sqrt((t + 1 - 2 * h + h * (h - 1) / t) / t)
    return float(stat), float(2 * stats.t.sf(abs(stat), df=t - 1))


def compare_pair(P, B, Y, M, days, rng, n_boot: int) -> list[dict]:
    """Paired day-level tests of forecast P against baseline B at the headline horizons."""
    rows = []
    for h in HORIZONS:
        eb, n = daily_errors(B, Y, M, days, h)
        eg, _ = daily_errors(P, Y, M, days, h)
        gain = (eb.sum() - eg.sum()) / n.sum()                           # mph, > 0 = model better than baseline
        idx = rng.integers(0, len(n), (n_boot, len(n)))
        boot = (eb[idx].sum(1) - eg[idx].sum(1)) / n[idx].sum(1)
        p_boot = float(min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean())))
        stat, p_dm = dm_test(eb / n - eg / n)
        rows.append({"minutes": 5 * h, "baseline_mae": eb.sum() / n.sum(), "model_mae": eg.sum() / n.sum(),
                     "gain_mph": gain, "gain_pct": 100 * gain / (eb.sum() / n.sum()),
                     "ci95_low": float(np.percentile(boot, 2.5)), "ci95_high": float(np.percentile(boot, 97.5)),
                     "p_bootstrap": p_boot, "dm_stat": stat, "p_dm": p_dm, "test_days": int(len(n)),
                     "days_model_better": int((eg / n < eb / n).sum())})
    return rows


def significance(Y, M, days, models, n_boot: int = 10_000, seed: int = 0) -> pd.DataFrame:
    """Every graph model / hybrid seed against XGBoost, and the hybrid against the DCRNN seed it was built on."""
    rng = np.random.default_rng(seed)
    base = load_pred(models[BEST_INDEPENDENT][0])
    pairs = []
    for label in [g for g in (*GRAPH_MODELS, HYBRID, "XGBoost + neighbours") if g in models]:
        pairs += [(label, f, base, BEST_INDEPENDENT) for f in models[label]]
    if HYBRID in models:
        for f in models[HYBRID]:
            dl = f.replace("hybrid_resid", "dcrnn")
            if (PRED_DIR / f"{dl}_test.npy").exists():
                pairs.append((HYBRID, f, load_pred(dl), "DCRNN (same seed)"))
    rows = []
    for label, f, B, bname in pairs:
        for r in compare_pair(load_pred(f), B, Y, M, days, rng, n_boot):
            rows.append({"model": label, "run": f, "baseline": bname, **r})
    return pd.DataFrame(rows)


def hybrid_ablation(Y, M) -> pd.DataFrame:
    """results/hybrid_ablation.csv: test MAE of the hybrid's variants (mean and std over DCRNN seeds)."""
    rows = []
    for label, pattern in HYBRID_VARIANTS.items():
        runs = [pattern.format(s=s) for s in SEEDS if (PRED_DIR / f"{pattern.format(s=s)}_test.npy").exists()]
        if not runs:
            continue
        vals = pd.DataFrame([evaluate_rows(Y, load_pred(f), M).set_index("horizon").mae for f in runs])
        r = {"variant": label, "runs": len(runs)}
        for h, mins in REPORT_HORIZONS.items():
            r[f"MAE_{mins}min"] = round(float(vals[str(h)].mean()), 3)
            r[f"MAE_{mins}min_std"] = round(float(vals[str(h)].std()), 3) if len(runs) > 1 else 0.0
        rows.append(r)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- where does the gain come from?
def gain_by_degree(Y, M, models, adj: np.ndarray, h: int = 12) -> pd.DataFrame:
    """Per-sensor MAE at horizon h, grouped by how many road links a sensor has (in + out, self excluded)."""
    off = (adj > 0) & ~np.eye(len(adj), dtype=bool)
    deg = off.sum(0) + off.sum(1)
    n = len(adj)
    bins = [(-1, 4, "0-4"), (4, 8, "5-8"), (8, 12, "9-12"), (12, 99, "13+")]
    y = np.asarray(Y[:, h - 1]).reshape(-1, n)
    m = np.asarray(M[:, h - 1]).reshape(-1, n)
    rows = []
    for label, files in models.items():
        err = np.mean([np.where(m, np.abs(np.asarray(load_pred(f)[:, h - 1]).reshape(-1, n) - np.nan_to_num(y)), 0).sum(0)
                       for f in files], axis=0) / m.sum(0)
        for lo, hi, name in bins:
            sel = (deg > lo) & (deg <= hi)
            rows.append({"model": label, "degree": name, "sensors": int(sel.sum()), "mae_60min": float(err[sel].mean())})
    return pd.DataFrame(rows)


def gain_at_onset(Y, M, models, origins: np.ndarray) -> pd.DataFrame:
    """MAE in the 30 minutes after a congestion onset vs all other targets (Phase 4 labels), per model and horizon."""
    labels = gap.onset_labels(origins, gap.post_onset_grid(load_speed()))
    rows = []
    for label, files in models.items():
        for h in HORIZONS:
            for grp, sel in (("onset", labels[:, h - 1]), ("normal", ~labels[:, h - 1])):
                mm = np.asarray(M[:, h - 1]) & sel
                vals = [float(np.abs(np.asarray(load_pred(f)[:, h - 1]) - np.nan_to_num(np.asarray(Y[:, h - 1])))[mm].mean())
                        for f in files]
                rows.append({"model": label, "minutes": 5 * h, "group": grp, "mae": float(np.mean(vals)), "n": int(mm.sum())})
    return pd.DataFrame(rows)


def main() -> None:
    models = available()
    print("models with test predictions:", ", ".join(f"{k} ({len(v)})" for k, v in models.items()))
    _, Y, M = load_features("test")
    test = load_split("test", keys=["target_time", "origin_idx"])
    n = load_adjacency()["adj"].shape[0]
    labels = period_labels(test["target_time"], load_config()["peaks"], n)
    days = np.repeat(pd.DatetimeIndex(test["target_time"].ravel()).normalize().asi8.reshape(test["target_time"].shape)[:, None, :],
                     n, axis=1).reshape(-1, test["target_time"].shape[1])

    final, long = headline(Y, M, labels, models)
    final.to_csv(OUT / "final.csv", index=False)
    long.to_csv(OUT / "final_by_horizon.csv", index=False, float_format="%.4f")
    cols = ["model", "runs", "MAE_15min", "MAE_30min", "MAE_60min", "peak_MAE_60min", "offpeak_MAE_60min"]
    print(final[cols].to_string(index=False))

    sig = significance(Y, M, days, models)
    hyb = hybrid_ablation(Y, M)
    hyb.to_csv(OUT / "hybrid_ablation.csv", index=False)
    if len(hyb):
        print(hyb.to_string(index=False))
    sig.to_csv(OUT / "significance.csv", index=False, float_format="%.5g")
    deg = gain_by_degree(Y, M, models, load_adjacency()["adj"])
    deg.to_csv(OUT / "gain_by_degree.csv", index=False, float_format="%.4f")
    ons = gain_at_onset(Y, M, models, test["origin_idx"])
    ons.to_csv(OUT / "gain_at_onset.csv", index=False, float_format="%.4f")

    fm = final.set_index("model")
    best_graph = min((g for g in GRAPH_MODELS if g in models), key=lambda g: fm.loc[g, "MAE_60min"], default=None)
    findings = {"best_independent": BEST_INDEPENDENT, "best_graph_model": best_graph,
                "test_days": int(sig.test_days.max()) if len(sig) else 0}
    if best_graph:
        fb = final.set_index("model")
        for mins in (15, 30, 60):
            b, g = fb.loc[BEST_INDEPENDENT, f"MAE_{mins}min"], fb.loc[best_graph, f"MAE_{mins}min"]
            findings[f"gain_{mins}min_pct"] = round(100 * (b - g) / b, 2)
        s = sig[(sig.model == best_graph) & (sig.baseline == BEST_INDEPENDENT)]
        findings["max_p_bootstrap"] = {int(m): float(s[s.minutes == m].p_bootstrap.max()) for m in (15, 30, 60)}
        findings["max_p_dm"] = {int(m): float(s[s.minutes == m].p_dm.max()) for m in (15, 30, 60)}
    if HYBRID in models:
        fh = final.set_index("model").loc[HYBRID]
        hs = sig[(sig.model == HYBRID) & (sig.baseline == "DCRNN (same seed)")]
        findings["hybrid"] = {f"MAE_{m}min": float(fh[f"MAE_{m}min"]) for m in (15, 30, 60)}
        findings["hybrid_vs_dcrnn"] = {int(m): {"gain_pct": float(g.gain_pct.mean()), "max_p_bootstrap": float(g.p_bootstrap.max()),
                                                "max_p_dm": float(g.p_dm.max()), "min_days_better": int(g.days_model_better.min())}
                                       for m, g in hs.groupby("minutes")}
    (OUT / "phase6_findings.json").write_text(json.dumps(findings, indent=2, default=float))
    print(json.dumps(findings, indent=2, default=float))

    from src.viz import compare_plots
    compare_plots.make_all(final, long, deg, ons, sig)


if __name__ == "__main__":
    main()
