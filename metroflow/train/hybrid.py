"""Hybrid runner: DCRNN + XGBoost residual corrector, and the DCRNN/XGBoost blend, one per DCRNN seed.

    python -m metroflow.train.hybrid            # -> data/processed/strict/preds/hybrid_{resid,residfull,blend}_s<k>_test.npy,
                                          #    results/hybrid/*.csv, results/hybrid/summary.json

Loads only XGBoost (never PyTorch): the graph-model forecasts are read from the prediction files the graph runner saved.
See metroflow/models/hybrid.py for the design and the leakage rule (fit on the validation block, score on test).
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from metroflow.eval.evaluate import REPORT_HORIZONS, evaluate_rows
from metroflow.features.build import load_features
from metroflow.models import hybrid
from metroflow.train.baselines import PRED_DIR, ROOT

OUT = ROOT / "results" / "hybrid"
SEEDS = (0, 1, 2)


def load(name: str, split: str) -> np.ndarray:
    return np.asarray(np.load(PRED_DIR / f"{name}_{split}.npy"), dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS), help="DCRNN seeds to build hybrids on")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    Fv, Yv, Mv = load_features("val")
    Ft, Yt, Mt = load_features("test")
    Fv, Ft = np.asarray(Fv), np.asarray(Ft)
    xv, xt = load("xgb", "val"), load("xgb", "test")
    fit_rows, stop_rows = hybrid.chrono_split(len(Fv))
    summary = {"fit_windows": int(fit_rows.sum() // hybrid.N_SENSORS), "stop_windows": int(stop_rows.sum() // hybrid.N_SENSORS),
               "features": hybrid.feature_names(True), "params": hybrid.COMPACT_PARAMS, "runs": {}}
    summary_path = OUT / "summary.json"
    if summary_path.exists():                                  # keep the seeds built by earlier runs
        summary["runs"] = json.loads(summary_path.read_text()).get("runs", {})
    for s in args.seeds:
        if not (PRED_DIR / f"dcrnn_s{s}_test.npy").exists():
            print(f"dcrnn_s{s}: no predictions yet, skipped")
            continue
        t0 = time.time()
        dv, dt = load(f"dcrnn_s{s}", "val"), load(f"dcrnn_s{s}", "test")
        w = hybrid.fit_blend(dv, xv, Yv, Mv, fit_rows | stop_rows)
        blend = hybrid.apply_blend(dt, xt, w)
        models, best = hybrid.fit_corrector(Fv, dv, xv, Yv, Mv, fit_rows, stop_rows, compact=True)
        resid = hybrid.apply_corrector(models, Ft, dt, xt, compact=True)
        full_models, _ = hybrid.fit_corrector(Fv, dv, xv, Yv, Mv, fit_rows, stop_rows, compact=False)
        full = hybrid.apply_corrector(full_models, Ft, dt, xt, compact=False)
        gains = {}
        for name, P in ((f"hybrid_blend_s{s}", blend), (f"hybrid_resid_s{s}", resid), (f"hybrid_residfull_s{s}", full)):
            np.save(PRED_DIR / f"{name}_test.npy", P)
            ev = evaluate_rows(Yt, P, Mt)
            ev.insert(0, "split", "test")
            ev.to_csv(OUT / f"{name}.csv", index=False)
            gains[name] = {m: float(ev.set_index("horizon").loc[str(h), "mae"]) for h, m in REPORT_HORIZONS.items()}
        imp = pd.DataFrame({h: pd.Series(b.get_score(importance_type="gain")) for h, b in models.items()}).fillna(0)
        imp.index = [summary["features"][int(i[1:])] for i in imp.index]          # f0, f1, ... -> names
        (imp / imp.sum()).to_csv(OUT / f"corrector_importance_s{s}.csv")
        summary["runs"][f"s{s}"] = {"blend_weight_on_dcrnn": [round(float(v), 3) for v in w], "corrector_rounds": best,
                                    "test_mae": gains, "seconds": round(time.time() - t0)}
        print(f"dcrnn_s{s}: blend w(15/30/60) = {w[2]:.2f}/{w[5]:.2f}/{w[11]:.2f}; test MAE "
              + "; ".join(f"{k}: " + "/".join(f"{v:.3f}" for v in g.values()) for k, g in gains.items())
              + f"  ({time.time() - t0:.0f}s)", flush=True)
    summary["runs"] = dict(sorted(summary["runs"].items()))
    summary_path.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
