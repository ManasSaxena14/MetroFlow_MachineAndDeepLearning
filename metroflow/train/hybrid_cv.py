"""Refit the compact hybrid corrector and, for calibration, get its errors on data each corrector never saw.

    python -m metroflow.train.hybrid_cv [--seed 0]

Needs the DCRNN and XGBoost predictions on val and test. Loads only XGBoost (never PyTorch).

The corrector is fitted on the validation block, so its own validation errors are too small to calibrate intervals or
alert thresholds on. The published recipe is kept for the point forecast (first 75% of validation fits, last 25% stops
early). For calibration only, whole validation days are split into four interleaved folds; each fold is forecast by a
corrector fitted on the other folds, with a gap of 24 windows on both sides of every held-out day, because windows less
than 24 steps apart share readings. The test block is not touched by either step.

Writes (data/processed/strict/preds/): hybrid_resid_s<k>_test.npy, hybrid_oof_s<k>_val.npy
       (checkpoints/hybrid/s<k>/)    : h01.json ... h12.json, the correctors behind the point forecast
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import xgboost as xgb

from metroflow.data.dataset import load_split
from metroflow.eval.evaluate import REPORT_HORIZONS, evaluate_rows
from metroflow.features.build import load_features
from metroflow.models import hybrid
from metroflow.train.baselines import PRED_DIR, ROOT

N_FOLDS = 4
GAP = 24          # windows: 12 history + 12 horizon steps


def day_folds(target_time: np.ndarray, n_folds: int = N_FOLDS) -> np.ndarray:
    """Fold of every window: the calendar day of its first target, interleaved over the folds."""
    day = pd.DatetimeIndex(target_time[:, 0]).normalize()
    return ((day - day[0]).days.to_numpy() % n_folds).astype(int)


def fit_mask(folds: np.ndarray, held_out: int, gap: int = GAP) -> np.ndarray:
    """Windows that may fit the corrector for one fold: other folds, and not within `gap` windows of a held-out one."""
    out = folds == held_out
    near = out.copy()
    for k in range(1, gap + 1):
        near[k:] |= out[:-k]
        near[:-k] |= out[k:]
    return ~near


def oof_forecast(F, dl, ml, Y, M, folds: np.ndarray, rounds: dict[int, int], log=print) -> np.ndarray:
    """Out-of-fold hybrid forecast of the validation rows. `rounds` = boosting rounds per horizon of the point model."""
    out = np.array(dl, dtype=np.float32, copy=True)
    for h, n_rounds in rounds.items():
        t0, k = time.time(), h - 1
        Xh = hybrid.stack_features(F, dl, ml, h, compact=True)
        resid, real = np.asarray(Y[:, k]) - dl[:, k], np.asarray(M[:, k])
        for f in np.unique(folds):
            fit = np.repeat(fit_mask(folds, f), hybrid.N_SENSORS) & real
            rows = np.repeat(folds == f, hybrid.N_SENSORS)
            bst = xgb.train(hybrid.COMPACT_PARAMS, xgb.DMatrix(Xh[fit], label=resid[fit]), n_rounds)
            out[rows, k] += bst.predict(xgb.DMatrix(Xh[rows]))
        log(f"  oof h={h:2d}: {n_rounds} rounds, {time.time() - t0:.0f}s")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    s = ap.parse_args().seed
    Fv, Yv, Mv = (np.asarray(a) for a in load_features("val"))
    Ft, Yt, Mt = (np.asarray(a) for a in load_features("test"))
    dv, dt, xv, xt = (np.load(PRED_DIR / f"{n}.npy").astype(np.float32)
                      for n in (f"dcrnn_s{s}_val", f"dcrnn_s{s}_test", "xgb_val", "xgb_test"))
    fit_rows, stop_rows = hybrid.chrono_split(len(Fv))
    models, best = hybrid.fit_corrector(Fv, dv, xv, Yv, Mv, fit_rows, stop_rows, compact=True)
    folder = ROOT / "checkpoints" / "hybrid" / f"s{s}"
    folder.mkdir(parents=True, exist_ok=True)
    for h, b in models.items():
        b[: best[h] + 1].save_model(folder / f"h{h:02d}.json")
    P = hybrid.apply_corrector(models, Ft, dt, xt, compact=True)
    np.save(PRED_DIR / f"hybrid_resid_s{s}_test.npy", P)
    for name, A in (("dcrnn", dt), ("hybrid", P)):
        e = evaluate_rows(Yt, A, Mt).set_index("horizon").mae
        print(f"{name} test MAE 15/30/60: " + " / ".join(f"{e[str(h)]:.3f}" for h in REPORT_HORIZONS), flush=True)
    folds = day_folds(load_split("val", keys=["target_time"])["target_time"])
    oof = oof_forecast(Fv, dv, xv, Yv, Mv, folds, {h: b + 1 for h, b in best.items()}, log=lambda t: print(t, flush=True))
    np.save(PRED_DIR / f"hybrid_oof_s{s}_val.npy", oof)
    e = evaluate_rows(Yv, oof, Mv).set_index("horizon").mae
    print("hybrid out-of-fold val MAE 15/30/60: " + " / ".join(f"{e[str(h)]:.3f}" for h in REPORT_HORIZONS), flush=True)


if __name__ == "__main__":
    main()
