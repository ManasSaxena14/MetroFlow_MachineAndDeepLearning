"""Hybrid ML + DL forecaster: a graph model (DCRNN) corrected by gradient-boosted trees (XGBoost).

Why a hybrid, from our own results:
* DCRNN is the best model overall: it reads the road graph at every recurrent step.
* XGBoost, with engineered features (time of day, recent change, sensor statistics), is better than DCRNN in the
  30 minutes after a congestion onset at 60 minutes ahead, and its errors are not the same as DCRNN's.
So the two are complementary. The hybrid keeps the graph model as the backbone and lets a tree model learn where the
backbone is systematically wrong:

    final[h] = dcrnn[h] + r_h(features),     r_h = XGBoost trained on (y[h] - dcrnn[h])

Main hybrid ("compact"): r_h sees a few interpretable inputs, the current speed, its recent change, the clock, the
weekend flag and the sensor's free-flow speed, plus both forecasts and their gap, and uses shallow, heavily
regularised trees (depth 3, >= 200 samples per leaf, L2 = 10). Ablations of the hybrid:
  * "full": the same corrector on all 46 table features with deeper trees; it overfits the short fitting block.
  * "blend": a per-horizon convex blend final[h] = w_h * dcrnn[h] + (1 - w_h) * xgb[h].

Leakage: the corrector must see forecasts made on data the graph model was NOT trained on, or it learns in-sample
residuals that are far too small. So both hybrids are fitted on the validation block (DCRNN only used it to pick its
best epoch), split chronologically: the first 75% of validation windows fit the corrector, the last 25% decide its
early stopping. The test block is used only once, for the final score.
"""
from __future__ import annotations

import numpy as np
import xgboost as xgb

from metroflow.features.build import FEATURE_NAMES

SPEED_T = FEATURE_NAMES.index("speed_t")
N_SENSORS = 207
PARAMS = {"max_depth": 6, "eta": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 20,
          "tree_method": "hist", "max_bin": 128, "objective": "reg:absoluteerror", "eval_metric": "mae", "seed": 42}
COMPACT_PARAMS = {**PARAMS, "max_depth": 3, "min_child_weight": 200, "eta": 0.03, "reg_lambda": 10}
COMPACT_COLS = ["speed_t", "delta_1", "delta_3", "sin_tod", "cos_tod", "is_weekend", "free_flow_p85"]


def chrono_split(n_rows: int, frac: float = 0.75) -> tuple[np.ndarray, np.ndarray]:
    """Row masks for the first `frac` of windows (fit) and the rest (early stopping). Rows are window-major."""
    windows = n_rows // N_SENSORS
    cut = int(windows * frac) * N_SENSORS
    idx = np.arange(n_rows)
    return idx < cut, idx >= cut


def stack_features(F: np.ndarray, dl: np.ndarray, ml: np.ndarray, h: int, compact: bool = True) -> np.ndarray:
    """Per-row inputs of the corrector for horizon h (1..12): base features + both forecasts and their gaps."""
    d, m, now = dl[:, h - 1], ml[:, h - 1], F[:, SPEED_T]
    base = F[:, [FEATURE_NAMES.index(c) for c in COMPACT_COLS]] if compact else F
    extra = np.stack([d, m, d - now, m - d], axis=1)
    return np.concatenate([np.asarray(base, dtype=np.float32), extra.astype(np.float32)], axis=1)


def feature_names(compact: bool = True) -> list[str]:
    return (COMPACT_COLS if compact else FEATURE_NAMES) + ["dl_forecast", "xgb_forecast", "dl_minus_now", "xgb_minus_dl"]


def fit_blend(dl, ml, Y, M, rows) -> np.ndarray:
    """Per-horizon weight w in [0, 1] on the DL forecast minimising MAE on the given rows."""
    grid = np.linspace(0, 1, 41)
    w = np.empty(Y.shape[1])
    for k in range(Y.shape[1]):
        m = np.asarray(M[rows, k])
        y, d, x = (np.asarray(a[rows, k])[m] for a in (Y, dl, ml))
        w[k] = grid[np.argmin([np.abs(g * d + (1 - g) * x - y).mean() for g in grid])]
    return w


def apply_blend(dl, ml, w) -> np.ndarray:
    return (w[None, :] * dl + (1 - w[None, :]) * ml).astype(np.float32)


def fit_corrector(F, dl, ml, Y, M, fit_rows, stop_rows, horizons=range(1, 13), rounds: int = 1500, compact: bool = True):
    """One XGBoost residual model per horizon. Returns {h: booster} and {h: best iteration}."""
    models, best = {}, {}
    params = COMPACT_PARAMS if compact else PARAMS
    for h in horizons:
        k = h - 1
        Xf = stack_features(F, dl, ml, h, compact)
        target = np.asarray(Y[:, k]) - dl[:, k]
        m = np.asarray(M[:, k])
        dfit = xgb.DMatrix(Xf[fit_rows & m], label=target[fit_rows & m])
        dstop = xgb.DMatrix(Xf[stop_rows & m], label=target[stop_rows & m])
        bst = xgb.train(params, dfit, rounds, evals=[(dstop, "stop")], early_stopping_rounds=50, verbose_eval=False)
        models[h], best[h] = bst, int(bst.best_iteration)
    return models, best


def apply_corrector(models, F, dl, ml, compact: bool = True) -> np.ndarray:
    out = np.array(dl, dtype=np.float32, copy=True)
    for h, bst in models.items():
        r = bst.predict(xgb.DMatrix(stack_features(F, dl, ml, h, compact)), iteration_range=(0, bst.best_iteration + 1))
        out[:, h - 1] += r
    return out
