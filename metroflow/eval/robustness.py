"""Failure testing: how do the trained models behave when the detectors misbehave? Nothing is retrained.

    python -m metroflow.eval.robustness --stage graph     # PyTorch process: DCRNN and STGAT forecasts for every scenario
    python -m metroflow.eval.robustness --stage trees     # XGBoost process: XGBoost, hybrid, scoring

Only the INPUTS of the test windows are broken; targets stay the real readings. Every third test window is used (one
origin per 15 minutes, all 24 days). Random scenarios are drawn three times and averaged.

Scenarios
  dead     a share of sensors reports nothing for the whole test period (flagged missing, so inputs are imputed)
  stuck    a share of sensors silently repeats its free-flow speed (looks like a valid reading: the dangerous case)
  delay    the whole feed is late: the last readings of every window are unknown
  noise    Gaussian error on every reading
  spikes   a share of readings is replaced by a random speed

Writes results/robustness/{scores,summary}.csv
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from metroflow.data.dataset import PROCESSED, load_adjacency, load_scaler, load_split
from metroflow.data.load import load_speed
from metroflow.data.preprocess import N_FEATURES, load_config, slot_mean_table, split_bounds, time_features
from metroflow.features.build import build_features, flatten_targets, static_features
from metroflow.train.baselines import ROOT

OUT = ROOT / "results" / "robustness"
TMP = PROCESSED / "strict" / "robust"
STRIDE = 3
DRAWS = (0, 1, 2)
SHARES = (10, 20, 30)
REPORT = {3: 15, 6: 30, 12: 60}


def scenarios() -> list[dict]:
    out = [{"kind": "clean", "level": 0, "draw": 0}]
    out += [{"kind": k, "level": s, "draw": d} for k in ("dead", "stuck") for s in SHARES for d in DRAWS]
    out += [{"kind": "delay", "level": m, "draw": 0} for m in (5, 10, 15, 30)]
    out += [{"kind": "noise", "level": s, "draw": 0} for s in (5, 10)]
    out += [{"kind": "spikes", "level": s, "draw": 0} for s in SHARES]
    return out


def key(s: dict) -> str:
    return f"{s['kind']}_{s['level']}_{s['draw']}"


def affected_sensors(s: dict, n: int) -> np.ndarray:
    """(N,) bool. The same draw breaks the same sensors in 'dead' and 'stuck', and a higher share contains the lower one."""
    if s["kind"] not in ("dead", "stuck"):
        return np.ones(n, bool) if s["kind"] != "clean" else np.zeros(n, bool)
    order = np.random.default_rng(100 + s["draw"]).permutation(n)
    out = np.zeros(n, bool)
    out[order[: int(round(n * s["level"] / 100))]] = True
    return out


def corrupt(raw: np.ndarray, s: dict, ff: np.ndarray) -> tuple[np.ndarray, int]:
    """Broken copy of the (T, N) readings (NaN = outage) and the feed delay in steps. Real outages stay outages."""
    out, rng = raw.copy(), np.random.default_rng(7 + s["draw"])
    real = ~np.isnan(raw)
    if s["kind"] == "dead":
        out[:, affected_sensors(s, raw.shape[1])] = np.nan
    elif s["kind"] == "stuck":
        a = affected_sensors(s, raw.shape[1])
        out[:, a] = np.where(real[:, a], ff[a], np.nan)
    elif s["kind"] == "noise":
        out = np.where(real, np.clip(raw + rng.normal(0, s["level"], raw.shape), 1, None), np.nan)
    elif s["kind"] == "spikes":
        hit = real & (rng.random(raw.shape) < s["level"] / 100)
        out = np.where(hit, rng.uniform(1, 70, raw.shape), raw)
    return out.astype(np.float32), s["level"] // 5 if s["kind"] == "delay" else 0


def rebuild_inputs(origins, raw, tfeat, slot_tab, slots, scaler, cfg, delay: int = 0) -> np.ndarray:
    """The preprocessed inputs X, built from broken readings of which only those up to origin - delay have arrived.

    With the real readings and no delay this equals metroflow.data.preprocess.build_split (tests/test_robustness.py)."""
    h, limit = cfg["window"]["history"], cfg["preprocess"]["impute_ffill_limit"]
    n, cols = raw.shape[1], np.arange(raw.shape[1])
    last_valid = np.maximum.accumulate(np.where(~np.isnan(raw), np.arange(len(raw))[:, None], -1), axis=0)
    raw0 = np.nan_to_num(raw)
    X = np.empty((len(origins), h, n, N_FEATURES), np.float32)
    for a in range(0, len(origins), 512):
        o = origins[a:a + 512]
        t_in = o[:, None] + np.arange(-h + 1, 1)
        arrived = t_in <= (o - delay)[:, None]
        lv = last_valid[np.minimum(t_in, (o - delay)[:, None])]
        usable = (lv >= (o - h + 1)[:, None, None]) & ((t_in[:, :, None] - lv) <= limit)
        value = np.where(usable, raw0[np.clip(lv, 0, None), cols], slot_tab[slots[t_in]])
        X[a:a + 512, :, :, 0] = (value - scaler["mu"]) / scaler["sd"]
        X[a:a + 512, :, :, 1] = ~np.isnan(raw[t_in]) & arrived[:, :, None]
        X[a:a + 512, :, :, 2:] = tfeat[t_in][:, :, None, :]
    return X


class World:
    """Everything the two stages share: clean readings, train-only statistics, test windows and their true targets."""

    def __init__(self):
        self.cfg, self.scaler, speed = load_config(), load_scaler(), load_speed()
        v = speed.values.astype(np.float32)
        self.raw = np.where(v > 0, v, np.nan)
        self.train_end = split_bounds(len(v), self.cfg["split"])["train"][1]
        self.ff = np.nanpercentile(self.raw[: self.train_end], 85, axis=0).astype(np.float32)
        self.tfeat, self.slot_tab = time_features(speed.index), slot_mean_table(self.raw, speed.index, self.train_end)
        self.slots = (speed.index.hour * 12 + speed.index.minute // 5).to_numpy()
        self.adj = load_adjacency()["adj"]
        self.static = static_features(speed, self.adj, self.train_end)
        t = load_split("test", keys=["origin_idx", "Y", "M_y"])
        self.origins = t["origin_idx"][::STRIDE]
        self.Y, self.M = flatten_targets(t["Y"][::STRIDE], t["M_y"][::STRIDE])

    def inputs(self, s: dict) -> np.ndarray:
        broken, delay = corrupt(self.raw, s, self.ff)
        return rebuild_inputs(self.origins, broken, self.tfeat, self.slot_tab, self.slots, self.scaler, self.cfg, delay)


# --------------------------------------------------------------------------- stage 1: graph models (PyTorch)
def stage_graph() -> None:
    import torch

    from metroflow.train.graph import CKPT, RunConfig, build_model, device, predict_rows

    TMP.mkdir(parents=True, exist_ok=True)
    w, dev = World(), device()
    models = {}
    for run in ("dcrnn_s0", "stgat_s0"):
        if not (CKPT / run / "config.json").exists():
            print(run, "has no checkpoint, skipped")
            continue
        cfg = json.loads((CKPT / run / "config.json").read_text())["config"]
        m = build_model(RunConfig(**{k: v for k, v in cfg.items() if k in RunConfig.__dataclass_fields__}), w.adj).to(dev)
        m.load_state_dict(torch.load(CKPT / run / "best.pt", map_location=dev))
        models[run.split("_")[0]] = m
    for s in scenarios():
        X = w.inputs(s)
        for name, m in models.items():
            np.save(TMP / f"{key(s)}_{name}.npy", predict_rows(m, X, w.scaler, dev))
        print(key(s), "done", flush=True)


# --------------------------------------------------------------------------- stage 2: trees, hybrid, scoring
def stage_trees(seed: int = 0) -> None:
    import xgboost as xgb

    from metroflow.eval import uncertainty as unc
    from metroflow.models import hybrid

    OUT.mkdir(parents=True, exist_ok=True)
    w = World()

    def boosters(folder):
        out = {}
        for h in range(1, 13):
            b = xgb.Booster()
            b.load_model(folder / f"h{h:02d}.json")
            out[h] = b
        return out

    base, corr = boosters(ROOT / "checkpoints" / "xgb"), boosters(ROOT / "checkpoints" / "hybrid" / f"s{seed}")
    best = {h: (b.best_iteration if b.attr("best_iteration") else b.num_boosted_rounds() - 1) + 1 for h, b in base.items()}
    conf = unc.calibrate(unc.load_val_test(seed)["val"])
    n, rows = len(w.ff), []
    ff_rows = np.tile(w.ff, len(w.origins))
    for s in scenarios():
        F = build_features(w.inputs(s), w.scaler, w.static)
        d = xgb.DMatrix(F)
        px = np.stack([base[h].predict(d, iteration_range=(0, best[h])) for h in range(1, 13)], axis=1)
        pd_ = np.load(TMP / f"{key(s)}_dcrnn.npy")
        ph = pd_.copy()
        for h, b in corr.items():
            ph[:, h - 1] += b.predict(xgb.DMatrix(hybrid.stack_features(F, pd_, px, h, compact=True)))
        preds = {"Persistence": np.repeat(F[:, :1], 12, 1), "XGBoost": px, "DCRNN": pd_, "Hybrid": ph}
        if (TMP / f"{key(s)}_stgat.npy").exists():
            preds["STGAT"] = np.load(TMP / f"{key(s)}_stgat.npy")
        aff = np.tile(affected_sensors(s, n), len(w.origins))
        for step, mins in REPORT.items():
            m, y = w.M[:, step - 1], np.nan_to_num(w.Y[:, step - 1])
            lo, hi = conf.interval(ph[:, step - 1], np.abs(px - pd_)[:, step - 1], ff_rows, step - 1, 0.8)
            for name, P in preds.items():
                err = np.abs(P[:, step - 1] - y)
                r = {**s, "model": name, "minutes": mins, "mae": float(err[m].mean())}
                for lab, sel in (("affected", m & aff), ("healthy", m & ~aff)):
                    r[f"mae_{lab}"] = float(err[sel].mean()) if sel.any() and s["kind"] in ("dead", "stuck") else np.nan
                r["coverage_80_pct"] = 100 * float(((y >= lo) & (y <= hi))[m].mean()) if name == "Hybrid" else np.nan
                rows.append(r)
        print(key(s), {r["model"]: round(r["mae"], 2) for r in rows[-len(preds):]}, flush=True)
    raw = pd.DataFrame(rows)
    raw.to_csv(OUT / "scores.csv", index=False, float_format="%.4f")
    g = raw.groupby(["kind", "level", "model", "minutes"], sort=False)
    out = g[["mae", "mae_affected", "mae_healthy", "coverage_80_pct"]].mean().join(g.mae.std().rename("mae_draw_std")).reset_index()
    clean = out[out.kind == "clean"].set_index(["model", "minutes"]).mae
    out["increase_pct"] = [100 * (r.mae / clean[(r.model, r.minutes)] - 1) for r in out.itertuples()]
    out.to_csv(OUT / "summary.csv", index=False, float_format="%.4f")
    print(out[out.minutes == 30].pivot_table(index=["kind", "level"], columns="model", values="mae", sort=False).round(2).to_string())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["graph", "trees"], required=True)
    {"graph": stage_graph, "trees": stage_trees}[ap.parse_args().stage]()


if __name__ == "__main__":
    main()
