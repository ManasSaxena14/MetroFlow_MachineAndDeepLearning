"""Export two test days for the dashboard's replay: readings, hybrid forecasts with ranges, congestion probability, alerts.

    python -m metroflow.eval.replay [--seed 0] [--budget 50]        # needs metroflow.eval.alerts

Days are picked by rule, never by how good the forecasts look: the test weekday with the most jams, and the one with the
median number. Speeds are stored in half-mph steps in one byte (255 = no reading) to keep the file small.

Writes results/replay/{busiest,typical}.npz, sensors.csv, links.csv, summary.json and propagation.csv (the readings behind
figure 5d, the worked example of a jam travelling upstream)
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from metroflow.data.dataset import load_adjacency, load_split
from metroflow.data.load import load_distances, load_locations, load_speed
from metroflow.eval import alerts as al
from metroflow.eval import uncertainty as unc
from metroflow.train.baselines import ROOT

OUT = ROOT / "results" / "replay"
COVER = 0.8
ALERT_MODEL = "Hybrid, calibrated probability"
EXAMPLE = ("717462", "717472", "2012-05-02 05:40", "2012-05-02 08:10")          # downstream, upstream, from, to (figure 5d)


def pack(v: np.ndarray) -> np.ndarray:
    return np.where(np.isnan(v), 255, np.clip(np.round(np.nan_to_num(v) * 2), 0, 254)).astype(np.uint8)


def pick_days(starts: np.ndarray, index: pd.DatetimeIndex, first: pd.Timestamp, last: pd.Timestamp) -> dict[str, pd.Timestamp]:
    """Busiest and median full weekday of the test block by number of jam starts."""
    per = pd.Series(starts.sum(1), index=index).groupby(index.normalize()).sum()
    per = per[(per.index >= first) & (per.index <= last) & (per.index.dayofweek < 5)].sort_values(kind="stable")
    return {"busiest": per.index[-1], "typical": per.index[(len(per) - 1) // 2]}


def export_example(speed: pd.DataFrame) -> None:
    down, up, a, b = EXAMPLE
    x = speed.loc[a:b, [down, up]].where(lambda v: v > 0)
    x.columns = ["downstream", "upstream"]
    x.rename_axis("time").round(2).to_csv(OUT / "propagation.csv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=int, default=50)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    data, ff, speed = unc.load_val_test(args.seed), unc.free_flow(), load_speed()
    v = np.where(speed.values > 0, speed.values, np.nan).astype(np.float32)
    export_example(speed)
    starts, conf, d = al.jam_starts(v, ff), unc.calibrate(data["val"]), data["test"]
    org = load_split("test", keys=["origin_idx"])["origin_idx"]
    ops = pd.read_csv(ROOT / "results" / "alerts" / "operating_points.csv")
    thr = float(ops[(ops.model == ALERT_MODEL) & (ops.budget_per_day == args.budget)].threshold.iloc[0])
    score, now = al.probability_score(conf, d), v[org]
    fired = al.raise_alerts(score, ~np.isnan(now) & (now >= unc.CONGESTED * ff), thr)

    first = speed.index[org[0]].normalize() + pd.Timedelta(days=1)                # whole days only
    last = speed.index[org[-1]].normalize() - pd.Timedelta(days=1)
    days = pick_days(starts, speed.index, first, last)
    info = {"cover": COVER, "alert_model": ALERT_MODEL, "budget_per_day": args.budget, "threshold": thr, "days": {}}
    for name, day in days.items():
        t = int(speed.index.get_loc(day))
        s = int(t - org[0])                                                       # first origin of the day
        rows = slice(s * al.N, (s + 288) * al.N)
        f, lo, hi = (np.empty((288, 12, al.N), np.float32) for _ in range(3))
        for h in range(12):
            a, b = conf.interval(d["f"][rows, h], d["gap"][rows, h], d["ff"][rows], h, COVER)
            f[:, h], lo[:, h], hi[:, h] = (x.reshape(288, al.N) for x in (d["f"][rows, h], a, b))
        np.savez_compressed(OUT / f"{name}.npz", actual=pack(v[t - 12:t + 288 + 12]), forecast=pack(f), lo=pack(lo), hi=pack(hi),
                            prob=np.round(score[s:s + 288] * 100).astype(np.uint8), alert=fired[s:s + 288], jam=starts[t:t + 288])
        info["days"][name] = {"date": str(day.date()), "jams": int(starts[t:t + 288].sum()), "alerts": int(fired[s:s + 288].sum())}
    g = load_adjacency()
    ids = [str(x) for x in g["sensor_ids"]]
    off = (g["adj"] > 0) & ~np.eye(al.N, dtype=bool)
    loc = load_locations(sensor_ids=ids).reset_index()
    loc.assign(free_flow=ff.round(1), links=off.sum(0) + off.sum(1)).to_csv(OUT / "sensors.csv", index=False)
    pos, dist = {s: i for i, s in enumerate(ids)}, load_distances()
    dist = dist[dist["from"].isin(pos) & dist["to"].isin(pos)]
    dist = pd.DataFrame({"src": dist["from"].map(pos), "dst": dist["to"].map(pos), "m": dist["cost"]})
    dist = dist[(dist.src != dist.dst) & (dist.m > 0)]
    dist[dist.m <= 3500].sort_values("m").groupby("src").head(2).to_csv(OUT / "links.csv", index=False)   # nearest downstream sensors
    (OUT / "summary.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
