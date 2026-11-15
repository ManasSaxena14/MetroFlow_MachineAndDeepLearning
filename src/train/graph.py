"""Phase 5 runner: train a graph model, score it on val and test with the SAME evaluation path as the baselines.

    python -m src.train.graph stgat --seed 0                         # main model
    python -m src.train.graph stgat --graph identity --seed 0        # ablation: no graph (equal-budget control)
    python -m src.train.graph dcrnn --seed 0 [--k 2] [--supports dual]
    python -m src.train.graph dcrnn --protocol dcrnn --seed 0        # reference protocol, to compare with published numbers
    python -m src.train.graph --table                                # merge results/graph/*.csv -> results/graph_models.csv

Each run writes
    checkpoints/graph/<name>/best.pt + config.json   the weights and everything needed to rebuild them
    data/processed/<protocol>/preds/<name>_{val,test}.npy   predictions in the baselines' row layout (S*N, 12), mph
    results/graph/<name>.csv / .json                 per-horizon MAE / RMSE / MAPE (val + test), training history

Training (both models): masked MAE on inverse-scaled speed, Adam, MultiStepLR, grad-norm clip 5, early stopping on
validation masked MAE. Windows are shuffled only inside the train block. Test numbers are never used to choose anything.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
import torch

from src.data.dataset import PROCESSED, load_adjacency, load_scaler, load_split
from src.eval.evaluate import REPORT_HORIZONS, evaluate_rows
from src.features.build import flatten_targets
from src.models.graph_layers import diffusion_supports, road_graph
from src.models.graph_models import DCRNN, STGAT

ROOT = PROCESSED.parents[1]
RESULTS = ROOT / "results" / "graph"
CKPT = ROOT / "checkpoints" / "graph"


@dataclass
class RunConfig:
    model: str = "stgat"
    protocol: str = "strict"
    seed: int = 0
    graph: str = "bidirectional"        # STGAT edge variant (graph_layers.GRAPH_VARIANTS)
    gat_layers: int = 2
    supports: str = "dual"              # DCRNN: dual | forward | undirected
    k: int = 2                          # DCRNN diffusion hops
    hidden: int = 64
    batch: int = 32
    lr: float = 1e-3
    epochs: int = 40
    patience: int = 8
    milestones: list = field(default_factory=lambda: [20, 30])
    epoch_frac: float = 0.5             # each epoch sees a fresh random half of the training windows
    grad_clip: float = 5.0
    tag: str = ""

    @property
    def name(self) -> str:
        if self.tag:
            return self.tag
        parts = [self.model]
        if self.model == "stgat":
            parts += [] if self.graph == "bidirectional" else [self.graph]
            parts += [] if self.gat_layers == 2 else [f"{self.gat_layers}layer"]
        else:
            parts += [] if self.supports == "dual" else [self.supports]
            parts += [] if self.k == 2 else [f"k{self.k}"]
        if self.protocol != "strict":
            parts.append(self.protocol)
        return "_".join(parts) + f"_s{self.seed}"


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def build_model(cfg: RunConfig, adj: np.ndarray) -> torch.nn.Module:
    if cfg.model == "stgat":
        ei, ea = road_graph(adj, cfg.graph, seed=cfg.seed)
        return STGAT(ei, ea, hidden=cfg.hidden, gat_layers=cfg.gat_layers)
    if cfg.model == "dcrnn":
        return DCRNN(diffusion_supports(adj, cfg.supports), units=cfg.hidden, k=cfg.k)
    raise ValueError(cfg.model)


def masked_mae(pred_mph: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m = ~torch.isnan(y)
    return (pred_mph[m] - y[m]).abs().mean() if m.any() else pred_mph.sum() * 0


def predict_rows(model, X: np.ndarray, scaler: dict, dev, batch: int = 64) -> np.ndarray:
    """X (S, 12, N, 7) -> rows (S * N, 12) in mph, same row order as the feature table (window * N + sensor)."""
    model.eval()
    out = []
    with torch.no_grad():
        for a in range(0, len(X), batch):
            z = model(torch.from_numpy(np.ascontiguousarray(X[a:a + batch])).to(dev))
            out.append((z * scaler["sd"] + scaler["mu"]).permute(0, 2, 1).reshape(-1, z.shape[1]).float().cpu().numpy())
    return np.concatenate(out)


def val_mae(P: np.ndarray, Y: np.ndarray, M: np.ndarray) -> float:
    Yr, Mr = flatten_targets(Y, M)
    return float(np.abs(P - np.nan_to_num(Yr))[Mr].mean())


def train(cfg: RunConfig, log=print) -> dict:
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    dev = device()
    scaler = load_scaler(cfg.protocol)
    adj = load_adjacency()["adj"]
    model = build_model(cfg, adj).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    tr = load_split("train", cfg.protocol, keys=["X", "Y"])
    va = load_split("val", cfg.protocol, keys=["X", "Y", "M_y"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=cfg.milestones, gamma=0.1)
    mu, sd = scaler["mu"], scaler["sd"]
    out_dir = CKPT / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"[{cfg.name}] {n_params:,} parameters on {dev}; train windows {len(tr['X']):,}")
    best, bad, seen, history = np.inf, 0, 0, []
    for ep in range(1, cfg.epochs + 1):
        model.train()
        t0, losses = time.time(), []
        order = rng.permutation(len(tr["X"]))[: int(len(tr["X"]) * cfg.epoch_frac)]
        for a in range(0, len(order), cfg.batch):
            rows = np.sort(order[a:a + cfg.batch])
            x = torch.from_numpy(tr["X"][rows]).to(dev)
            y = torch.from_numpy(tr["Y"][rows]).to(dev)
            z = model(x, (y - mu) / sd, seen) if cfg.model == "dcrnn" else model(x)
            loss = masked_mae(z * sd + mu, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            losses.append(loss.item())
            seen += 1
        sched.step()
        v = val_mae(predict_rows(model, va["X"], scaler, dev), va["Y"], va["M_y"])
        history.append({"epoch": ep, "train_mae": float(np.mean(losses)), "val_mae": v, "seconds": round(time.time() - t0, 1)})
        log(f"[{cfg.name}] epoch {ep:3d}  train {np.mean(losses):.3f}  val {v:.3f}  ({time.time() - t0:.0f}s)")
        if v < best - 1e-4:
            best, bad = v, 0
            torch.save(model.state_dict(), out_dir / "best.pt")
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=dev))
    meta = {"config": asdict(cfg), "name": cfg.name, "parameters": n_params, "best_val_mae": best,
            "epochs_run": len(history), "history": history}
    (out_dir / "config.json").write_text(json.dumps(meta, indent=2))
    del tr
    score(cfg, model, scaler, dev, meta)
    return meta


def score(cfg: RunConfig, model, scaler: dict, dev, meta: dict) -> pd.DataFrame:
    RESULTS.mkdir(parents=True, exist_ok=True)
    pred_dir = PROCESSED / cfg.protocol / "preds"
    pred_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for split in ("val", "test"):
        d = load_split(split, cfg.protocol, keys=["X", "Y", "M_y"])
        P = predict_rows(model, d["X"], scaler, dev)
        np.save(pred_dir / f"{cfg.name}_{split}.npy", P)
        Yr, Mr = flatten_targets(d["Y"], d["M_y"])
        df = evaluate_rows(Yr, P, Mr)
        df.insert(0, "split", split)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(RESULTS / f"{cfg.name}.csv", index=False)
    (RESULTS / f"{cfg.name}.json").write_text(json.dumps(meta, indent=2))
    t = out[(out.split == "test")].set_index("horizon")
    print(f"[{cfg.name}] test MAE 15/30/60 min: " + " / ".join(f"{t.loc[str(h), 'mae']:.3f}" for h in REPORT_HORIZONS))
    return out


def make_table() -> pd.DataFrame:
    """results/graph_models.csv: one row per run and split, MAE / RMSE / MAPE at 15 / 30 / 60 min and pooled."""
    rows = []
    for f in sorted(RESULTS.glob("*.csv")):
        meta = json.loads(f.with_suffix(".json").read_text())
        df = pd.read_csv(f, dtype={"horizon": str})
        for split in ("val", "test"):
            d = df[df.split == split].set_index("horizon")
            keys = ("model", "protocol", "seed", "graph", "gat_layers", "supports", "k")
            r = {"run": f.stem, **{k: meta["config"][k] for k in keys},
                 "split": split, "parameters": meta["parameters"], "epochs_run": meta["epochs_run"]}
            for h, minutes in REPORT_HORIZONS.items():
                for m in ("mae", "rmse", "mape"):
                    r[f"{m.upper()}_{minutes}min"] = round(float(d.loc[str(h), m]), 3)
            for m in ("mae", "rmse", "mape"):
                r[f"{m.upper()}_avg"] = round(float(d.loc["avg", m]), 3)
            rows.append(r)
    table = pd.DataFrame(rows)
    table.to_csv(ROOT / "results" / "graph_models.csv", index=False)
    return table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", choices=["stgat", "dcrnn"])
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--skip-done", action="store_true", help="do nothing if results/graph/<run>.csv already exists")
    for f, v in asdict(RunConfig()).items():
        if f in ("model", "milestones"):
            continue
        ap.add_argument(f"--{f.replace('_', '-')}", type=type(v), default=v)
    args = ap.parse_args()
    if args.model:
        kw = {k: getattr(args, k) for k in asdict(RunConfig()) if k not in ("model", "milestones")}
        cfg = RunConfig(model=args.model, **kw)
        if args.model == "dcrnn" and args.lr == RunConfig.lr:
            cfg.lr = 1e-2                       # the reference learning rate for DCRNN
        if args.skip_done and (RESULTS / f"{cfg.name}.csv").exists():
            print(f"[{cfg.name}] already done, skipped")
            return
        train(cfg)
    if args.table or args.model:
        print(make_table().query("split == 'test'").to_string(index=False))


if __name__ == "__main__":
    main()
