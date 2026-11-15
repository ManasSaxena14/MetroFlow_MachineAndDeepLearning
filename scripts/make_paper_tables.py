"""Generate the IEEE paper's tables and quoted numbers from results/ (docs/paper/generated/).

    python scripts/make_paper_tables.py && cd docs/paper && tectonic paper.tex

Every number in docs/paper/paper.tex comes from here, so rerunning after new results keeps the paper consistent.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "docs" / "paper" / "generated"
WORDS = {15: "Fifteen", 30: "Thirty", 60: "Sixty"}


def num(x, d=2) -> str:
    return "--" if x is None or pd.isna(x) else f"{x:.{d}f}"


def pval(p: float) -> str:
    return r"$<$0.0001" if p < 1e-4 else f"{p:.3f}"


def macro(name: str, value) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    final = pd.read_csv(RES / "final.csv").set_index("model")
    sig = pd.read_csv(RES / "significance.csv")
    p5 = json.loads((RES / "phase5_findings.json").read_text())
    p6 = json.loads((RES / "phase6_findings.json").read_text())
    p7 = json.loads((RES / "phase7_maps.json").read_text())
    abl = pd.read_csv(RES / "ablations.csv") if (RES / "ablations.csv").exists() else pd.DataFrame()
    hyb = pd.read_csv(RES / "hybrid_ablation.csv") if (RES / "hybrid_ablation.csv").exists() else pd.DataFrame()
    deg = pd.read_csv(RES / "gain_by_degree.csv")
    ons = pd.read_csv(RES / "gain_at_onset.csv")
    att = pd.read_csv(RES / "attention_shares.csv") if (RES / "attention_shares.csv").exists() else pd.DataFrame()

    m = ""
    short = {"XGBoost per horizon": "Xgb", "LSTM (no graph)": "Lstm", "STGAT (ours)": "Stgat", "DCRNN": "Dcrnn",
             "XGBoost + neighbours": "XgbNb", "Persistence": "Persist", "Hybrid: DCRNN + XGBoost (ours)": "Hybrid"}
    for model, key in short.items():
        if model in final.index:
            for mins, w in WORDS.items():
                m += macro(f"{key}MAE{w}", num(final.loc[model, f"MAE_{mins}min"]))
    xg = final.loc["XGBoost per horizon"]
    for model, key in (("DCRNN", "Dcrnn"), ("STGAT (ours)", "Stgat"), ("Hybrid: DCRNN + XGBoost (ours)", "Hybrid")):
        if model in final.index:
            for mins, w in WORDS.items():
                g = 100 * (xg[f"MAE_{mins}min"] - final.loc[model, f"MAE_{mins}min"]) / xg[f"MAE_{mins}min"]
                m += macro(f"{key}Gain{w}", num(g, 1))
    if "Hybrid: DCRNN + XGBoost (ours)" in final.index:
        h, d = final.loc["Hybrid: DCRNN + XGBoost (ours)"], final.loc["DCRNN"]
        for mins, w in WORDS.items():
            m += macro(f"HybridOverDcrnn{w}", num(100 * (d[f"MAE_{mins}min"] - h[f"MAE_{mins}min"]) / d[f"MAE_{mins}min"], 1))
    m += macro("TestDays", int(sig.test_days.max()))
    dc = sig[(sig.model == "DCRNN") & (sig.baseline == "XGBoost per horizon")]
    m += macro("DcrnnMinDaysBetter", int(dc.days_model_better.min()) if len(dc) else "--")
    st60 = sig[(sig.model == "STGAT (ours)") & (sig.baseline == "XGBoost per horizon") & (sig.minutes == 60)]
    if len(st60):
        m += macro("StgatSixtyPBoot", pval(st60.p_bootstrap.max()))
        m += macro("StgatSixtyPDm", pval(st60.p_dm.max()))
        m += macro("StgatSixtyDays", int(st60.days_model_better.min()))
    ref = p5.get("reference_check")
    if ref:
        m += macro("RefWorstPct", num(max(abs(v["diff_pct"]) for v in ref.values()), 1))
        for mins, w in WORDS.items():
            m += macro(f"Ref{w}", num(ref[f"{mins}min"]["ours"]))
    if len(att):
        a = att.set_index(["situation", "attends_to"]).mean_share_pct
        m += macro("AttDownNormal", num(a[("all other windows", "downstream neighbour")], 1))
        m += macro("AttDownAhead", num(a[("slowdown ahead (30 min)", "downstream neighbour")], 1))
        m += macro("AttSensors", int(att.sensors.iloc[0]))
        m += macro("AttShifted", p5.get("sensors_shifting_downstream", "--"))
    best = p6["best_graph_model"]
    p = deg.pivot(index="degree", columns="model", values="mae_60min")
    for b, w in (("0-4", "Low"), ("13+", "High")):
        m += macro(f"DegGain{w}", num(100 * (p.loc[b, "XGBoost per horizon"] - p.loc[b, best]) / p.loc[b, "XGBoost per horizon"], 1))
    o = ons[ons.group == "onset"].pivot(index="model", columns="minutes", values="mae")
    for model, key in (("XGBoost per horizon", "Xgb"), ("DCRNN", "Dcrnn")):
        for mins, w in WORDS.items():
            m += macro(f"Onset{key}{w}", num(o.loc[model, mins], 1))
    sbs = p7["side_by_side"]
    m += macro("MapCongested", sbs["actual_congested"])
    m += macro("MapXgbCaught", sbs["XGBoost"]["congested_caught"])
    m += macro("MapGraphCaught", sbs.get("DCRNN", sbs.get("STGAT", {})).get("congested_caught", "--"))
    m += macro("MapSensorsBetter", p7["error_map"]["sensors_graph_better"])
    if len(abl):
        a = abl.set_index("run")
        for run, key in (("stgat_identity_s0", "NoGraph"), ("stgat_forward_s0", "Fwd"), ("stgat_backward_s0", "Bwd"),
                         ("stgat_random_s0", "Random"), ("stgat_drop0_s0", "NoDrop"), ("dcrnn_forward_s0", "DcrnnFwd"),
                         ("dcrnn_k1_s0", "DcrnnKone")):
            if run in a.index:
                m += macro(f"Abl{key}Sixty", num(a.loc[run, "MAE_60min"]))
                m += macro(f"Abl{key}Delta", num(a.loc[run, "delta_60min_vs_main"]))
    (OUT / "macros.tex").write_text(m)

    # ---- main table
    order = ["Persistence", "Historical average", "Ridge per sensor", "Multi-output Random Forest", "LSTM (no graph)",
             "XGBoost per horizon", "XGBoost + neighbours", "STGAT (ours)", "DCRNN", "Hybrid: DCRNN + XGBoost (ours)"]
    cols = ["MAE_15min", "MAE_30min", "MAE_60min", "RMSE_60min", "MAPE_60min", "peak_MAE_60min"]
    rows = [r for r in order if r in final.index]
    best_col = {c: final.loc[rows, c].min() for c in cols}
    lines = []
    for r in rows:
        name = r.replace("Hybrid: DCRNN + XGBoost (ours)", "Hybrid (ours)").replace("Multi-output Random Forest", "Random Forest")
        if final.loc[r, "runs"] > 1:
            name += r"$^{\dagger}$"
        vals = [(r"\textbf{" + num(final.loc[r, c], 1 if c == "MAPE_60min" else 2) + "}")
                if abs(final.loc[r, c] - best_col[c]) < 1e-9 else num(final.loc[r, c], 1 if c == "MAPE_60min" else 2) for c in cols]
        lines.append(f"{name} & " + " & ".join(vals) + r" \\")
        if r == "XGBoost + neighbours":
            lines.append(r"\midrule")
    (OUT / "tab_main.tex").write_text("\n".join(lines) + "\n")

    # ---- significance table
    lines = []
    for model, base, label in (("DCRNN", "XGBoost per horizon", "DCRNN vs XGB"),
                               ("STGAT (ours)", "XGBoost per horizon", "STGAT vs XGB"),
                               ("Hybrid: DCRNN + XGBoost (ours)", "XGBoost per horizon", "Hybrid vs XGB"),
                               ("Hybrid: DCRNN + XGBoost (ours)", "DCRNN (same seed)", "Hybrid vs DCRNN")):
        s = sig[(sig.model == model) & (sig.baseline == base)]
        for mins in (15, 30, 60):
            r = s[s.minutes == mins]
            if r.empty:
                continue
            lab = label if mins == 15 else ""
            lines.append(f"{lab} & {mins} & {num(r.gain_pct.mean(), 1)} & {int(r.days_model_better.min())}/{int(r.test_days.max())}"
                         f" & {pval(r.p_bootstrap.max())} & {pval(r.p_dm.max())} \\\\")
        if len(s):
            lines.append(r"\midrule")
    (OUT / "tab_sig.tex").write_text("\n".join(lines[:-1]) + "\n" if lines else "")

    # ---- ablation table
    lines = []
    if len(abl):
        runs = list(abl.run)
        for i, r in abl.iterrows():
            v = r.variant.replace("(main)", r"\textit{(main)}")
            lines.append(f"{v} & {num(r.MAE_15min)} & {num(r.MAE_30min)} & {num(r.MAE_60min)} & "
                         f"{'--' if r.delta_60min_vs_main == 0 else f'{r.delta_60min_vs_main:+.2f}'} \\\\")
            if r.run.startswith("stgat") and i + 1 < len(runs) and runs[i + 1].startswith("dcrnn"):
                lines.append(r"\midrule")                      # separate the STGAT and DCRNN blocks
    (OUT / "tab_ablation.tex").write_text("\n".join(lines) + "\n")

    # ---- hybrid variants
    short_names = {"DCRNN alone": "DCRNN alone", "Blend": "Blend, per-horizon weight",
                   "Residual corrector, all": "Corrector, all 46 features", "Residual corrector, compact": "Corrector, compact (main)"}
    lines = []
    for _, r in hyb.iterrows():
        name = next(v for k, v in short_names.items() if r.variant.startswith(k))
        cells = [f"{r[f'MAE_{x}min']:.3f}$\\pm${r[f'MAE_{x}min_std']:.3f}".replace("$\\pm$0.", "$\\pm$.")
                 if r.runs > 1 else f"{r[f'MAE_{x}min']:.3f}" for x in (15, 30, 60)]
        lines.append(f"{name} & " + " & ".join(cells) + r" \\")
    (OUT / "tab_hybrid.tex").write_text("\n".join(lines) + "\n")
    if "Hybrid: DCRNN + XGBoost (ours)" in final.index:
        for model, key in (("XGBoost per horizon", "Xgb"), ("XGBoost + neighbours", "XgbNb"), ("DCRNN", "Dcrnn"),
                           ("Hybrid: DCRNN + XGBoost (ours)", "Hybrid")):
            m2 = macro(f"{key}RMSESixty", num(final.loc[model, "RMSE_60min"]))
            (OUT / "macros.tex").write_text((OUT / "macros.tex").read_text() + m2)
    n_tests = sum(1 for f in (ROOT / "tests").glob("test_*.py") for line in f.read_text().splitlines() if line.startswith("def test_"))
    (OUT / "macros.tex").write_text((OUT / "macros.tex").read_text() + macro("NTests", n_tests))
    print("wrote", ", ".join(p.name for p in sorted(OUT.glob("*.tex"))))


if __name__ == "__main__":
    main()
