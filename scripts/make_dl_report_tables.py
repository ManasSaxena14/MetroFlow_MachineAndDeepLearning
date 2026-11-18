"""Generate the tables and quoted numbers of the DL project report from results/ (docs/dl_report/generated/).

    python scripts/make_dl_report_tables.py && cd docs/dl_report && tectonic dl_report.tex      (or: make dl-report)

Every number in docs/dl_report/dl_report.tex is read from a file in results/, so rerunning after new results keeps the
report consistent with the repository. Nothing here trains or evaluates a model.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "docs" / "dl_report" / "generated"
WORDS = {15: "Fifteen", 30: "Thirty", 60: "Sixty"}
HYBRID = "Hybrid: DCRNN + XGBoost (ours)"
ORDER = ["Persistence", "Historical average", "Ridge per sensor", "Multi-output Random Forest", "LSTM (no graph)",
         "XGBoost per horizon", "XGBoost + neighbours", "STGAT (ours)", "DCRNN", HYBRID]
SHORT = {HYBRID: "Hybrid: DCRNN + XGBoost (ours)", "Multi-output Random Forest": "Random Forest (multi-output)"}


def num(x, d: int = 2) -> str:
    return "--" if x is None or pd.isna(x) else f"{x:.{d}f}"


def pval(p: float) -> str:
    return r"$<$0.0001" if p < 1e-4 else f"{p:.3f}"


def macro(name: str, value) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}\n"


def thousands(n: int) -> str:
    return f"{n:,}".replace(",", r"\,")


def label(model: str, final: pd.DataFrame) -> str:
    name = SHORT.get(model, model)
    if model in final.index and final.loc[model, "runs"] > 1:
        name += r"$^{\dagger}$"
    return name


def bold_best(values: dict[str, float], d: int = 2) -> dict[str, str]:
    best = min(values.values())
    return {k: (r"\textbf{" + num(v, d) + "}") if abs(v - best) < 1e-9 else num(v, d) for k, v in values.items()}


def write(name: str, lines: list[str]) -> None:
    (OUT / name).write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    final = pd.read_csv(RES / "final.csv").set_index("model")
    long = pd.read_csv(RES / "final_by_horizon.csv")
    sig = pd.read_csv(RES / "significance.csv")
    abl = pd.read_csv(RES / "ablations.csv")
    hyb = pd.read_csv(RES / "hybrid_ablation.csv")
    deg = pd.read_csv(RES / "gain_by_degree.csv")
    ons = pd.read_csv(RES / "gain_at_onset.csv")
    att = pd.read_csv(RES / "attention_shares.csv")
    graph = pd.read_csv(RES / "graph_models.csv")
    p1 = json.loads((RES / "phase1_findings.json").read_text())
    p4 = json.loads((RES / "phase4_findings.json").read_text())
    p5 = json.loads((RES / "phase5_findings.json").read_text())
    p6 = json.loads((RES / "phase6_findings.json").read_text())
    p7 = json.loads((RES / "phase7_maps.json").read_text())
    hj = json.loads((RES / "hybrid.json").read_text())
    rows = [r for r in ORDER if r in final.index]

    # ------------------------------------------------------------------ quoted numbers
    m = ""
    keys = {"Persistence": "Persist", "XGBoost per horizon": "Xgb", "XGBoost + neighbours": "XgbNb", "LSTM (no graph)": "Lstm",
            "STGAT (ours)": "Stgat", "DCRNN": "Dcrnn", HYBRID: "Hybrid", "Multi-output Random Forest": "Rf"}
    for model, key in keys.items():
        for mins, w in WORDS.items():
            m += macro(f"{key}MAE{w}", num(final.loc[model, f"MAE_{mins}min"]))
            m += macro(f"{key}RMSE{w}", num(final.loc[model, f"RMSE_{mins}min"]))
            m += macro(f"{key}MAPE{w}", num(final.loc[model, f"MAPE_{mins}min"], 1))
        m += macro(f"{key}PeakSixty", num(final.loc[model, "peak_MAE_60min"]))
        m += macro(f"{key}OffpeakSixty", num(final.loc[model, "offpeak_MAE_60min"]))
    xg = final.loc["XGBoost per horizon"]
    for model, key in (("DCRNN", "Dcrnn"), ("STGAT (ours)", "Stgat"), (HYBRID, "Hybrid"), ("XGBoost + neighbours", "XgbNb")):
        for mins, w in WORDS.items():
            base, mod = xg[f"MAE_{mins}min"], final.loc[model, f"MAE_{mins}min"]
            m += macro(f"{key}Gain{w}", num(100 * (base - mod) / base, 1))
    h, d = final.loc[HYBRID], final.loc["DCRNN"]
    for mins, w in WORDS.items():
        m += macro(f"HybridOverDcrnn{w}", num(100 * (d[f"MAE_{mins}min"] - h[f"MAE_{mins}min"]) / d[f"MAE_{mins}min"], 1))
    m += macro("PeakGainDcrnn", num(100 * (xg.peak_MAE_60min - d.peak_MAE_60min) / xg.peak_MAE_60min, 1))
    m += macro("PeakGainHybrid", num(100 * (xg.peak_MAE_60min - h.peak_MAE_60min) / xg.peak_MAE_60min, 1))
    m += macro("TestDays", int(sig.test_days.max()))
    st60 = sig[(sig.model == "STGAT (ours)") & (sig.baseline == "XGBoost per horizon") & (sig.minutes == 60)]
    m += macro("StgatSixtyPBoot", pval(st60.p_bootstrap.max()))
    m += macro("StgatSixtyPDm", pval(st60.p_dm.max()))
    m += macro("StgatSixtyDays", int(st60.days_model_better.min()))
    hv = p6["hybrid_vs_dcrnn"]
    m += macro("HybridMinDays", min(v["min_days_better"] for v in hv.values()))
    ref = p5["reference_check"]
    for mins, w in WORDS.items():
        m += macro(f"Ref{w}", num(ref[f"{mins}min"]["ours"]))
        m += macro(f"Pub{w}", num(ref[f"{mins}min"]["published"]))
    m += macro("RefWorstPct", num(max(abs(v["diff_pct"]) for v in ref.values()), 1))
    a = att.set_index(["situation", "attends_to"]).mean_share_pct
    for sit, sk in (("all other windows", "Normal"), ("slowdown ahead (30 min)", "Ahead")):
        for who, wk in (("self", "Self"), ("upstream neighbour", "Up"), ("downstream neighbour", "Down")):
            m += macro(f"Att{wk}{sk}", num(a[(sit, who)], 1))
    m += macro("AttSensors", int(att.sensors.iloc[0]))
    m += macro("AttShifted", p5["sensors_shifting_downstream"])
    m += macro("AttWindows", thousands(p5["slowdown_ahead_windows"]))
    pv = deg.pivot(index="degree", columns="model", values="mae_60min")
    for b, w in (("0-4", "Low"), ("13+", "High")):
        base = pv.loc[b, "XGBoost per horizon"]
        m += macro(f"DegGain{w}", num(100 * (base - pv.loc[b, "DCRNN"]) / base, 1))
    o = ons[ons.group == "onset"].pivot(index="model", columns="minutes", values="mae")
    for model, key in (("XGBoost per horizon", "Xgb"), ("DCRNN", "Dcrnn"), (HYBRID, "Hybrid"), ("STGAT (ours)", "Stgat")):
        for mins, w in WORDS.items():
            m += macro(f"Onset{key}{w}", num(o.loc[model, mins], 1))
    m += macro("OnsetTargets", thousands(int(ons.n[ons.group == "onset"].iloc[0])))
    m += macro("TestOnsets", thousands(p4["test_onsets"]))
    sbs = p7["side_by_side"]
    m += macro("MapTime", sbs["target_time"][:16])
    m += macro("MapCongested", sbs["actual_congested"])
    m += macro("MapXgbCaught", sbs["XGBoost"]["congested_caught"])
    m += macro("MapGraphCaught", sbs["DCRNN"]["congested_caught"])
    m += macro("MapXgbLevel", num(sbs["XGBoost"]["level_accuracy_pct"], 1))
    m += macro("MapGraphLevel", num(sbs["DCRNN"]["level_accuracy_pct"], 1))
    m += macro("MapSensorsBetter", p7["error_map"]["sensors_graph_better"])
    m += macro("MapMedianGain", num(p7["error_map"]["median_gain_mph"]))
    ts = p7["time_space"]
    m += macro("TsDay", ts["day"])
    m += macro("TsSensors", len(ts["sensors"]))
    m += macro("TsKm", num(ts["km"], 1))
    ab = abl.set_index("run")
    for run, key in (("stgat_identity_s0", "NoGraph"), ("stgat_forward_s0", "Fwd"), ("stgat_backward_s0", "Bwd"),
                     ("stgat_undirected_s0", "Undir"), ("stgat_random_s0", "Random"), ("stgat_1layer_s0", "OneLayer"),
                     ("stgat_drop0_s0", "NoDrop"), ("dcrnn_forward_s0", "DcrnnFwd"), ("dcrnn_k1_s0", "DcrnnKone")):
        m += macro(f"Abl{key}Sixty", num(ab.loc[run, "MAE_60min"]))
        m += macro(f"Abl{key}Delta", num(ab.loc[run, "delta_60min_vs_main"]))
    m += macro("StgatParams", thousands(int(ab.loc["stgat_s0", "parameters"])))
    m += macro("DcrnnParams", thousands(int(ab.loc["dcrnn_s0", "parameters"])))
    m += macro("HybFitWindows", thousands(hj["fit_windows"]))
    m += macro("HybStopWindows", thousands(hj["stop_windows"]))
    pr = p1["propagation"]
    m += macro("PropDownFirst", pr["downstream_first"])
    m += macro("PropUpFirst", pr["upstream_first"])
    m += macro("PropLag", num(pr["median_lag_min"], 0))
    m += macro("PropKmh", num(pr["shockwave_kmh"], 0))
    g = p1["graph"]
    m += macro("GraphEdges", thousands(g["edges"]))
    m += macro("GraphDegree", num(g["mean_degree"]))
    m += macro("ZeroPct", num(p1["missingness"]["zero_pct"], 1))
    m += macro("NTests", sum(1 for f in (ROOT / "tests").glob("test_*.py")
                             for line in f.read_text().splitlines() if line.startswith("def test_")))
    (OUT / "macros.tex").write_text(m)

    # ------------------------------------------------------------------ main table: MAE / RMSE / MAPE by horizon
    cols = [f"{k}_{x}min" for k in ("MAE", "RMSE", "MAPE") for x in (15, 30, 60)]
    best = {c: bold_best({r: final.loc[r, c] for r in rows}, 1 if c.startswith("MAPE") else 2) for c in cols}
    lines = []
    for r in rows:
        lines.append(f"{label(r, final)} & " + " & ".join(best[c][r] for c in cols) + r" \\")
        if r == "XGBoost + neighbours":
            lines.append(r"\midrule")
    write("tab_main.tex", lines)

    # ------------------------------------------------------------------ all 12 horizons, MAE (mean over runs)
    sel = ["Persistence", "LSTM (no graph)", "XGBoost per horizon", "STGAT (ours)", "DCRNN", HYBRID]
    lg = long[(long.horizon != "avg")].copy()
    lg["horizon"] = lg.horizon.astype(int)
    mean = lg.groupby(["model", "horizon"]).mae.mean().unstack(0)
    lines = []
    for hh in range(1, 13):
        vals = bold_best({mdl: mean.loc[hh, mdl] for mdl in sel})
        lines.append(f"{hh} & {5 * hh} & " + " & ".join(vals[mdl] for mdl in sel) + r" \\")
    write("tab_horizons.tex", lines)

    # ------------------------------------------------------------------ peak / off-peak / weekend at 60 min
    lines = []
    pk = {c: bold_best({r: final.loc[r, c] for r in rows}) for c in ("peak_MAE_60min", "offpeak_MAE_60min", "weekend_MAE_60min")}
    for r in rows:
        ratio = final.loc[r, "peak_MAE_60min"] / final.loc[r, "offpeak_MAE_60min"]
        lines.append(f"{label(r, final)} & {pk['peak_MAE_60min'][r]} & {pk['offpeak_MAE_60min'][r]} & "
                     f"{pk['weekend_MAE_60min'][r]} & {ratio:.2f} \\\\")
        if r == "XGBoost + neighbours":
            lines.append(r"\midrule")
    write("tab_period.tex", lines)

    # ------------------------------------------------------------------ significance
    lines = []
    for model, base, lab in (("STGAT (ours)", "XGBoost per horizon", "STGAT vs XGBoost"),
                             ("DCRNN", "XGBoost per horizon", "DCRNN vs XGBoost"),
                             (HYBRID, "XGBoost per horizon", "Hybrid vs XGBoost"),
                             (HYBRID, "DCRNN (same seed)", "Hybrid vs DCRNN"),
                             ("XGBoost + neighbours", "XGBoost per horizon", "XGB + neighbours vs XGBoost")):
        s = sig[(sig.model == model) & (sig.baseline == base)]
        for i, mins in enumerate((15, 30, 60)):
            r = s[s.minutes == mins]
            lines.append(f"{lab if i == 0 else ''} & {mins} & {num(r.gain_mph.mean(), 3)} & {num(r.gain_pct.mean(), 1)} & "
                         f"[{num(r.ci95_low.min(), 3)}, {num(r.ci95_high.max(), 3)}] & "
                         f"{int(r.days_model_better.min())}/{int(r.test_days.max())} & "
                         f"{pval(r.p_bootstrap.max())} & {pval(r.p_dm.max())} \\\\")
        lines.append(r"\midrule")
    write("tab_sig.tex", lines[:-1])

    # ------------------------------------------------------------------ per-seed stability of the graph models
    t = graph[(graph.split == "test") & (graph.protocol == "strict")]
    lines = []
    for model, gsel in (("stgat", t[(t.model == "stgat") & (t.graph == "bidirectional") & (t.gat_layers == 2)
                                    & ~t.run.str.contains("drop")]),
                        ("dcrnn", t[(t.model == "dcrnn") & (t.supports == "dual") & (t.k == 2)])):
        for _, r in gsel.sort_values("seed").iterrows():
            meta = json.loads((RES / "graph" / f"{r.run}.json").read_text())
            best_ep = min(meta["history"], key=lambda x: x["val_mae"])["epoch"]
            lines.append(f"{'STGAT' if model == 'stgat' else 'DCRNN'} & {r.seed} & {thousands(int(r.parameters))} & "
                         f"{r.epochs_run} & {best_ep} & {num(meta['best_val_mae'], 3)} & {num(r.MAE_15min, 3)} & "
                         f"{num(r.MAE_30min, 3)} & {num(r.MAE_60min, 3)} \\\\")
        lines.append(r"\midrule")
    write("tab_seeds.tex", lines[:-1])

    # ------------------------------------------------------------------ reference protocol vs published
    lines = []
    for mins in (15, 30, 60):
        v = ref[f"{mins}min"]
        lines.append(f"{mins} & {num(v['published'])} & {num(v['ours'])} & {v['diff_pct']:+.1f} & "
                     f"{num(final.loc['DCRNN', f'MAE_{mins}min'])} \\\\")
    write("tab_reference.tex", lines)

    # ------------------------------------------------------------------ ablations
    lines = []
    runs = list(abl.run)
    for i, r in abl.iterrows():
        v = r.variant.replace("(main)", r"\textit{(main)}")
        deltas = [("--" if r[f"delta_{x}min_vs_main"] == 0 else f"{r[f'delta_{x}min_vs_main']:+.3f}") for x in (15, 30, 60)]
        lines.append(f"{v} & {thousands(int(r.parameters))} & {num(r.MAE_15min, 3)} & {num(r.MAE_30min, 3)} & "
                     f"{num(r.MAE_60min, 3)} & {deltas[2]} \\\\")
        if r.run.startswith("stgat") and i + 1 < len(runs) and runs[i + 1].startswith("dcrnn"):
            lines.append(r"\midrule")
    write("tab_ablation.tex", lines)

    # ------------------------------------------------------------------ hybrid designs
    names = {"DCRNN alone": "DCRNN alone (backbone)",
             "Blend": r"Blend: $w_h\,\hat y^{\mathrm{DCRNN}} + (1-w_h)\,\hat y^{\mathrm{XGB}}$",
             "Residual corrector, all": "Residual corrector, all 46 features, deep trees",
             "Residual corrector, compact": r"Residual corrector, compact and regularised \textit{(main)}"}
    vals = {c: bold_best({r.variant: r[c] for _, r in hyb.iterrows()}, 3) for c in ("MAE_15min", "MAE_30min", "MAE_60min")}
    lines = []
    for _, r in hyb.iterrows():
        name = next(v for k, v in names.items() if r.variant.startswith(k))
        cells = [f"{vals[f'MAE_{x}min'][r.variant]}\\,$\\pm$\\,{r[f'MAE_{x}min_std']:.3f}" for x in (15, 30, 60)]
        lines.append(f"{name} & " + " & ".join(cells) + r" \\")
    write("tab_hybrid.tex", lines)

    # ------------------------------------------------------------------ corrector feature importance (mean over seeds, horizons)
    imps = [pd.read_csv(f, index_col=0) for f in sorted((RES / "hybrid").glob("corrector_importance_s*.csv"))]
    imp = pd.concat(imps).groupby(level=0).mean()
    imp = pd.DataFrame({"all": imp.mean(axis=1), "h3": imp["3"], "h6": imp["6"], "h12": imp["12"]}).sort_values("all", ascending=False)
    pretty = {"xgb_minus_dl": "XGBoost forecast minus DCRNN forecast", "sin_tod": "time of day (sine)",
              "cos_tod": "time of day (cosine)", "is_weekend": "weekend flag", "xgb_forecast": "XGBoost forecast",
              "dl_minus_now": "DCRNN forecast minus current speed", "dl_forecast": "DCRNN forecast",
              "speed_t": "current speed", "free_flow_p85": "free-flow speed (85th percentile)",
              "delta_1": "change over the last 5 min", "delta_3": "change over the last 15 min"}
    lines = [f"{pretty.get(k, k)} & {100 * r['all']:.1f} & {100 * r.h3:.1f} & {100 * r.h6:.1f} & {100 * r.h12:.1f} \\\\"
             for k, r in imp.iterrows()]
    write("tab_corrector.tex", lines)
    w = pd.DataFrame({k: v["blend_weight_on_dcrnn"] for k, v in hj["runs"].items()}).mean(axis=1)
    (OUT / "macros.tex").write_text((OUT / "macros.tex").read_text()
                                    + macro("BlendWFifteen", num(w[2])) + macro("BlendWSixty", num(w[11]))
                                    + macro("CorrTopShare", num(100 * imp["all"].iloc[0], 0)))

    # ------------------------------------------------------------------ gain by connectivity
    sel = ["XGBoost per horizon", "STGAT (ours)", "DCRNN", HYBRID]
    lines = []
    for b in ("0-4", "5-8", "9-12", "13+"):
        n = int(deg[deg.degree == b].sensors.iloc[0])
        g = 100 * (pv.loc[b, "XGBoost per horizon"] - pv.loc[b, "DCRNN"]) / pv.loc[b, "XGBoost per horizon"]
        lines.append(f"{b.replace('-', '--')} & {n} & " + " & ".join(num(pv.loc[b, s]) for s in sel) + f" & {g:.1f} \\\\")
    write("tab_degree.tex", lines)

    # ------------------------------------------------------------------ onset vs normal
    sel = ["XGBoost per horizon", "XGBoost + neighbours", "STGAT (ours)", "DCRNN", HYBRID]
    on = ons.pivot_table(index="model", columns=["group", "minutes"], values="mae")
    lines = []
    for s in sel:
        onv = [on.loc[s, ("onset", x)] for x in (15, 30, 60)]
        nov = [on.loc[s, ("normal", x)] for x in (15, 30, 60)]
        lines.append(f"{label(s, final)} & " + " & ".join(num(v) for v in onv) + " & " + " & ".join(num(v) for v in nov) + r" \\")
    best_on = {x: min(on.loc[s, ("onset", x)] for s in sel) for x in (15, 30, 60)}
    for x in (15, 30, 60):     # bold the best onset cell per horizon
        for i, s in enumerate(sel):
            v = num(on.loc[s, ("onset", x)])
            if abs(on.loc[s, ("onset", x)] - best_on[x]) < 1e-9:
                lines[i] = lines[i].replace(f" {v} ", r" \textbf{" + v + "} ", 1)
    write("tab_onset.tex", lines)

    # ------------------------------------------------------------------ attention
    lines = []
    for sit, lab in (("all other windows", "All other windows"), ("slowdown ahead (30 min)", "Slowdown within 30 min")):
        lines.append(f"{lab} & {num(a[(sit, 'self')], 1)} & {num(a[(sit, 'upstream neighbour')], 1)} & "
                     f"{num(a[(sit, 'downstream neighbour')], 1)} \\\\")
    write("tab_attention.tex", lines)
    print("wrote", ", ".join(p.name for p in sorted(OUT.glob("*.tex"))))


if __name__ == "__main__":
    main()
