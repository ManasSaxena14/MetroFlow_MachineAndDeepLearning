"""Build the DL-phase slide deck (Phases 5 to 7) from the result files, on the same template as the AML deck.

    python scripts/make_dl_slides.py        # -> docs/presentation/DL_Graph_Models.pptx

Every number on the slides is read from results/; rerun after new results and the deck updates. The template
(theme, fonts, layouts) is taken from docs/presentation/AML_Traffic_Flow_Forecasting.pptx so both decks match.
Speaker notes carry the talking points for each slide.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_MARKER_STYLE
from pptx.enum.dml import MSO_THEME_COLOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "docs" / "figures"
TEMPLATE = ROOT / "docs" / "presentation" / "AML_Traffic_Flow_Forecasting.pptx"
OUT = ROOT / "docs" / "presentation" / "DL_Graph_Models.pptx"
FOOTER = "DL phase  |  Graph models for traffic forecasting on METR-LA"
HYBRID = "Hybrid: DCRNN + XGBoost (ours)"

T = MSO_THEME_COLOR
INK, WHITE, TINT, GREY = T.TEXT_1, T.BACKGROUND_1, T.BACKGROUND_2, T.ACCENT_6
BLUE, ORANGE, AQUA, YELLOW, RED = T.ACCENT_1, T.ACCENT_2, T.ACCENT_3, T.ACCENT_4, T.ACCENT_5
HEX = {"grid": RGBColor(0xDD, 0xE3, 0xEA)}          # chart gridlines: a light tint of the theme's navy


# --------------------------------------------------------------------------- data
def load() -> dict:
    d = {"final": pd.read_csv(RES / "final.csv").set_index("model"),
         "long": pd.read_csv(RES / "final_by_horizon.csv"),
         "sig": pd.read_csv(RES / "significance.csv"),
         "deg": pd.read_csv(RES / "gain_by_degree.csv"),
         "ons": pd.read_csv(RES / "gain_at_onset.csv"),
         "p6": json.loads((RES / "phase6_findings.json").read_text()),
         "p5": json.loads((RES / "phase5_findings.json").read_text()),
         "p7": json.loads((RES / "phase7_maps.json").read_text()),
         "graph": pd.read_csv(RES / "graph_models.csv")}
    ab = RES / "ablations.csv"
    d["abl"] = pd.read_csv(ab) if ab.exists() else pd.DataFrame()
    hy = RES / "hybrid_ablation.csv"
    d["hyb"] = pd.read_csv(hy) if hy.exists() else pd.DataFrame()
    sh = RES / "attention_shares.csv"
    d["att"] = pd.read_csv(sh) if sh.exists() else pd.DataFrame()
    return d


def f2(x) -> str:
    return f"{x:.2f}"


def pct(a, b) -> float:
    """Relative reduction of b versus a, in percent."""
    return 100 * (a - b) / a


# --------------------------------------------------------------------------- drawing helpers
def _color(target, theme):
    """Set a theme colour on a font (target.color) or directly on a fill's ColorFormat."""
    (target.color if hasattr(target, "color") else target).theme_color = theme


def text(slide, x, y, w, h, value, size=14, bold=False, color=INK, head=False, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, name=None, italic=False):
    """Text box with zero padding. value may be a string or a list of (text, overrides) paragraphs."""
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        box.name = name
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    paras = value if isinstance(value, list) else [(value, {})]
    for i, (t, o) in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = o.get("align", align)
        if i:
            p.space_before = Pt(o.get("space", 6))
        r = p.add_run()
        r.text = t
        f = r.font
        f.size = Pt(o.get("size", size))
        f.bold = o.get("bold", bold)
        f.italic = o.get("italic", italic)
        _color(f, o.get("color", color))
        if o.get("head", head):
            f.name = "+mj-lt"
    return box


def shape(slide, kind, x, y, w, h, fill, name=None, radius=None, line=None):
    s = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        s.name = name
    s.fill.solid()
    _color(s.fill.fore_color, fill)
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.theme_color = line
        s.line.width = Pt(1.25)
    if radius is not None:
        s.adjustments[0] = radius
    s.shadow.inherit = False
    s.text_frame.text = ""
    return s


def card(slide, x, y, w, h, fill=TINT, name=None):
    return shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h, fill, name, radius=0.05)


def badge(slide, x, y, label, fill=BLUE, d=0.42, size=15):
    shape(slide, MSO_SHAPE.OVAL, x, y, d, d, fill, name=f"Badge {label}")
    text(slide, x, y, d, d, str(label), size=size, bold=True, color=WHITE, align=PP_ALIGN.CENTER,
         anchor=MSO_ANCHOR.MIDDLE, name=f"Badge text {label}")


def pill(slide, x, y, w, label, fill=AQUA):
    shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, 0.3, fill, name=f"Pill {label}", radius=0.5)
    text(slide, x, y, w, 0.3, label, size=12, bold=True, color=WHITE, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)


def signals(slide, y=0.5):
    for i, c in enumerate((RED, YELLOW, T.ACCENT_3)):
        shape(slide, MSO_SHAPE.OVAL, 8.78 + 0.24 * i, y, 0.16, 0.16, c, name=f"Signal {i}")


def slide_number(slide, layout):
    """python-pptx does not copy the layout's slide-number placeholder; clone it so numbers show."""
    for sp in layout.placeholders:
        if sp.placeholder_format.type is not None and "SLIDE_NUMBER" in str(sp.placeholder_format.type):
            slide.shapes._spTree.append(copy.deepcopy(sp._element))


def content(prs, title, subtitle, notes, layout="CONTENT"):
    lay = next(lay for lay in prs.slide_layouts if lay.name == layout)
    s = prs.slides.add_slide(lay)
    s.placeholders[101].text = title                         # the template's title placeholder has idx 101
    if subtitle:
        text(s, 0.5, 0.98, 8.9, 0.32, subtitle, size=16, color=GREY if layout == "CONTENT" else TINT,
             anchor=MSO_ANCHOR.MIDDLE, name="Subtitle")
    signals(s)
    slide_number(s, lay)
    s.notes_slide.notes_text_frame.text = notes
    return s


def stat(slide, x, y, w, h, number, label, num_size=26, name="Result"):
    card(slide, x, y, w, h, name=f"{name} card")
    text(slide, x + 0.15, y + 0.08, w - 0.3, 0.45, number, size=num_size, bold=True, head=True, name=f"{name} number")
    text(slide, x + 0.15, y + 0.56, w - 0.3, h - 0.62, label, size=12, name=f"{name} label")


def picture(slide, path, x, y, w=None, h=None):
    return slide.shapes.add_picture(str(path), Inches(x), Inches(y), Inches(w) if w else None, Inches(h) if h else None)


def style_chart(chart, legend=True, size=11):
    chart.font.size = Pt(size)
    chart.font.name = "+mn-lt"
    chart.font.color.theme_color = INK
    chart.has_legend = legend
    if legend:
        chart.legend.position = XL_LEGEND_POSITION.BOTTOM
        chart.legend.include_in_layout = False
        chart.legend.font.size = Pt(size)
    va = chart.value_axis
    va.has_major_gridlines = True
    va.major_gridlines.format.line.color.rgb = HEX["grid"]
    va.format.line.fill.background()
    va.tick_labels.font.size = Pt(size)
    ca = chart.category_axis
    ca.tick_labels.font.size = Pt(size)
    ca.format.line.color.rgb = HEX["grid"]


def table(slide, x, y, w, rows, col_w, header_fill=INK, size=12, best_cols=(), row_h=0.32):
    """rows[0] = header. best_cols: column indices where the lowest value is bolded (error metrics)."""
    shp = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(row_h * len(rows)))
    tb = shp.table
    for j, cw in enumerate(col_w):
        tb.columns[j].width = Inches(cw)
    best = {}
    for j in best_cols:
        vals = [float(r[j]) for r in rows[1:] if r[j] not in ("", "–")]
        best[j] = min(vals)
    for i, r in enumerate(rows):
        tb.rows[i].height = Inches(row_h)
        for j, v in enumerate(r):
            c = tb.cell(i, j)
            c.margin_left = c.margin_right = Inches(0.06)
            c.margin_top = c.margin_bottom = Inches(0.02)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            c.fill.solid()
            _color(c.fill.fore_color, header_fill if i == 0 else (TINT if i % 2 == 0 else WHITE))
            p = c.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.RIGHT
            p.font.size = Pt(size)                            # also sizes empty cells, so rows stay even
            run = p.add_run()
            run.text = str(v)
            run.font.size = Pt(size)
            _color(run.font, WHITE if i == 0 else INK)
            run.font.bold = i == 0 or (j in best and v not in ("", "–") and abs(float(v) - best[j]) < 1e-9)
    return tb


# --------------------------------------------------------------------------- slides
def title_slide(prs, d):
    lay = next(lay for lay in prs.slide_layouts if lay.name == "DEFAULT")
    s = prs.slides.add_slide(lay)
    text(s, 0.6, 0.7, 5.0, 0.3, "DEEP LEARNING  |  PROJECT T036", size=12, bold=True, color=BLUE, name="Kicker")
    text(s, 0.6, 1.2, 5.0, 1.9, "Learning the road network", size=38, bold=True, head=True, name="Title")
    text(s, 0.6, 3.15, 4.9, 1.0, "Graph attention and diffusion models on METR-LA: do they close the gap that "
         "per-sensor models leave?", size=16, color=GREY, name="Subtitle")
    text(s, 0.6, 4.75, 5.2, 0.3, "Team T036  |  DL phase: Phases 5 to 7", size=14, bold=True, name="Author")
    # motif: a road of sensors with messages flowing both ways (the bidirectional graph)
    xs = [5.9, 6.75, 7.6, 8.45, 9.3]
    for i in range(4):
        a = s.shapes.add_connector(1, Inches(xs[i] + 0.24), Inches(2.3), Inches(xs[i + 1]), Inches(2.3))
        a.line.color.theme_color = GREY
        a.line.width = Pt(2)
    for i, x in enumerate(xs):
        shape(s, MSO_SHAPE.OVAL, x, 2.18, 0.24, 0.24, RED if i == 4 else BLUE, name=f"Sensor {i}")
    shape(s, MSO_SHAPE.RIGHT_ARROW, 6.2, 2.75, 2.9, 0.22, GREY, name="Traffic arrow")
    text(s, 5.9, 3.03, 3.7, 0.3, "traffic flow", size=12, color=GREY, align=PP_ALIGN.CENTER)
    shape(s, MSO_SHAPE.LEFT_ARROW, 6.2, 3.45, 2.9, 0.22, ORANGE, name="Message arrow")
    text(s, 5.9, 3.73, 3.7, 0.5, "each sensor also listens downstream,\nwhere jams start", size=12, bold=True,
         color=ORANGE, align=PP_ALIGN.CENTER)
    signals(s, y=0.7)
    s.notes_slide.notes_text_frame.text = (
        "The AML part showed that per-sensor models miss congestion that comes from neighbouring sensors. "
        "This part builds two graph deep-learning models that read the road network, scores them under exactly the "
        "same protocol, and checks where and why they help.")


def recap(prs, d):
    f = d["final"]
    s = content(prs, "Where the AML part left us", "Strong per-sensor models, and a measured gap they cannot close",
                "XGBoost is the strongest model that looks at one sensor at a time. Phase 4 measured what it misses: "
                "error explodes right after congestion starts, linked sensors share their errors, and hand-made "
                "neighbour features help a little. That is the case for a model that learns from the graph.")
    xg = f.loc["XGBoost per horizon"]
    stat(s, 0.5, 1.5, 2.85, 1.2, f"{f2(xg.MAE_15min)} | {f2(xg.MAE_30min)} | {f2(xg.MAE_60min)}",
         "XGBoost test MAE (mph) at 15, 30 and 60 min: the bar to beat", num_size=20, name="XGB")
    stat(s, 3.575, 1.5, 2.85, 1.2, "5.9 to 6.4x", "higher XGBoost error in the 30 min after a congestion onset",
         name="Onset")
    stat(s, 6.65, 1.5, 2.85, 1.2, "0.20 vs 0.00", "error correlation of linked vs unconnected sensors (60 min)",
         name="Residual")
    xn = f.loc["XGBoost + neighbours"]
    card(s, 0.5, 2.95, 9.0, 0.95, name="Neighbour card")
    text(s, 0.7, 3.05, 8.6, 0.8, [
        ("Hand-made neighbour speeds help, but only a little", {"bold": True, "size": 15}),
        (f"XGBoost + neighbours: {f2(xn.MAE_15min)} | {f2(xn.MAE_30min)} | {f2(xn.MAE_60min)} mph, "
         f"{pct(xg.MAE_15min, xn.MAE_15min):.1f}% to {pct(xg.MAE_60min, xn.MAE_60min):.1f}% better. "
         "A model must learn which neighbours matter, in which direction, over several hops.", {"size": 13, "space": 3}),
    ])
    shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, 0.5, 4.1, 9.0, 0.6, INK, name="Question banner", radius=0.12)
    text(s, 0.7, 4.1, 8.6, 0.6, "Question for the DL phase: does a model that learns from the road graph close this "
         "gap, under exactly the same protocol?", size=14, bold=True, color=WHITE, anchor=MSO_ANCHOR.MIDDLE)


def models(prs, d):
    g = d["graph"].query("split == 'test' and protocol == 'strict'")
    p_st = int(g[g.run == "stgat_s0"].parameters.iloc[0]) if (g.run == "stgat_s0").any() else 0
    p_dc = int(g[g.run == "dcrnn_s0"].parameters.iloc[0]) if (g.run == "dcrnn_s0").any() else 0
    s = content(prs, "Two graph models",
                "Same input (12 steps x 207 sensors x 7 features), same output (12 horizons), same road graph",
                "STGAT is our graph attention network: a GRU summarises each sensor's last hour, then two GATv2 layers "
                "let every sensor attend to its road neighbours' summaries, in both directions, with the link weight "
                "and direction as edge features. DCRNN is the reference diffusion model: the graph enters every GRU "
                "step through forward and backward random walks, and an encoder-decoder emits the 12 horizons one by one "
                "with scheduled sampling. Both are written in plain PyTorch; our attention layer is tested to give "
                "exactly the same output as PyTorch Geometric's GATv2Conv.")
    cols = [("STGAT (ours)", ORANGE, p_st, [
        ("GRU per sensor", "each sensor's own 60-minute history, shared weights"),
        ("2 x GATv2 over the road graph", "attend to upstream AND downstream neighbours; edge features: weight, direction"),
        ("Head", "12 horizons as the change from the last speed")]),
        ("DCRNN (Li et al., 2018)", AQUA, p_dc, [
        ("Diffusion convolution", "forward and backward random walks on the directed graph, K = 2 hops"),
        ("DCGRU encoder-decoder", "the graph enters every one of the 24 recurrent steps"),
        ("Scheduled sampling", "decoder fed the truth early in training, its own forecasts later")])]
    for k, (name, color, params, steps) in enumerate(cols):
        x = 0.5 + k * 4.6
        card(s, x, 1.5, 4.4, 3.25, name=f"{name} card")
        text(s, x + 0.2, 1.62, 2.9, 0.35, name, size=18, bold=True, head=True)
        pill(s, x + 3.0, 1.65, 1.2, f"{params / 1000:.0f}k params", fill=color)
        for i, (t1, t2) in enumerate(steps):
            yy = 2.15 + i * 0.84
            badge(s, x + 0.2, yy + 0.05, i + 1, fill=color, d=0.38, size=13)
            text(s, x + 0.72, yy, 3.5, 0.75, [(t1, {"bold": True, "size": 14}), (t2, {"size": 12, "space": 2})])


def direction(prs, d):
    abl = d["abl"].set_index("run") if len(d["abl"]) else pd.DataFrame()
    s = content(prs, "Design choice: listen downstream",
                "Phase 1: jams reach the downstream sensor first, 4 times more often than the upstream one",
                "The raw adjacency points from upstream to downstream, so ordinary message passing only lets a sensor "
                "hear its upstream neighbours. But jams grow backwards from bottlenecks, so the early warning is "
                "downstream. We add every road link in both directions with an is-reverse edge feature and let "
                "attention decide. The ablation keeps only one direction to test this.")
    # diagram: three sensors, forward and reverse messages
    xs = [0.9, 2.5, 4.1]
    for i, x in enumerate(xs):
        shape(s, MSO_SHAPE.OVAL, x, 2.35, 0.42, 0.42, BLUE if i < 2 else RED, name=f"Diagram sensor {i}")
        text(s, x - 0.3, 2.85, 1.02, 0.3, ["upstream", "sensor i", "downstream"][i], size=12, align=PP_ALIGN.CENTER,
             color=GREY)
    for i in range(2):
        shape(s, MSO_SHAPE.RIGHT_ARROW, xs[i] + 0.5, 2.25, 1.02, 0.18, GREY, name=f"Forward {i}")
        shape(s, MSO_SHAPE.LEFT_ARROW, xs[i] + 0.5, 2.7, 1.02, 0.18, ORANGE, name=f"Reverse {i}")
    text(s, 0.5, 1.6, 4.4, 0.5, [("grey: traffic direction (raw graph)", {"color": GREY, "size": 12}),
                                 ("orange: reverse edges we add (is_reverse = 1)", {"color": ORANGE, "size": 12,
                                                                                    "bold": True, "space": 2})])
    text(s, 0.5, 3.35, 4.4, 1.3, "Unit test: with forward-only edges, a jam at the downstream sensor cannot change "
         "sensor i's forecast at all; with both directions it does. Attention then learns how much to trust each "
         "direction.", size=13)
    if len(abl) and {"stgat_s0", "stgat_forward_s0", "stgat_backward_s0"} <= set(abl.index):
        rows = [["STGAT edges (seed 0)", "15 min", "30 min", "60 min"]]
        for run, lab in (("stgat_s0", "both directions"), ("stgat_forward_s0", "forward only"),
                         ("stgat_backward_s0", "backward only")):
            r = abl.loc[run]
            rows.append([lab, f2(r.MAE_15min), f2(r.MAE_30min), f2(r.MAE_60min)])
        text(s, 5.3, 1.6, 4.2, 0.3, "Ablation: test MAE (mph)", size=14, bold=True)
        table(s, 5.3, 2.0, 4.2, rows, [1.8, 0.8, 0.8, 0.8], best_cols=(1, 2, 3))
    else:
        card(s, 5.3, 1.6, 4.2, 2.0, name="Pending card")
        text(s, 5.5, 1.75, 3.8, 1.7, "Direction ablation: results pending (scripts/run_phase5.sh).", size=13)


def protocol(prs, d):
    g = d["graph"].query("split == 'test' and protocol == 'strict'")
    s = content(prs, "Same rules for every model", "The comparison is fair by construction, not by promise",
                "Every graph model reads the same Phase 2 files as the baselines, sees the same 60 minutes, is scored by "
                "the same evaluation function on the same masked test targets, and is tuned on validation only. Graph "
                "models get 3 seeds each; ablations one seed with the same budget as their main run.")
    rules = [("Same data", "Phase 2 test block: last 24 days, 6,843 windows, never used for tuning"),
             ("Same history", "12 steps (60 min) in, 12 out, for every model"),
             ("Same metric", "masked MAE / RMSE / MAPE in mph; outages never count"),
             ("Same tuning rule", "early stopping on validation MAE only"),
             ("Seeds", "3 per graph model, reported as mean and spread"),
             ("Reproducible", "results reproduce the committed baselines exactly")]
    for i, (h, b) in enumerate(rules):
        col, row = i % 2, i // 2
        x, y = 0.5 + col * 4.6, 1.5 + row * 0.98
        card(s, x, y, 4.4, 0.85, name=f"Rule card {i}")
        badge(s, x + 0.15, y + 0.21, i + 1, d=0.42)
        text(s, x + 0.72, y + 0.1, 3.55, 0.7, [(h, {"bold": True, "size": 14}), (b, {"size": 12, "space": 2})])
    eps = g.groupby("model").epochs_run.mean() if len(g) else pd.Series(dtype=float)
    text(s, 0.5, 4.5, 9.0, 0.3, "Training: Adam, masked MAE in mph, MultiStepLR, grad clip 5; STGAT up to 40 epochs "
         f"(used {eps.get('stgat', 0):.0f} on average), DCRNN up to 30 (used {eps.get('dcrnn', 0):.0f}); Apple M3 Pro GPU.",
         size=12, color=GREY)


def headline_chart(prs, d):
    f, long = d["final"], d["long"]
    best = d["p6"]["best_graph_model"]
    b, xg = f.loc[best], f.loc["XGBoost per horizon"]
    s = content(prs, "Graph models win at every horizon", "Test MAE (mph) by forecast horizon, lower is better",
                "One line per model, all 12 horizons, test block. Graph models are the mean of 3 seeds. The gap to the "
                "per-sensor models appears at the first horizon and stays: the graph models see a slowdown building at "
                "a neighbour before it reaches the sensor.")
    cd = CategoryChartData()
    cd.categories = [str(5 * h) for h in range(1, 13)]
    names = [m for m in ("LSTM (no graph)", "XGBoost per horizon", "STGAT (ours)", "DCRNN", HYBRID) if m in set(long.model)]
    for m in names:
        g = long[long.model == m].groupby("horizon").mae.mean().sort_index()
        cd.add_series(m, [round(v, 3) for v in g.values])
    ch = s.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, Inches(0.4), Inches(1.4), Inches(6.2), Inches(3.55), cd).chart
    style_chart(ch)
    colors = {"LSTM (no graph)": YELLOW, "XGBoost per horizon": BLUE, "STGAT (ours)": ORANGE, "DCRNN": AQUA, HYBRID: RED}
    for ser, m in zip(ch.series, names):
        ser.format.line.color.theme_color = colors[m]
        ser.format.line.width = Pt(2.25)
        ser.marker.style = XL_MARKER_STYLE.CIRCLE
        ser.marker.size = 5
        ser.marker.format.fill.solid()
        ser.marker.format.fill.fore_color.theme_color = colors[m]
        ser.marker.format.line.color.theme_color = colors[m]
        ser.smooth = False
    ch.value_axis.minimum_scale = 2.0
    ch.category_axis.has_title = True
    ch.category_axis.axis_title.text_frame.text = "minutes ahead"
    ch.category_axis.axis_title.text_frame.paragraphs[0].runs[0].font.size = Pt(11)
    stat(s, 6.8, 1.45, 2.7, 1.1, f"{f2(b.MAE_15min)} | {f2(b.MAE_30min)} | {f2(b.MAE_60min)}",
         f"{best} MAE (mph) at 15, 30, 60 min", num_size=20, name="Best")
    stat(s, 6.8, 2.67, 2.7, 1.1, f"-{pct(xg.MAE_60min, b.MAE_60min):.0f}%",
         f"vs XGBoost at 60 min ({f2(b.MAE_60min)} vs {f2(xg.MAE_60min)}); -{pct(xg.MAE_15min, b.MAE_15min):.0f}% at 15 min",
         name="Gain")
    ref = d["p5"].get("reference_check")
    if HYBRID in f.index:
        hb = f.loc[HYBRID]
        stat(s, 6.8, 3.89, 2.7, 1.1, f"-{pct(b.MAE_60min, hb.MAE_60min):.1f}%",
             f"extra gain of the hybrid over {best} at 60 min ({f2(hb.MAE_60min)} mph)", name="Hybrid")
    elif ref:
        worst = max(abs(v["diff_pct"]) for v in ref.values())
        stat(s, 6.8, 3.89, 2.7, 1.1, f"within {worst:.0f}%", "of the published DCRNN numbers on the reference split",
             name="Reference")
    else:
        st = f.loc["STGAT (ours)"]
        stat(s, 6.8, 3.89, 2.7, 1.1, f"-{pct(xg.MAE_15min, st.MAE_15min):.0f}%", "STGAT vs XGBoost at 15 min", name="STGAT")


def results_table(prs, d):
    f = d["final"]
    s = content(prs, "All models on the test block", "The last 24 days, never used for tuning; graph models: mean of 3 seeds",
                "Bold marks the best value in each column. RMSE punishes large misses, which happen in jams, so the "
                "graph models' larger RMSE gain matters. Peak = weekdays 07:00 to 09:59 and 16:00 to 19:59, judged on "
                "the predicted timestamp.")
    order = ["Persistence", "Historical average", "Ridge per sensor", "Multi-output Random Forest", "LSTM (no graph)",
             "XGBoost per horizon", "XGBoost + neighbours", "STGAT (ours)", "DCRNN", HYBRID]
    rows = [["Model", "MAE 15", "MAE 30", "MAE 60", "RMSE 60", "MAPE 60 %", "Peak MAE 60"]]
    for m in order:
        if m not in f.index:
            continue
        r = f.loc[m]
        rows.append([m.replace(HYBRID, "Hybrid (ours)"), f2(r.MAE_15min), f2(r.MAE_30min), f2(r.MAE_60min),
                     f2(r.RMSE_60min), f"{r.MAPE_60min:.1f}",
                     f2(r.peak_MAE_60min)])
    table(s, 0.5, 1.4, 9.0, rows, [2.7, 1.0, 1.0, 1.0, 1.05, 1.1, 1.15], best_cols=(1, 2, 3, 4, 5, 6), size=12, row_h=0.29)
    text(s, 0.5, 4.65, 9.0, 0.3, "MAE and RMSE in mph; minutes = forecast horizon. Full tables: results/final.csv, "
         "results/final_by_horizon.csv.", size=11, color=GREY)


def significance(prs, d):
    sig = d["sig"]
    s = content(prs, "Is the gain real?", "Paired tests over the 24 test days (days, not points, are the unit)",
                "Errors within a day are strongly correlated, so treating 1.2 million predictions as independent would "
                "make any difference look significant. We aggregate errors per day, then run a paired bootstrap over "
                "days (10,000 resamples) and a Diebold-Mariano test with the small-sample correction. Every seed is "
                "tested; we report the worst one.")
    rows = [["Comparison", "Minutes", "Gain %", "95% CI (mph)", "Days better", "p (bootstrap)", "p (DM)"]]
    for m, base, lab in (("DCRNN", "XGBoost per horizon", "DCRNN vs XGB"), ("STGAT (ours)", "XGBoost per horizon", "STGAT vs XGB"),
                         (HYBRID, "DCRNN (same seed)", "Hybrid vs DCRNN")):
        sm = sig[(sig.model == m) & (sig.baseline == base)]
        for mins in (15, 30, 60):
            r = sm[sm.minutes == mins]
            if r.empty:
                continue
            worst = r.loc[r.p_bootstrap.idxmax()]
            pb = "< 0.0001" if r.p_bootstrap.max() == 0 else f"{r.p_bootstrap.max():.3f}"
            pdm = "< 0.0001" if r.p_dm.max() < 1e-4 else f"{r.p_dm.max():.3f}"
            days = f"{int(r.days_model_better.min())} / {int(worst.test_days)}"
            rows.append([lab if mins == 15 else "", str(mins), f"{r.gain_pct.mean():.1f}",
                         f"{worst.ci95_low:.2f} to {worst.ci95_high:.2f}", days, pb, pdm])
    table(s, 0.5, 1.45, 9.0, rows, [2.2, 0.85, 0.85, 1.45, 1.15, 1.25, 1.25], size=12, row_h=0.285)
    text(s, 0.5, 4.65, 9.0, 0.3, "Gain = reduction in MAE vs the baseline (mean over seeds). Days better and p-values: worst seed.",
         size=11, color=GREY)


def ablations(prs, d):
    ab = d["abl"]
    s = content(prs, "Ablations: what matters?", "Test MAE at 60 min (mph), seed 0, same budget as the main run",
                "Each row removes or changes one thing. No graph keeps self-loops only: the same model with no "
                "neighbours, an equal-budget no-graph control. A random graph with the same number of links tests "
                "whether the real road structure matters. Direction variants test the Phase 1 finding.")
    if ab.empty:
        text(s, 0.5, 1.6, 9.0, 1.0, "Ablation results pending (scripts/run_phase5.sh).", size=14)
        return
    cd = CategoryChartData()
    ab = ab.iloc[::-1]
    cd.categories = list(ab.variant)
    cd.add_series("MAE at 60 min", [round(v, 3) for v in ab.MAE_60min])
    ch = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(0.4), Inches(1.4), Inches(6.4), Inches(3.55), cd).chart
    style_chart(ch, legend=False, size=10)
    ch.has_title = False
    ser = ch.series[0]
    ser.format.fill.solid()
    ser.format.fill.fore_color.theme_color = BLUE
    for i, v in enumerate(ab.variant):
        pt = ser.points[i]
        pt.format.fill.solid()
        pt.format.fill.fore_color.theme_color = AQUA if v.startswith("DCRNN") else ORANGE if "(main)" in v else BLUE
    pl = ch.plots[0]
    pl.gap_width = 60
    pl.has_data_labels = True
    pl.data_labels.number_format = "0.00"
    pl.data_labels.number_format_is_linked = False
    pl.data_labels.font.size = Pt(10)
    ch.value_axis.minimum_scale = round(float(ab.MAE_60min.min()) - 0.4, 1)
    abl = ab.set_index("run")
    msgs = []
    if {"stgat_identity_s0", "stgat_s0"} <= set(abl.index):
        dl = abl.loc["stgat_identity_s0"].MAE_60min - abl.loc["stgat_s0"].MAE_60min
        msgs.append(("No graph", f"{dl:+.2f} mph at 60 min: the neighbours are where the gain comes from"))
    if "stgat_random_s0" in abl.index:
        dl = abl.loc["stgat_random_s0"].MAE_60min - abl.loc["stgat_s0"].MAE_60min
        msgs.append(("Random graph", f"{dl:+.2f} mph: the real road links matter, not just extra inputs"))
    if "dcrnn_forward_s0" in abl.index:
        dl = abl.loc["dcrnn_forward_s0"].MAE_60min - abl.loc["dcrnn_s0"].MAE_60min
        msgs.append(("DCRNN, forward walk only", f"{dl:+.2f} mph: listening downstream matters for DCRNN too"))
    elif "dcrnn_k1_s0" in abl.index:
        dl = abl.loc["dcrnn_k1_s0"].MAE_60min - abl.loc["dcrnn_s0"].MAE_60min
        msgs.append(("DCRNN, 1 hop", f"{dl:+.2f} mph vs 2 hops"))
    for i, (h, b) in enumerate(msgs[:3]):
        y = 1.45 + i * 1.18
        card(s, 6.95, y, 2.55, 1.05, name=f"Ablation note {i}")
        text(s, 7.1, y + 0.08, 2.3, 0.95, [(h, {"bold": True, "size": 14}), (b, {"size": 12, "space": 2})])


def hybrid_slide(prs, d):
    hyb, f, sig = d["hyb"], d["final"], d["sig"]
    s = content(prs, "Hybrid: graph model + tree corrector",
                "DCRNN is the backbone; a small XGBoost learns where it is systematically wrong",
                "Why a hybrid: DCRNN is best overall, but after a congestion onset at 60 minutes ahead XGBoost's calendar "
                "knowledge is slightly better, and the two make different errors. Following stacked generalisation, a "
                "shallow XGBoost model learns DCRNN's residual from a few interpretable inputs. It must see forecasts "
                "DCRNN was not trained on, so it is fitted on the validation block and scored once on test. A version "
                "with all 46 features overfits; the compact, regularised one generalises.")
    flow = [("DCRNN forecast", "graph, every recurrent step", AQUA), ("+ XGBoost residual", "depth 3, >=200 per leaf", BLUE),
            ("Hybrid forecast", "fitted on validation only", RED)]
    for i, (t1, t2, c) in enumerate(flow):
        x = 0.5 + i * 3.05
        card(s, x, 1.45, 2.75, 0.95, name=f"Hybrid step {i}")
        badge(s, x + 0.15, 1.71, i + 1, fill=c, d=0.42)
        text(s, x + 0.7, 1.55, 1.95, 0.8, [(t1, {"bold": True, "size": 14}), (t2, {"size": 11, "space": 2})])
        if i < 2:
            shape(s, MSO_SHAPE.RIGHT_ARROW, x + 2.8, 1.8, 0.2, 0.25, GREY, name=f"Hybrid arrow {i}")
    if len(hyb):
        rows = [["Design (mean of DCRNN seeds)", "15 min", "30 min", "60 min"]]
        for _, r in hyb.iterrows():
            lab = (r.variant.replace("w_h * DCRNN + (1 - w_h) * XGBoost", "weighted average")
                   .replace("Residual corrector, ", "Corrector, ").replace(" (deep trees)", ""))
            rows.append([lab, f2(r.MAE_15min), f2(r.MAE_30min), f2(r.MAE_60min)])
        table(s, 0.5, 2.65, 5.6, rows, [3.2, 0.8, 0.8, 0.8], best_cols=(1, 2, 3), size=12)
    hs = sig[(sig.model == HYBRID) & (sig.baseline == "DCRNN (same seed)")]
    if len(hs) and HYBRID in f.index:
        g60 = hs[hs.minutes == 60]
        stat(s, 6.35, 2.65, 3.15, 1.0, f"-{g60.gain_pct.mean():.1f}%", "vs DCRNN at 60 min, same seed", num_size=22,
             name="Hybrid gain")
        stat(s, 6.35, 3.75, 3.15, 1.0, f"{int(hs.days_model_better.min())} / {int(hs.test_days.max())} days",
             "on which the hybrid beats DCRNN (worst horizon, worst seed)", num_size=20, name="Hybrid days")


def attention(prs, d):
    att = d["att"]
    s = content(prs, "What STGAT learned to look at", "Attention of the trained model on the test block, by situation",
                "We read the attention weights of the trained STGAT. Each sensor splits its attention between itself, "
                "its upstream neighbours and its downstream neighbours. When the sensor will slow down within 30 "
                "minutes, attention moves downstream: the model learned on its own that jams arrive from downstream, "
                "which is what Phase 1 measured directly in the data.")
    if att.empty:
        picture(s, FIG / "fig14_attention.png", 0.45, 1.42, w=6.35)
        return
    a = att.set_index(["situation", "attends_to"]).mean_share_pct
    kinds = ["self", "upstream neighbour", "downstream neighbour"]
    cd = CategoryChartData()
    cd.categories = ["itself", "upstream neighbours", "downstream neighbours"]
    cd.add_series("all other windows", [round(a[("all other windows", k)], 1) for k in kinds])
    cd.add_series("slowdown 30 min ahead", [round(a[("slowdown ahead (30 min)", k)], 1) for k in kinds])
    ch = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.4), Inches(1.4), Inches(6.3), Inches(3.55), cd).chart
    style_chart(ch)
    for ser, c in zip(ch.series, (GREY, ORANGE)):
        ser.format.fill.solid()
        ser.format.fill.fore_color.theme_color = c
    pl = ch.plots[0]
    pl.gap_width = 60
    pl.overlap = -10
    pl.has_data_labels = True
    pl.data_labels.number_format = '0"%"'
    pl.data_labels.number_format_is_linked = False
    ch.value_axis.has_title = True
    ch.value_axis.axis_title.text_frame.text = "share of a sensor's attention (%)"
    ch.value_axis.axis_title.text_frame.paragraphs[0].runs[0].font.size = Pt(11)
    down_o, down_s = a[("all other windows", "downstream neighbour")], a[("slowdown ahead (30 min)", "downstream neighbour")]
    stat(s, 6.9, 1.45, 2.6, 1.32, f"{down_o:.0f}% to {down_s:.0f}%", "attention to downstream neighbours when a "
         "slowdown is 30 min away", num_size=22, name="Attention")
    shifted = d["p5"].get("sensors_shifting_downstream")
    n = int(att.sensors.iloc[0])
    label = f"{shifted} of {n} sensors" if shifted else f"{n} sensors"
    card(s, 6.9, 2.92, 2.6, 1.83, name="Attention note")
    text(s, 7.05, 3.02, 2.3, 1.7, [("Learned, not told", {"bold": True, "size": 14}),
                                    (f"{label} shift their attention downstream when a jam is coming. No rule in the "
                                     "code says 'look downstream'.", {"size": 12, "space": 3})])


def where(prs, d):
    deg, ons = d["deg"], d["ons"]
    best = d["p6"]["best_graph_model"]
    s = content(prs, "Where the graph helps most", f"{best} vs XGBoost: by sensor connectivity and right after congestion starts",
                "Left: the more road links a sensor has, the bigger the gain, which is what you expect if the gain "
                "comes from neighbours. Right: in the 30 minutes after a congestion onset, the graph model reacts much "
                "faster at short horizons. At 60 minutes ahead nobody can see a jam that has not started yet, and "
                "the advantage disappears.")
    p = deg.pivot(index="degree", columns="model", values="mae_60min").loc[["0-4", "5-8", "9-12", "13+"]]
    n = deg.drop_duplicates("degree").set_index("degree").sensors
    gains = pct(p["XGBoost per horizon"], p[best])
    cd = CategoryChartData()
    cd.categories = [f"{k} links ({n[k]})" for k in gains.index]
    cd.add_series("MAE reduction at 60 min (%)", [round(v, 1) for v in gains.values])
    ch = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.4), Inches(1.45), Inches(4.5), Inches(3.4), cd).chart
    style_chart(ch, legend=False, size=10)
    ch.has_title = True
    ch.chart_title.text_frame.text = "60-min MAE reduction vs XGBoost (%)"
    ch.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(12)
    ser = ch.series[0]
    ser.format.fill.solid()
    ser.format.fill.fore_color.theme_color = AQUA if best == "DCRNN" else ORANGE
    pl = ch.plots[0]
    pl.gap_width = 70
    pl.has_data_labels = True
    pl.data_labels.number_format = '0.0"%"'
    pl.data_labels.number_format_is_linked = False
    o = ons[ons.group == "onset"].pivot(index="model", columns="minutes", values="mae")
    rows = [["MAE after an onset", "15 min", "30 min", "60 min"]]
    for m in ("XGBoost per horizon", "XGBoost + neighbours", "STGAT (ours)", "DCRNN"):
        if m in o.index:
            rows.append([m.replace(" per horizon", ""), f"{o.loc[m, 15]:.1f}", f"{o.loc[m, 30]:.1f}", f"{o.loc[m, 60]:.1f}"])
    table(s, 5.15, 1.5, 4.35, rows, [1.95, 0.8, 0.8, 0.8], best_cols=(1, 2, 3), size=12)
    x15 = o.loc["XGBoost per horizon", 15]
    b15 = o.loc[best, 15]
    text(s, 5.15, 3.3, 4.35, 1.5, [(f"-{pct(x15, b15):.0f}% error at 15 min after an onset", {"bold": True, "size": 14}),
                                  ("The graph model sees the slowdown at a neighbour first. At 60 min ahead the jam "
                                   "has usually not started anywhere yet, so no model can see it.", {"size": 12, "space": 3})])


def maps(prs, d):
    p7 = d["p7"]["side_by_side"]
    best = d["p6"]["best_graph_model"]
    s = content(prs, "Congestion maps", f"{pd.Timestamp(p7['target_time']):%a %d %b %Y, %H:%M} (test block); "
                f"forecasts made {p7['horizon_min']} min earlier",
                "The moment is chosen by a fixed rule: the weekday PM-peak test time with the most congested sensors, "
                "not the moment where our model looks best. Congestion level is speed divided by the sensor's own "
                "free-flow speed: below 0.5 congested, 0.5 to 0.8 slow.")
    pic = picture(s, FIG / "fig18_congestion_map_side_by_side.png", 0.3, 1.4, h=2.85)
    pic.left = int((prs.slide_width - pic.width) / 2)
    xg, gm = p7["XGBoost"], p7.get(best.replace(" (ours)", ""), p7.get("DCRNN", p7.get("STGAT")))
    em = d["p7"]["error_map"]
    text(s, 0.5, 4.38, 9.0, 0.55, f"XGBoost caught {xg['congested_caught']} of {p7['actual_congested']} congested sensors, "
         f"{best} caught {gm['congested_caught']}. Over the whole test block {best} has the lower 60-min error at "
         f"{em['sensors_graph_better']} of {em['sensors']} sensors.", size=13)


def time_space(prs, d):
    best = d["p6"]["best_graph_model"]
    s = content(prs, "A rush hour along one freeway", "15 consecutive sensors, one test evening; forecasts made 30 min earlier",
                "Each row is a sensor along the corridor, each column a 5-minute step. The actual jam starts at the "
                "downstream end around 15:15. The graph model's 30-minute forecast picks it up almost immediately and "
                "keeps its shape; XGBoost lags and smears it, because it only sees each sensor's own past.")
    picture(s, FIG / "fig20_time_space_diagram.png", 0.4, 1.4, h=3.5)
    card(s, 6.35, 1.45, 3.15, 3.3, name="Time-space note")
    text(s, 6.5, 1.55, 2.85, 3.1, [
        ("How to read it", {"bold": True, "size": 14}),
        ("Rows: sensors along the freeway in the driving direction. Columns: 5-minute steps. Dark red: slow. "
         "Grey: sensor down.", {"size": 12, "space": 3}),
        ("What it shows", {"bold": True, "size": 14, "space": 10}),
        (f"The jam starts at the downstream end. {best} picks it up from its neighbours and keeps its shape; "
         "XGBoost reacts later and smears it.", {"size": 12, "space": 3}),
    ])
    text(s, 0.4, 4.95, 6.0, 0.25, f"Panels, top to bottom: actual, XGBoost, {best}.", size=11, color=GREY)


def conclusion(prs, d):
    f = d["final"]
    best = d["p6"]["best_graph_model"]
    b, xg = f.loc[best], f.loc["XGBoost per horizon"]
    st = f.loc["STGAT (ours)"] if "STGAT (ours)" in f.index else None
    s = content(prs, "Conclusion", "The graph closes a real part of the gap, and we can show where and why",
                "Summarise the three findings, then the limitations. Signal control is not part of these results: "
                "METR-LA freeways have no traffic lights, so any control experiment would be a SUMO simulation of a "
                "synthetic network.", layout="DARK")
    pts = [f"{best} beats the best per-sensor model at every horizon: {f2(b.MAE_15min)} | {f2(b.MAE_30min)} | "
           f"{f2(b.MAE_60min)} mph vs {f2(xg.MAE_15min)} | {f2(xg.MAE_30min)} | {f2(xg.MAE_60min)}, better on every test day.",
           "The gain grows with connectivity and is largest right after congestion starts: it comes from the graph.",
           ("STGAT also beats XGBoost and learned to look downstream; mixing the graph into every recurrent step "
            "(DCRNN) works better." if st is not None else ""),
           (f"A regularised tree corrector on top of DCRNN (the hybrid) cuts error further: {f2(f.loc[HYBRID].MAE_60min)} mph "
            "at 60 min." if HYBRID in f.index else "")]
    for i, t in enumerate([p for p in pts if p]):
        y = 1.5 + i * 0.84
        badge(s, 0.5, y + 0.05, i + 1, fill=ORANGE)
        text(s, 1.1, y, 3.95, 0.82, t, size=12, color=WHITE)
    card(s, 5.3, 1.5, 4.2, 3.3, fill=T.TEXT_2, name="Limits panel")
    text(s, 5.5, 1.62, 3.8, 3.1, [
        ("Limits", {"bold": True, "size": 17, "head": True, "color": WHITE}),
        ("Speed only: no flow or occupancy (PeMS needed)", {"size": 12, "space": 4, "color": WHITE}),
        ("2012 freeway data from one city", {"size": 12, "space": 4, "color": WHITE}),
        ("Ablations use one seed", {"size": 12, "space": 4, "color": WHITE}),
        ("Next", {"bold": True, "size": 17, "head": True, "color": WHITE, "space": 10}),
        ("Learned adjacency (Graph WaveNet); PeMS flow; optional SUMO signal-control simulation "
         "(simulation only, never live control)", {"size": 12, "space": 4, "color": WHITE}),
    ])


def references(prs, d):
    s = content(prs, "References", "Full bibliography with DOIs and arXiv identifiers: docs/paper/references.bib",
                "The two papers the brief names are Velickovic et al. (graph attention) and Zheng et al. (CityFlow). "
                "DCRNN supplies the data, the graph, the protocol and the model we reproduce.")
    refs = ["[1] Li, Yu, Shahabi, Liu. Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic "
            "Forecasting. ICLR 2018. arXiv:1707.01926",
            "[2] Velickovic, Cucurull, Casanova, Romero, Lio, Bengio. Graph Attention Networks. ICLR 2018. arXiv:1710.10903",
            "[3] Brody, Alon, Yahav. How Attentive are Graph Attention Networks? ICLR 2022. arXiv:2105.14491",
            "[4] Wu, Pan, Long, Jiang, Zhang. Graph WaveNet for Deep Spatial-Temporal Graph Modeling. IJCAI 2019.",
            "[5] Zhang et al. CityFlow: A Multi-Agent Reinforcement Learning Environment for Large Scale City Traffic "
            "Scenario. WWW 2019.",
            "[6] Cho et al. Learning Phrase Representations using RNN Encoder-Decoder. EMNLP 2014 (GRU).",
            "[7] Bengio, Vinyals, Jaitly, Shazeer. Scheduled Sampling for Sequence Prediction with RNNs. NeurIPS 2015.",
            "[8] Diebold, Mariano. Comparing Predictive Accuracy. J. Business & Economic Statistics 13(3), 1995.",
            "[9] Harvey, Leybourne, Newbold. Testing the Equality of Prediction Mean Squared Errors. IJF 13(2), 1997.",
            "[10] Wolpert. Stacked Generalization. Neural Networks 5(2), 1992."]
    text(s, 0.5, 1.5, 9.0, 3.4, [(r, {"size": 12, "space": 5}) for r in refs])


def main() -> None:
    d = load()
    prs = Presentation(str(TEMPLATE))
    ids = prs.slides._sldIdLst
    for sid in list(ids):                                     # keep the template, drop the AML slides
        prs.part.drop_rel(sid.rId)
        ids.remove(sid)
    for lay in prs.slide_layouts:                             # footer text lives on the layouts
        for sh in lay.shapes:
            if sh.has_text_frame and "AML phase" in sh.text_frame.text:
                sh.text_frame.paragraphs[0].runs[0].text = FOOTER
    for build in (title_slide, recap, models, direction, protocol, headline_chart, results_table, significance,
                  ablations, hybrid_slide, attention, where, maps, time_space, conclusion, references):
        build(prs, d)
    prs.core_properties.title = "Learning the road network: graph models for traffic forecasting (DL phase)"
    prs.core_properties.author = "Team T036"
    prs.core_properties.subject = "Traffic Flow Prediction and Signal Control, METR-LA"
    prs.save(OUT)
    print(f"saved {OUT.relative_to(ROOT)} ({len(prs.slides)} slides)")


if __name__ == "__main__":
    main()
