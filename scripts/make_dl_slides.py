"""Build the DL-phase slide deck from the result files, on the same template as the AML deck.

    python scripts/make_dl_slides.py        # -> docs/presentation/DL_Graph_Models.pptx

Design rules: one message per slide, the title states that message, at most three supporting numbers, figures large,
tables short. Every number is read from results/, so rerunning after new results updates the deck. The theme, fonts
and layouts are taken from docs/presentation/AML_Traffic_Flow_Forecasting.pptx so both decks match. Every slide fades
in, and speaker notes carry the talking points.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
from lxml import etree
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_MARKER_STYLE
from pptx.enum.dml import MSO_THEME_COLOR
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "docs" / "figures"
TEMPLATE = ROOT / "docs" / "presentation" / "AML_Traffic_Flow_Forecasting.pptx"
OUT = ROOT / "docs" / "presentation" / "DL_Graph_Models.pptx"
FOOTER = "DL phase  |  Graph models for traffic forecasting on METR-LA"
HYBRID = "Hybrid: DCRNN + XGBoost (ours)"
XGB = "XGBoost per horizon"

T = MSO_THEME_COLOR
INK, WHITE, TINT, GREY = T.TEXT_1, T.BACKGROUND_1, T.BACKGROUND_2, T.ACCENT_6
BLUE, ORANGE, AQUA, YELLOW, RED = T.ACCENT_1, T.ACCENT_2, T.ACCENT_3, T.ACCENT_4, T.ACCENT_5
GRID = RGBColor(0xDD, 0xE3, 0xEA)
MODEL_COLOR = {"LSTM (no graph)": YELLOW, XGB: BLUE, "STGAT (ours)": ORANGE, "DCRNN": AQUA, HYBRID: RED}
SHORT = {XGB: "XGBoost", "STGAT (ours)": "STGAT", HYBRID: "Hybrid", "LSTM (no graph)": "LSTM (no graph)", "DCRNN": "DCRNN",
         "XGBoost + neighbours": "XGBoost + neighbours"}


# --------------------------------------------------------------------------- data
def load() -> dict:
    def j(name):
        return json.loads((RES / name).read_text())

    return {"final": pd.read_csv(RES / "final.csv").set_index("model"),
            "long": pd.read_csv(RES / "final_by_horizon.csv"),
            "sig": pd.read_csv(RES / "significance.csv"),
            "deg": pd.read_csv(RES / "gain_by_degree.csv"),
            "ons": pd.read_csv(RES / "gain_at_onset.csv"),
            "abl": pd.read_csv(RES / "ablations.csv"),
            "hyb": pd.read_csv(RES / "hybrid_ablation.csv"),
            "att": pd.read_csv(RES / "attention_shares.csv"),
            "p1": j("phase1_findings.json"), "p4": j("phase4_findings.json"), "p5": j("phase5_findings.json"),
            "p6": j("phase6_findings.json"), "p7": j("phase7_maps.json")}


def f2(x) -> str:
    return f"{x:.2f}"


def pct(a, b) -> float:
    """Relative reduction of b versus a, in percent."""
    return 100 * (a - b) / a


def trio(r, metric="MAE") -> str:
    return " | ".join(f2(r[f"{metric}_{m}min"]) for m in (15, 30, 60))


# --------------------------------------------------------------------------- drawing helpers
def _color(target, theme):
    (target.color if hasattr(target, "color") else target).theme_color = theme


def text(slide, x, y, w, h, value, size=14, bold=False, color=INK, head=False, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, name=None, italic=False):
    """Text box with zero padding. value: a string or a list of (text, overrides) paragraphs."""
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        box.name = name
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    for i, (t, o) in enumerate(value if isinstance(value, list) else [(value, {})]):
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
        s.line.width = Pt(1.5)
    if radius is not None:
        s.adjustments[0] = radius
    s.shadow.inherit = False
    s.text_frame.text = ""
    return s


def card(slide, x, y, w, h, fill=TINT, name=None, line=None):
    return shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h, fill, name, radius=0.06, line=line)


def badge(slide, x, y, label, fill=BLUE, d=0.42, size=15):
    shape(slide, MSO_SHAPE.OVAL, x, y, d, d, fill, name=f"Badge {label}")
    text(slide, x, y, d, d, str(label), size=size, bold=True, color=WHITE, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)


def pill(slide, x, y, w, label, fill=AQUA, h=0.3, size=12):
    shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h, fill, name=f"Pill {label}", radius=0.5)
    text(slide, x, y, w, h, label, size=size, bold=True, color=WHITE, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)


def arrow(slide, x1, y1, x2, y2, color=GREY, width=2.0):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.theme_color = color
    c.line.width = Pt(width)
    tail = etree.SubElement(c.line._get_or_add_ln(), qn("a:tailEnd"))
    tail.set("type", "triangle")
    tail.set("w", "med")
    tail.set("len", "med")
    return c


def signals(slide, y=0.5):
    for i, c in enumerate((RED, YELLOW, AQUA)):
        shape(slide, MSO_SHAPE.OVAL, 8.78 + 0.24 * i, y, 0.16, 0.16, c, name=f"Signal {i}")


def fade(slide) -> None:
    """A gentle fade into every slide, so the deck moves smoothly."""
    sld = slide._element
    for old in sld.findall(qn("p:transition")):
        sld.remove(old)
    tr = etree.Element(qn("p:transition"), spd="med")
    etree.SubElement(tr, qn("p:fade"))
    anchor = sld.find(qn("p:clrMapOvr"))
    (anchor if anchor is not None else sld.find(qn("p:cSld"))).addnext(tr)


def slide_number(slide, layout):
    """python-pptx does not copy the layout's slide-number placeholder; clone it so numbers show."""
    for sp in layout.placeholders:
        if "SLIDE_NUMBER" in str(sp.placeholder_format.type):
            slide.shapes._spTree.append(copy.deepcopy(sp._element))


def content(prs, title, subtitle, notes, layout="CONTENT"):
    lay = next(la for la in prs.slide_layouts if la.name == layout)
    s = prs.slides.add_slide(lay)
    s.placeholders[101].text = title
    if subtitle:
        text(s, 0.5, 0.98, 8.9, 0.32, subtitle, size=15, color=GREY if layout == "CONTENT" else TINT,
             anchor=MSO_ANCHOR.MIDDLE, name="Subtitle")
    signals(s)
    slide_number(s, lay)
    s.notes_slide.notes_text_frame.text = notes
    fade(s)
    return s


def stat(slide, x, y, w, h, number, label, num_size=26, name="Result"):
    card(slide, x, y, w, h, name=f"{name} card")
    text(slide, x + 0.16, y + 0.1, w - 0.32, 0.5, number, size=num_size, bold=True, head=True)
    top = 0.18 + num_size / 60
    text(slide, x + 0.16, y + top, w - 0.32, h - top - 0.08, label, size=12)


def takeaway(slide, value, y=4.42, h=0.52):
    """A dark banner with the one sentence the audience should remember."""
    shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, 0.5, y, 9.0, h, INK, name="Takeaway", radius=0.14)
    text(slide, 0.72, y, 8.56, h, value, size=14, bold=True, color=WHITE, anchor=MSO_ANCHOR.MIDDLE)


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
    va.major_gridlines.format.line.color.rgb = GRID
    va.format.line.fill.background()
    va.tick_labels.font.size = Pt(size)
    ca = chart.category_axis
    ca.tick_labels.font.size = Pt(size)
    ca.format.line.color.rgb = GRID


def color_series(chart, colors):
    for ser, c in zip(chart.series, colors):
        ser.format.fill.solid()
        ser.format.fill.fore_color.theme_color = c


def labels(plot, fmt="0.00", size=10):
    plot.has_data_labels = True
    plot.data_labels.number_format = fmt
    plot.data_labels.number_format_is_linked = False
    plot.data_labels.font.size = Pt(size)


def table(slide, x, y, w, rows, col_w, size=12, best_cols=(), row_h=0.32, highlight=()):
    """rows[0] = header. best_cols: columns whose lowest value is bolded. highlight: body rows drawn in bold on a tint."""
    shp = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(row_h * len(rows)))
    tb = shp.table
    for j, cw in enumerate(col_w):
        tb.columns[j].width = Inches(cw)
    best = {j: min(float(r[j]) for r in rows[1:]) for j in best_cols}
    for i, r in enumerate(rows):
        tb.rows[i].height = Inches(row_h)
        for j, v in enumerate(r):
            c = tb.cell(i, j)
            c.margin_left = c.margin_right = Inches(0.08)
            c.margin_top = c.margin_bottom = Inches(0.02)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            c.fill.solid()
            _color(c.fill.fore_color, INK if i == 0 else (TINT if (i % 2 == 0 or i in highlight) else WHITE))
            p = c.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.RIGHT
            p.font.size = Pt(size)
            run = p.add_run()
            run.text = str(v)
            run.font.size = Pt(size)
            _color(run.font, WHITE if i == 0 else (RED if (i in highlight and j == 0) else INK))
            run.font.bold = i == 0 or i in highlight and j == 0 or (j in best and abs(float(v) - best[j]) < 1e-9)
    return tb


# --------------------------------------------------------------------------- 1. opening
def title_slide(prs, d):
    f = d["final"]
    lay = next(la for la in prs.slide_layouts if la.name == "DEFAULT")
    s = prs.slides.add_slide(lay)
    fade(s)
    text(s, 0.6, 0.62, 5.2, 0.3, "DEEP LEARNING  |  PROJECT T036", size=12, bold=True, color=BLUE)
    text(s, 0.6, 1.05, 5.2, 1.5, "Learning the road network", size=38, bold=True, head=True)
    text(s, 0.6, 2.62, 4.9, 1.0, "Graph attention, diffusion and a hybrid model for traffic forecasting on METR-LA",
         size=16, color=GREY)
    text(s, 0.6, 4.2, 5.2, 0.7, [("Team T036", {"bold": True, "size": 14}),
                                ("Harsh Raj, Manas Saxena, Adil Mirza, Augustya Purohit, Sumit Yadav",
                                 {"size": 12, "space": 3, "color": GREY})])
    xs = [5.95, 6.8, 7.65, 8.5, 9.35]
    for i in range(4):
        c = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(xs[i] + 0.24), Inches(2.05), Inches(xs[i + 1]), Inches(2.05))
        c.line.color.theme_color = GREY
        c.line.width = Pt(2)
    for i, x in enumerate(xs):
        shape(s, MSO_SHAPE.OVAL, x, 1.93, 0.24, 0.24, RED if i == 4 else BLUE, name=f"Sensor {i}")
    shape(s, MSO_SHAPE.RIGHT_ARROW, 6.25, 2.5, 2.9, 0.2, GREY, name="Traffic arrow")
    text(s, 5.95, 2.75, 3.6, 0.3, "traffic flow", size=12, color=GREY, align=PP_ALIGN.CENTER)
    shape(s, MSO_SHAPE.LEFT_ARROW, 6.25, 3.15, 2.9, 0.2, ORANGE, name="Message arrow")
    text(s, 5.95, 3.4, 3.6, 0.5, "the model also listens downstream,\nwhere jams start", size=12, bold=True,
         color=ORANGE, align=PP_ALIGN.CENTER)
    hy, xg = f.loc[HYBRID], f.loc[XGB]
    pill(s, 6.25, 4.3, 2.9, f"{pct(xg.MAE_60min, hy.MAE_60min):.1f}% lower error at 60 min", fill=INK, h=0.36, size=12)
    signals(s, y=0.66)
    s.notes_slide.notes_text_frame.text = (
        "The AML phase showed that per-sensor models miss congestion that comes from neighbouring sensors. This phase "
        "builds two graph deep-learning models and a hybrid, scores them under exactly the same protocol, and checks "
        "where and why they help. Headline: our hybrid cuts the 60-minute error of the best per-sensor model by "
        f"{pct(xg.MAE_60min, hy.MAE_60min):.1f}%.")


def gap(prs, d):
    f, ons, p4, pr = d["final"], d["ons"], d["p4"], d["p1"]["propagation"]
    xg, xn = f.loc[XGB], f.loc["XGBoost + neighbours"]
    o = ons[ons.model == XGB].pivot(index="minutes", columns="group", values="mae")
    ratio = o.onset / o.normal
    s = content(prs, "Per-sensor models miss the neighbours", "What the AML phase measured: jams travel backwards along the road",
                "The brief's hybrid logic: per-sensor baselines first, a measured gap second, graph learning third. XGBoost "
                "is the strongest model that looks at one sensor at a time, but it is several times worse right after a "
                "jam starts. The figure shows why: the downstream sensor slows first and the jam reaches the upstream "
                "sensor about ten minutes later. Hand-made neighbour features help only a little.")
    picture(s, FIG / "fig05d_propagation_example.png", 0.45, 1.45, w=5.7)
    stat(s, 6.45, 1.45, 3.05, 0.92, trio(xg), "XGBoost MAE (mph), 15 / 30 / 60 min: the bar to beat", num_size=20,
         name="XGB")
    stat(s, 6.45, 2.47, 3.05, 0.92, f"{ratio.min():.1f} to {ratio.max():.1f}x",
         f"its error after a congestion onset ({p4['test_onsets']:,} onsets)", num_size=20, name="Onset")
    stat(s, 6.45, 3.49, 3.05, 0.92, f"{pr['downstream_first']} vs {pr['upstream_first']}",
         f"links where the downstream sensor slows first; jam moves back at {pr['shockwave_kmh']:.0f} km/h",
         num_size=20, name="First")
    takeaway(s, f"Neighbour features help by only {pct(xg.MAE_15min, xn.MAE_15min):.1f}%: can a model that reads the "
             "road graph close the gap?", y=4.5, h=0.5)


# --------------------------------------------------------------------------- 2. method
def models(prs, d):
    ab = d["abl"].set_index("run")
    s = content(prs, "Two graph models and a hybrid", "Same input (60 min, 207 sensors), same output (12 horizons), same road graph",
                "STGAT is our graph attention network: a GRU summarises each sensor's last hour, then two GATv2 layers let "
                "every sensor attend to its road neighbours over both directions of every link, because jams arrive from "
                "downstream. DCRNN is the reference diffusion model: the graph enters every GRU step. The hybrid keeps "
                "DCRNN and adds a small XGBoost per horizon that learns its residual; it is fitted on the validation block "
                "only, so no test information leaks.")
    cols = [("STGAT (ours)", "graph attention", ORANGE, f"{ab.loc['stgat_s0', 'parameters'] / 1000:.0f}k params", [
        ("Remember", "a shared GRU summarises each sensor's last hour"),
        ("Listen", "2 GATv2 layers over both directions of each link"),
        ("Predict", "12 horizons as the change from the last speed")]),
        ("DCRNN", "diffusion (Li et al., 2018)", AQUA, f"{ab.loc['dcrnn_s0', 'parameters'] / 1000:.0f}k params", [
        ("Diffuse", "forward and backward random walks, 2 hops"),
        ("Recur", "the graph enters every encoder and decoder step"),
        ("Decode", "12 horizons, with scheduled sampling")]),
        ("Hybrid (ours)", "graph model + trees", RED, "fit on val only", [
        ("Backbone", "the DCRNN forecast: the spatial view"),
        ("Correct", "XGBoost per horizon learns DCRNN's error"),
        ("Regularise", "depth 3, 200+ samples per leaf: no overfitting")])]
    for k, (name, kind, color, tag, steps) in enumerate(cols):
        x = 0.5 + k * 3.05
        card(s, x, 1.5, 2.85, 3.45, name=f"{name} card")
        shape(s, MSO_SHAPE.RECTANGLE, x + 0.2, 1.5, 2.45, 0.08, color, name=f"{name} bar")
        text(s, x + 0.2, 1.7, 2.45, 0.6, [(name, {"bold": True, "size": 16, "head": True}),
                                          (kind, {"size": 11, "color": GREY, "space": 1})])
        pill(s, x + 0.2, 2.35, 1.5, tag, fill=color, h=0.27, size=10)
        for i, (t1, t2) in enumerate(steps):
            yy = 2.8 + i * 0.7
            badge(s, x + 0.2, yy + 0.03, i + 1, fill=color, d=0.34, size=12)
            text(s, x + 0.66, yy, 2.05, 0.68, [(t1, {"bold": True, "size": 13}), (t2, {"size": 11, "space": 1})])


def protocol(prs, d):
    s = content(prs, "Same rules for every model", "The comparison is fair by construction, as the brief requires",
                "Every model reads the same preprocessed files, sees the same 60 minutes, is scored by the same function "
                "on the same masked test targets and is tuned on validation only. Graph models get 3 seeds each.")
    rules = [("Chronological split", "70 / 10 / 20 by time, purged, never random"),
             ("Equal history", "12 steps (60 min) in, 12 out, for every model"),
             ("Train-only statistics", "scaler and imputation fitted on training time only"),
             ("Outages masked", "0 mph is a broken sensor, never counted as traffic"),
             ("Same metrics", "MAE, RMSE, MAPE in mph, by horizon and peak / off-peak"),
             ("Fixed graph, 3 seeds", "METR-LA adjacency; mean of 3 runs per graph model")]
    for i, (h, b) in enumerate(rules):
        col, row = i % 2, i // 2
        x, y = 0.5 + col * 4.6, 1.5 + row * 1.05
        card(s, x, y, 4.4, 0.9, name=f"Rule {i}")
        badge(s, x + 0.18, y + 0.24, i + 1, d=0.42)
        text(s, x + 0.78, y + 0.12, 3.5, 0.7, [(h, {"bold": True, "size": 14}), (b, {"size": 12, "space": 2})])
    text(s, 0.5, 4.7, 9.0, 0.3, "Test block: the last 24 days, never used for tuning or model choice.", size=12, color=GREY)


# --------------------------------------------------------------------------- 3. results
def headline_chart(prs, d):
    f, long = d["final"], d["long"]
    dc, xg, hy = f.loc["DCRNN"], f.loc[XGB], f.loc[HYBRID]
    s = content(prs, "Graph models win at every horizon", "Test MAE (mph) from 5 to 60 minutes ahead; lower is better",
                "One line per model, all 12 horizons, on the test block. Graph models are the mean of 3 seeds. The gap "
                "to the per-sensor models opens at the first horizon and stays: the graph models see a slowdown "
                "building at a neighbour before it arrives.")
    cd = CategoryChartData()
    cd.categories = [str(5 * h) for h in range(1, 13)]
    names = ["LSTM (no graph)", XGB, "STGAT (ours)", "DCRNN", HYBRID]
    for m in names:
        g = long[(long.model == m) & (long.horizon != "avg")].copy()
        g["horizon"] = g.horizon.astype(int)
        cd.add_series(SHORT[m], [round(v, 3) for v in g.groupby("horizon").mae.mean().sort_index().values])
    ch = s.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, Inches(0.4), Inches(1.4), Inches(6.2), Inches(3.6), cd).chart
    style_chart(ch)
    for ser, m in zip(ch.series, names):
        c = MODEL_COLOR[m]
        ser.format.line.color.theme_color = c
        ser.format.line.width = Pt(3 if m == HYBRID else 2.25)
        ser.marker.style = XL_MARKER_STYLE.CIRCLE
        ser.marker.size = 5
        ser.marker.format.fill.solid()
        ser.marker.format.fill.fore_color.theme_color = c
        ser.marker.format.line.color.theme_color = c
        ser.smooth = False
    ch.value_axis.minimum_scale = 2.0
    ch.category_axis.has_title = True
    ch.category_axis.axis_title.text_frame.text = "minutes ahead"
    ch.category_axis.axis_title.text_frame.paragraphs[0].runs[0].font.size = Pt(11)
    stat(s, 6.8, 1.45, 2.7, 1.12, f"-{pct(xg.MAE_60min, dc.MAE_60min):.0f}%",
         f"DCRNN vs XGBoost at 60 min ({f2(dc.MAE_60min)} vs {f2(xg.MAE_60min)} mph)", name="DCRNN")
    stat(s, 6.8, 2.69, 2.7, 1.12, f2(hy.MAE_60min), "mph: the hybrid's 60-min error, the best of all models", name="Hybrid")
    stat(s, 6.8, 3.93, 2.7, 1.07, f"-{pct(xg.MAE_15min, f.loc['STGAT (ours)'].MAE_15min):.0f}%",
         "STGAT vs XGBoost at 15 min", name="STGAT")


def results_table(prs, d):
    f = d["final"]
    s = content(prs, "Results at a glance", "Test block, mph; graph models and hybrid: mean of 3 seeds; best in bold",
                "MAE is the headline. RMSE weighs rare large misses: at 60 minutes DCRNN's RMSE is above XGBoost's because "
                "its rarer misses, mostly when a jam starts, are larger; the hybrid brings it down. Peak means weekdays "
                "07:00 to 09:59 and 16:00 to 19:59, judged on the predicted time. Every model is weakest at peak hours, "
                "and that is where the graph models help most.")
    order = ["LSTM (no graph)", XGB, "XGBoost + neighbours", "STGAT (ours)", "DCRNN", HYBRID]
    rows = [["Model", "MAE 15", "MAE 30", "MAE 60", "RMSE 60", "Peak 60", "Off-peak 60"]]
    for m in order:
        r = f.loc[m]
        rows.append([SHORT[m], f2(r.MAE_15min), f2(r.MAE_30min), f2(r.MAE_60min), f2(r.RMSE_60min),
                     f2(r.peak_MAE_60min), f2(r.offpeak_MAE_60min)])
    table(s, 0.5, 1.5, 9.0, rows, [2.7, 1.0, 1.0, 1.0, 1.05, 1.05, 1.2], best_cols=(1, 2, 3, 4, 5, 6), size=13,
          row_h=0.36, highlight=(6,))
    xg, hy = f.loc[XGB], f.loc[HYBRID]
    text(s, 0.5, 4.08, 9.0, 0.3, "Top three: no graph (LSTM, XGBoost) and hand-made neighbour features. Bottom three: "
         "graph models. All ten models: results/final.csv.", size=11, color=GREY)
    takeaway(s, f"The hybrid is best on MAE at every horizon and cuts peak-hour error by "
             f"{pct(xg.peak_MAE_60min, hy.peak_MAE_60min):.1f}% against XGBoost.", y=4.48, h=0.48)


def significance(prs, d):
    sig, ref = d["sig"], d["p5"]["reference_check"]
    s = content(prs, "Reliable, and built on a faithful DCRNN", "Paired tests over the 24 test days (worst of 3 seeds); "
                "check against the published DCRNN",
                "Errors within a day are strongly correlated, so a day, not a single prediction, is the unit. We compare "
                "daily errors with a paired bootstrap (10,000 resamples) and a Diebold-Mariano test, for every seed, and "
                "report the worst seed. To check our DCRNN port we also trained it under the paper's own protocol: it "
                "lands within 1.5% of the published numbers.")

    def days(model, base, minutes=None):
        r = sig[(sig.model == model) & (sig.baseline == base)]
        if minutes:
            r = r[r.minutes == minutes]
        return int(r.days_model_better.min()), int(r.test_days.max()), max(float(r.p_bootstrap.max()), float(r.p_dm.max()))

    worst = max(abs(v["diff_pct"]) for v in ref.values())
    ours = " | ".join(f2(ref[f"{m}min"]["ours"]) for m in (15, 30, 60))
    pub = " | ".join(f2(ref[f"{m}min"]["published"]) for m in (15, 30, 60))
    cards = []
    for h, (b, n, p), sub, c in (("DCRNN vs XGBoost", days("DCRNN", XGB), "days better at every horizon", AQUA),
                                 ("Hybrid vs DCRNN", days(HYBRID, "DCRNN (same seed)"), "days better at every horizon", RED),
                                 ("STGAT vs XGBoost", days("STGAT (ours)", XGB, 60), "days better at 60 min: weak", ORANGE)):
        cards.append((h, f"{b} / {n}", sub, "p < 0.0001" if p < 1e-4 else f"p up to {p:.3f}", c))
    cards.append(("Published DCRNN", f"{worst:.1f}%", f"largest gap: ours {ours} vs published {pub} mph", "", BLUE))
    for i, (h, big, sub, ptxt, c) in enumerate(cards):
        x = 0.5 + i * 2.3
        card(s, x, 1.5, 2.15, 2.75, name=f"Sig {i}")
        shape(s, MSO_SHAPE.RECTANGLE, x + 0.18, 1.5, 1.79, 0.08, c, name=f"Sig bar {i}")
        text(s, x + 0.18, 1.72, 1.85, 0.35, h, size=13, bold=True)
        text(s, x + 0.18, 2.1, 1.85, 0.6, big, size=30, bold=True, head=True)
        text(s, x + 0.18, 2.85, 1.85, 1.35, [(sub, {"size": 11}), (ptxt, {"size": 12, "bold": True, "space": 4})])
    takeaway(s, "DCRNN beats XGBoost on every test day, and our port matches the published paper.", y=4.48, h=0.48)


# --------------------------------------------------------------------------- 4. why it works
def ablations(prs, d):
    ab = d["abl"].set_index("run")
    s = content(prs, "Ablations: the road graph is what helps", "How much STGAT's 60-min test MAE rises (mph) when one thing changes",
                "Each bar changes one component, same budget, seed 0. No graph keeps self-loops only: an equal-budget "
                "no-graph control, which falls back to the LSTM's level. A random graph with as many links is clearly "
                "worse, so the real road structure matters. Downstream-only beats upstream-only, as the propagation "
                "analysis predicts.")
    rows = [("stgat_identity_s0", "no graph"), ("stgat_random_s0", "random graph, same size"),
            ("stgat_1layer_s0", "1 attention layer, not 2"), ("stgat_forward_s0", "upstream messages only"),
            ("stgat_undirected_s0", "no direction feature"), ("stgat_backward_s0", "downstream messages only")]
    cd = CategoryChartData()
    cd.categories = [lab for _, lab in rows][::-1]
    cd.add_series("worse by (mph)", [round(ab.loc[r, "delta_60min_vs_main"], 3) for r, _ in rows][::-1])
    ch = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(0.4), Inches(1.4), Inches(5.9), Inches(3.6), cd).chart
    style_chart(ch, legend=False, size=11)
    ch.has_title = False
    color_series(ch, [ORANGE])
    ser = ch.series[0]
    for i, (r, _) in enumerate(rows[::-1]):
        if r in ("stgat_identity_s0", "stgat_random_s0"):
            ser.points[i].format.fill.solid()
            ser.points[i].format.fill.fore_color.theme_color = RED
    pl = ch.plots[0]
    pl.gap_width = 55
    labels(pl, "+0.00", 11)
    ch.value_axis.minimum_scale = 0
    notes = [("Graph, not size", f"no graph: +{ab.loc['stgat_identity_s0', 'delta_60min_vs_main']:.2f} mph; "
              f"random graph: +{ab.loc['stgat_random_s0', 'delta_60min_vs_main']:.2f} mph"),
             ("Downstream matters", f"DCRNN without its backward walk: +{ab.loc['dcrnn_forward_s0', 'delta_60min_vs_main']:.2f} mph"),
             ("No overfitting", f"dropout off: {ab.loc['stgat_drop0_s0', 'delta_60min_vs_main']:+.2f} mph, within noise")]
    for i, (h, b) in enumerate(notes):
        y = 1.45 + i * 1.2
        card(s, 6.55, y, 2.95, 1.08, name=f"Abl note {i}")
        text(s, 6.72, y + 0.1, 2.65, 0.9, [(h, {"bold": True, "size": 14}), (b, {"size": 12, "space": 3})])


def hybrid_results(prs, d):
    hyb, sig, f = d["hyb"], d["sig"], d["final"]
    s = content(prs, "The hybrid adds a further, reliable gain", "Hybrid designs on the same three DCRNN seeds: test MAE (mph)",
                "A per-horizon blend helps a little. A corrector on all 46 features with deep trees overfits the short "
                "fitting block and is worse than DCRNN alone at 30 and 60 minutes. The compact, regularised corrector "
                "improves DCRNN at every horizon, most at 60 minutes, where calendar knowledge matters most.")
    names = {"DCRNN alone": "DCRNN alone", "Blend": "Blend of DCRNN and XGBoost",
             "Residual corrector, all": "Corrector, all 46 features", "Residual corrector, compact": "Corrector, compact (ours)"}
    rows = [["Design", "15 min", "30 min", "60 min"]]
    for _, r in hyb.iterrows():
        lab = next(v for k, v in names.items() if r.variant.startswith(k))
        rows.append([lab, f"{r.MAE_15min:.3f}", f"{r.MAE_30min:.3f}", f"{r.MAE_60min:.3f}"])
    table(s, 0.5, 1.5, 5.7, rows, [2.85, 0.95, 0.95, 0.95], best_cols=(1, 2, 3), size=13, row_h=0.42, highlight=(4,))
    text(s, 0.5, 3.8, 5.7, 0.6, "Regularisation matters where data are scarce: the corrector only has the validation "
         "block to learn from.", size=12, color=GREY)
    hs = sig[(sig.model == HYBRID) & (sig.baseline == "DCRNN (same seed)")]
    stat(s, 6.5, 1.5, 3.0, 1.12, f"-{pct(f.loc['DCRNN'].MAE_60min, f.loc[HYBRID].MAE_60min):.1f}%",
         "vs DCRNN at 60 min, on top of the graph model's gain", name="HybGain")
    stat(s, 6.5, 2.74, 3.0, 1.12, f"{int(hs.days_model_better.min())} / {int(hs.test_days.max())} days",
         "on which the hybrid beats DCRNN (any horizon, any seed)", num_size=24, name="HybDays")
    stat(s, 6.5, 3.98, 3.0, 1.0, f"{f2(f.loc['DCRNN'].RMSE_60min)} to {f2(f.loc[HYBRID].RMSE_60min)}",
         "60-min RMSE: fewer large misses", num_size=22, name="HybRmse")


def where(prs, d):
    deg, ons = d["deg"], d["ons"]
    s = content(prs, "Where the graph helps most", "DCRNN vs XGBoost: by sensor connectivity, and right after a jam starts",
                "Left: the more road links a sensor has, the bigger the gain, as expected if the gain comes from "
                "neighbours. Right: in the 30 minutes after a congestion onset, the graph model reacts much faster at "
                "short horizons. At 60 minutes ahead the jam has usually not started anywhere yet, so the graph cannot "
                "see it; there the trees' calendar knowledge is slightly better, which the hybrid exploits.")
    p = deg.pivot(index="degree", columns="model", values="mae_60min").loc[["0-4", "5-8", "9-12", "13+"]]
    n = deg.drop_duplicates("degree").set_index("degree").sensors
    gains = pct(p[XGB], p["DCRNN"])
    cd = CategoryChartData()
    cd.categories = [f"{k} links ({n[k]})" for k in gains.index]
    cd.add_series("60-min error reduction (%)", [round(v, 1) for v in gains.values])
    ch = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.4), Inches(1.4), Inches(4.6), Inches(3.6), cd).chart
    style_chart(ch, legend=False, size=10)
    ch.has_title = True
    ch.chart_title.text_frame.text = "60-min error reduction vs XGBoost (%)"
    ch.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(12)
    color_series(ch, [AQUA])
    pl = ch.plots[0]
    pl.gap_width = 70
    labels(pl, '0.0"%"', 11)
    ch.value_axis.minimum_scale = 0
    o = ons[ons.group == "onset"].pivot(index="model", columns="minutes", values="mae")
    rows = [["After a jam starts", "15 min", "30 min", "60 min"]]
    for m in (XGB, "DCRNN", HYBRID):
        rows.append([SHORT[m], f"{o.loc[m, 15]:.1f}", f"{o.loc[m, 30]:.1f}", f"{o.loc[m, 60]:.1f}"])
    table(s, 5.3, 1.5, 4.2, rows, [1.8, 0.8, 0.8, 0.8], best_cols=(1, 2, 3), size=12, row_h=0.36)
    text(s, 5.3, 3.15, 4.2, 1.8, [(f"-{pct(o.loc[XGB, 15], o.loc['DCRNN', 15]):.0f}% error 15 min after a jam starts",
                                   {"bold": True, "size": 14}),
                                  ("DCRNN sees the slowdown at a neighbour first. At 60 min ahead no model can see a jam "
                                   "that has not started yet.", {"size": 12, "space": 3})])


def attention(prs, d):
    att, p5 = d["att"], d["p5"]
    s = content(prs, "STGAT learned to look downstream", "Share of a sensor's attention on the test block, by situation",
                "We read the attention weights of the trained STGAT. When a sensor will slow down within 30 minutes, its "
                "attention moves to downstream neighbours. No rule in the code says 'look downstream': the model learned "
                "the direction the data analysis found.")
    a = att.set_index(["situation", "attends_to"]).mean_share_pct
    kinds = ["self", "upstream neighbour", "downstream neighbour"]
    cd = CategoryChartData()
    cd.categories = ["itself", "upstream neighbours", "downstream neighbours"]
    cd.add_series("normal windows", [round(a[("all other windows", k)], 1) for k in kinds])
    cd.add_series("slowdown 30 min ahead", [round(a[("slowdown ahead (30 min)", k)], 1) for k in kinds])
    ch = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.4), Inches(1.4), Inches(6.2), Inches(3.6), cd).chart
    style_chart(ch)
    color_series(ch, [GREY, ORANGE])
    pl = ch.plots[0]
    pl.gap_width = 60
    pl.overlap = -10
    labels(pl, '0"%"', 11)
    ch.value_axis.minimum_scale = 0
    lo, hi = a[("all other windows", "downstream neighbour")], a[("slowdown ahead (30 min)", "downstream neighbour")]
    stat(s, 6.85, 1.45, 2.65, 1.3, f"{lo:.0f}% to {hi:.0f}%", "attention on downstream neighbours when a slowdown is "
         "coming", num_size=24, name="Att")
    stat(s, 6.85, 2.9, 2.65, 1.3, f"{p5['sensors_shifting_downstream']} of {int(att.sensors.iloc[0])}",
         "sensors shift their attention downstream", num_size=24, name="Shift")
    text(s, 6.85, 4.4, 2.65, 0.5, "Learned, not told.", size=15, bold=True, color=ORANGE)


# --------------------------------------------------------------------------- 5. maps
def maps(prs, d):
    p7, em = d["p7"]["side_by_side"], d["p7"]["error_map"]
    s = content(prs, "Who sees the rush hour coming?", f"Congestion levels, {pd.Timestamp(p7['target_time']):%a %d %b %Y, %H:%M}; "
                f"forecasts made {p7['horizon_min']} minutes earlier",
                "The moment is fixed by a rule: the weekday PM-peak test time with the most congested sensors, not the "
                "moment where our model looks best. Congestion level is speed divided by the sensor's own free-flow "
                f"speed. Over the whole test block DCRNN has the lower 60-minute error at {em['sensors_graph_better']} of "
                f"{em['sensors']} sensors, a median gain of {em['median_gain_mph']:.2f} mph per sensor (error map in the "
                "report, figure 19).")
    pic = picture(s, FIG / "fig18_congestion_map_side_by_side.png", 0.3, 1.4, h=2.9)
    pic.left = int((prs.slide_width - pic.width) / 2)
    xg, dc = p7["XGBoost"], p7["DCRNN"]
    takeaway(s, f"Of {p7['actual_congested']} congested sensors, XGBoost caught {xg['congested_caught']}, DCRNN "
             f"{dc['congested_caught']}; DCRNN is better at {em['sensors_graph_better']} of {em['sensors']} sensors.",
             y=4.45, h=0.5)


def time_space(prs, d):
    ts = d["p7"]["time_space"]
    s = content(prs, "A rush hour along one freeway", f"{len(ts['sensors'])} consecutive sensors over {ts['km']:.1f} km, "
                f"{pd.Timestamp(ts['day']):%a %d %b %Y}; forecasts made 30 minutes earlier",
                "Each row is a sensor along the corridor in the driving direction, each column five minutes. The jam "
                "starts at the downstream end and grows backwards. DCRNN's forecast picks up its start and backward "
                "spread; XGBoost's jam starts later and is weaker. DCRNN over-extends the jam on part of the corridor.")
    picture(s, FIG / "fig20_time_space_diagram.png", 0.4, 1.42, h=3.55)
    card(s, 6.75, 1.45, 2.75, 3.5, name="TS note")
    text(s, 6.9, 1.55, 2.45, 3.3, [
        ("How to read it", {"bold": True, "size": 14}),
        ("Panels: actual, XGBoost, DCRNN. Dark red = slow; grey = sensor down.", {"size": 12, "space": 3}),
        ("What it shows", {"bold": True, "size": 14, "space": 10}),
        ("The jam starts downstream and grows backwards. DCRNN catches its start and spread; XGBoost reacts later "
         "and weaker.", {"size": 12, "space": 3})])


# --------------------------------------------------------------------------- 6. outcomes
def hypotheses(prs, d):
    f, ab, sig = d["final"], d["abl"].set_index("run"), d["sig"]
    xg, dc = f.loc[XGB], f.loc["DCRNN"]
    hd = int(sig[(sig.model == HYBRID) & (sig.baseline == "DCRNN (same seed)")].days_model_better.min())
    s = content(prs, "Outcome: five hypotheses tested", "What we expected from the literature and the AML gap, and what we found",
                "H5 to H9 continue the AML phase's H1 to H4. Four are fully supported; H8 is partly supported because "
                "60 minutes before a jam starts no model can see it.")
    gains = [pct(xg[f"MAE_{m}min"], dc[f"MAE_{m}min"]) for m in (15, 30, 60)]
    rows = [("H5", "A graph model beats the best per-sensor model at every horizon",
             f"DCRNN: {min(gains):.0f}% to {max(gains):.0f}% lower MAE, better on all 24 test days", "Supported", AQUA),
            ("H6", "The gain comes from the real road graph",
             f"no graph +{ab.loc['stgat_identity_s0', 'delta_60min_vs_main']:.2f} mph, random graph "
             f"+{ab.loc['stgat_random_s0', 'delta_60min_vs_main']:.2f} mph at 60 min", "Supported", AQUA),
            ("H7", "Downstream information is worth more than upstream",
             "downstream-only beats upstream-only; attention shifts downstream", "Supported", AQUA),
            ("H8", "Gains are larger for connected sensors and after jams start",
             "true for connectivity and at 15-30 min; not at 60 min after an onset", "Partly", YELLOW),
            ("H9", "A leak-free hybrid improves on the best graph model",
             f"{pct(dc.MAE_60min, f.loc[HYBRID].MAE_60min):.1f}% lower MAE at 60 min, better on at least {hd} of 24 days",
             "Supported", AQUA)]
    for i, (h, claim, ev, tag, c) in enumerate(rows):
        y = 1.48 + i * 0.7
        card(s, 0.5, y, 9.0, 0.6, name=f"H {i}")
        badge(s, 0.62, y + 0.09, h, fill=INK, d=0.42, size=11)
        text(s, 1.2, y + 0.05, 6.5, 0.52, [(claim, {"bold": True, "size": 13}), (ev, {"size": 11, "space": 1, "color": GREY})])
        pill(s, 7.95, y + 0.15, 1.4, tag, fill=c, size=11)


def brief(prs, d):
    s = content(prs, "What the brief asked, what we delivered", "Both phases together",
                "Everything the brief requires is delivered. Flow and occupancy are not in METR-LA, which records speed "
                "only; they need PeMS data. The optional signal-control simulation was not run, since METR-LA freeways "
                "have no signals; nothing in this project claims live traffic control.")
    rows = [("Predict congestion on a road network", "207 sensors, 5 to 60 min ahead", "Done", AQUA),
            ("Baseline: XGBoost per segment or multi-output RF", "both, plus four more baselines (AML)", "Done", AQUA),
            ("Identify the gap", "four gap experiments (AML)", "Done", AQUA),
            ("GAT or diffusion model over a fixed graph", "both, STGAT and DCRNN, plus a hybrid", "Done", AQUA),
            ("Fair comparison: by horizon, equal history, peak", "one protocol, day-level significance tests", "Done", AQUA),
            ("Congestion maps, horizon-wise metrics", "maps, error map, time-space, animation", "Done", AQUA),
            ("Flow and occupancy features", "not in METR-LA (speed only)", "No data", YELLOW),
            ("Optional simulated signal policy (SUMO)", "not run: freeways have no signals", "Optional", GREY)]
    for i, (req, where_, tag, c) in enumerate(rows):
        y = 1.45 + i * 0.44
        if i % 2 == 0:
            shape(s, MSO_SHAPE.RECTANGLE, 0.5, y - 0.03, 9.0, 0.42, TINT, name=f"Row {i}")
        text(s, 0.65, y, 4.6, 0.36, req, size=12, bold=True, anchor=MSO_ANCHOR.MIDDLE)
        text(s, 5.3, y, 2.9, 0.36, where_, size=11, color=GREY, anchor=MSO_ANCHOR.MIDDLE)
        pill(s, 8.3, y + 0.05, 1.1, tag, fill=c, h=0.27, size=10)


def conclusion(prs, d):
    f = d["final"]
    dc, xg, hy = f.loc["DCRNN"], f.loc[XGB], f.loc[HYBRID]
    s = content(prs, "Conclusion", "The graph closes a real part of the gap, and we can show where and why",
                "Summarise the four findings, then the limits and next steps. Signal control is not part of these "
                "results: METR-LA freeways have no traffic lights, so a control study would be a SUMO or CityFlow "
                "simulation of a synthetic network, never live control.", layout="DARK")
    pts = [f"DCRNN beats the best per-sensor model at every horizon and on every test day: {trio(dc)} vs {trio(xg)} mph.",
           "The gain comes from the real road graph and from listening downstream; it is largest at well-connected "
           "sensors, at peak hours and right after a jam starts.",
           "STGAT also beats XGBoost, and its attention learned on its own to look downstream.",
           f"Our hybrid adds the trees' calendar knowledge to the graph: {trio(hy)} mph, the best model of the project."]
    for i, t in enumerate(pts):
        y = 1.5 + i * 0.86
        badge(s, 0.5, y + 0.06, i + 1, fill=ORANGE)
        text(s, 1.1, y, 4.0, 0.84, t, size=12, color=WHITE)
    card(s, 5.4, 1.5, 4.1, 3.4, fill=T.TEXT_2, name="Limits panel")
    text(s, 5.6, 1.62, 3.7, 3.2, [
        ("Limits", {"bold": True, "size": 16, "head": True, "color": WHITE}),
        ("Speed only: flow and occupancy need PeMS", {"size": 12, "space": 4, "color": WHITE}),
        ("2012 freeway data from one city", {"size": 12, "space": 3, "color": WHITE}),
        ("Ablations use one seed", {"size": 12, "space": 3, "color": WHITE}),
        ("Next", {"bold": True, "size": 16, "head": True, "color": WHITE, "space": 12}),
        ("PeMS with flow and occupancy; learned adjacency (Graph WaveNet); probabilistic forecasts; a "
         "simulation-only signal-control study", {"size": 12, "space": 4, "color": WHITE})])


def references(prs, d):
    s = content(prs, "References", "Full bibliography with DOIs and arXiv identifiers: docs/dl_report/references.bib",
                "The two pre-reads named in the brief are Velickovic et al. (graph attention) and CityFlow. DCRNN supplies "
                "the data, the graph, the protocol and the model we reproduce.")
    refs = ["Li, Yu, Shahabi, Liu. Diffusion Convolutional Recurrent Neural Network. ICLR 2018.",
            "Velickovic et al. Graph Attention Networks. ICLR 2018.",
            "Brody, Alon, Yahav. How Attentive are Graph Attention Networks? ICLR 2022.",
            "Yu, Yin, Zhu. Spatio-Temporal Graph Convolutional Networks. IJCAI 2018.",
            "Wu et al. Graph WaveNet for Deep Spatial-Temporal Graph Modeling. IJCAI 2019.",
            "Zheng et al. GMAN: A Graph Multi-Attention Network for Traffic Prediction. AAAI 2020.",
            "Zhang et al. CityFlow: A Multi-Agent RL Environment for City Traffic. WWW 2019.",
            "Chen, Guestrin. XGBoost: A Scalable Tree Boosting System. KDD 2016.",
            "Cho et al. Learning Phrase Representations using RNN Encoder-Decoder. EMNLP 2014.",
            "Bengio et al. Scheduled Sampling for Sequence Prediction with RNNs. NeurIPS 2015.",
            "Wolpert. Stacked Generalization. Neural Networks 5(2), 1992.",
            "Diebold, Mariano. Comparing Predictive Accuracy. JBES 13(3), 1995.",
            "Harvey, Leybourne, Newbold. Testing the Equality of Prediction MSE. IJF 13(2), 1997.",
            "Jagadish et al. Big Data and Its Technical Challenges. CACM 57(7), 2014."]
    half = (len(refs) + 1) // 2
    for k in range(2):
        chunk = refs[k * half:(k + 1) * half]
        text(s, 0.5 + k * 4.6, 1.5, 4.4, 3.5, [(f"[{k * half + i + 1}] {r}", {"size": 11, "space": 6})
                                                for i, r in enumerate(chunk)])


def team(prs, d):
    s = content(prs, "Team and resources", "The work is divided equally across both phases",
                "Each member owns one part of the machine-learning phase and one part of the deep-learning phase. "
                "Everything shown today is in the repository, with a report, a research paper and a live dashboard.")
    rows = [["Member", "DL phase"],
            ["Harsh Raj", "STGAT graph attention model, graph layers, attention inspection"],
            ["Manas Saxena", "fair comparison and day-level significance tests"],
            ["Adil Mirza", "DCRNN, ablation studies, published-number reproduction"],
            ["Augustya Purohit", "hybrid DCRNN + XGBoost model, IEEE research paper"],
            ["Sumit Yadav", "congestion maps, time-space diagram, DL slides"]]
    tb = table(s, 0.5, 1.5, 5.7, rows, [1.6, 4.1], size=12, row_h=0.42)
    for i in range(len(rows)):
        tb.cell(i, 1).text_frame.paragraphs[0].alignment = PP_ALIGN.LEFT
    card(s, 6.45, 1.5, 3.05, 2.52, name="Resources")
    text(s, 6.62, 1.62, 2.75, 2.3, [("Resources", {"bold": True, "size": 15}),
                                    ("Code and results: github.com/ManasSaxena14/\nTrafficFlowControl_AIML-DL_T036",
                                     {"size": 11, "space": 6}),
                                    ("DL report: docs/dl_report/dl_report.pdf", {"size": 11, "space": 6}),
                                    ("Dashboard: trafficflowamldl.streamlit.app", {"size": 11, "space": 6})])
    text(s, 0.5, 4.4, 9.0, 0.4, "Newton School of Technology, Rishihood University  |  Team T036", size=12, color=GREY)


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
    for build in (title_slide, gap, models, protocol, headline_chart, results_table, significance, ablations,
                  hybrid_results, where, attention, maps, time_space, hypotheses, brief, conclusion, references, team):
        build(prs, d)
    cp = prs.core_properties
    cp.title = "Learning the road network: graph models for traffic forecasting (DL phase)"
    cp.author = cp.last_modified_by = "Team T036"
    cp.subject = "Traffic Flow Prediction and Signal Control, METR-LA"
    cp.keywords = "traffic forecasting; graph neural networks; GATv2; DCRNN; METR-LA"
    prs.save(OUT)
    print(f"saved {OUT.relative_to(ROOT)} ({len(prs.slides)} slides)")


if __name__ == "__main__":
    main()
