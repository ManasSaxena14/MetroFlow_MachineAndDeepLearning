"""One look for the whole dashboard: colours, page styling and chart defaults.

The scene is a freeway at night: asphalt black, headlight white, tail-light red. On maps only trouble is lit. Free flow
stays dim, slow traffic is amber and congestion glows red. Type is Mona Sans throughout, set wide for titles and figures.
"""
from __future__ import annotations

from html import escape
from urllib.parse import quote

import plotly.graph_objects as go
import streamlit as st

INK, INK_2, MUTED = "#f2f0eb", "#b9bebd", "#8b9294"
PAGE, PANEL, RAISED, LINE, GRID = "#0c0e0f", "#15181a", "#1c2023", "#2a2f32", "#1d2123"
QUIET, QUIET_2, DIM = "#6f787b", "#4c5457", "#343a3d"                     # references and missing things
RED, AMBER = "#ff5b4d", "#f0a22e"

# colour follows the model on every page; checked for colour-blind separation on this background
MODEL = {"Hybrid": "#1fae7e", "DCRNN": "#4a90e8", "XGBoost": "#e0702f", "STGAT": "#9a7ae6"}
GREYS = {"Persistence": "#5d6669", "Historical average": "#6b7477", "Ridge": "#586063", "Random Forest": "#7a8386", "LSTM": "#8b9497"}
STATE = {"congested": RED, "slow": AMBER, "free": "#7d878a", "no reading": "#3a4144"}
CHANCE = [[0, "#394043"], [0.35, "#8a4a35"], [0.7, "#e8543f"], [1, "#ff8f7d"]]
FONT = "'Mona Sans', 'Helvetica Neue', Arial, sans-serif"


def colour(name: str) -> str:
    for key, c in {**MODEL, **GREYS}.items():
        if name.startswith(key):
            return c
    return MUTED


def alpha(hex_colour: str, a: float) -> str:
    r, g, b = (int(hex_colour[k:k + 2], 16) for k in (1, 3, 5))
    return f"rgba({r},{g},{b},{a})"


MARK = quote(f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke-linecap='round' stroke-width='2'>"
             f"<path d='M5.5 21 10 3M18.5 21 14 3' stroke='{INK}'/><path d='M12 20.5v-4.2M12 12.6V9.8M12 6.6V5' stroke='{MODEL['Hybrid']}'/></svg>")

CSS = f"""
<style>
:root {{ accent-color: {MODEL['Hybrid']}; color-scheme: dark; }}
.stApp, .stApp * {{ font-variant-numeric: normal; }}
.stApp {{ -webkit-font-smoothing: antialiased; }}
::selection {{ background: {alpha(MODEL['Hybrid'], .38)}; color: #fff; }}
* {{ scrollbar-color: {LINE} transparent; }}
*:focus-visible {{ outline: 2px solid {MODEL['Hybrid']} !important; outline-offset: 2px; }}

/* header */
[data-testid="stHeader"] {{ background: {PAGE}; height: 3.75rem; border-bottom: 1px solid {LINE}; }}
[data-testid="stToolbar"]::before {{ content: "MetroFlow"; color: {INK}; font: 600 1.12rem/1.6rem {FONT}; font-stretch: 112%; letter-spacing: -.012em;
  white-space: nowrap; padding: 0 1.8rem 0 2.15rem; margin-left: 1.4rem; align-self: center; position: relative; top: 2px;
  background: url("data:image/svg+xml,{MARK}") left center / 1.55rem no-repeat; }}
[data-testid="stTopNavLink"] {{ background: transparent !important; border-radius: 0; position: relative; transition: color .18s ease-out; }}
[data-testid="stTopNavLink"] p {{ color: {MUTED}; font-weight: 500; font-size: .95rem; transition: color .18s ease-out; }}
[data-testid="stTopNavLink"]:hover p, [data-testid="stTopNavLink"][aria-current="page"] p {{ color: {INK}; }}
[data-testid="stTopNavLink"][aria-current="page"] {{ background: linear-gradient({MODEL['Hybrid']}, {MODEL['Hybrid']}) center bottom / calc(100% - 1rem) 2px no-repeat !important; }}
[data-testid="stTopNavSection"] {{ color: {INK_2}; }}
[data-testid="stDecoration"], footer {{ display: none; }}
[data-testid="stMainBlockContainer"] {{ max-width: 1360px; padding-top: 6.4rem; padding-bottom: 5rem; }}

/* type */
h1, h2, h3 {{ text-wrap: balance; color: {INK}; }}
h1 {{ font-stretch: 112%; letter-spacing: -0.026em; line-height: 1.08; padding: 0 0 .55rem; max-width: 30ch; }}
h2 {{ font-stretch: 108%; letter-spacing: -0.018em; padding: 2.6rem 0 .2rem; line-height: 1.2; }}
h3 {{ letter-spacing: -0.01em; padding: 1.9rem 0 .25rem; line-height: 1.3; }}
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li {{ max-width: 70ch; line-height: 1.6; }}
[data-testid="stMarkdownContainer"] li {{ color: {INK_2}; }}
[data-testid="stMarkdownContainer"] li strong {{ color: {INK}; font-weight: 600; }}
.lede {{ font-size: 1.16rem; color: {INK_2}; max-width: 66ch; line-height: 1.55; margin: 0 0 1.7rem; }}
.lede b {{ color: {INK}; font-weight: 600; }}
.note {{ color: {MUTED}; font-size: .92rem; max-width: 68ch; line-height: 1.55; margin: .3rem 0 1rem; }}

/* readout: label left, figure right, hairlines between */
.readout {{ background: {PANEL}; border: 1px solid {LINE}; border-radius: 8px; padding: 1.05rem 1.25rem .55rem; margin: 0 0 1.2rem; }}
.readout .top {{ color: {INK_2}; font-weight: 500; font-size: .95rem; padding-bottom: .75rem; }}
.readout .row {{ display: flex; justify-content: space-between; align-items: baseline; gap: 1rem; border-top: 1px solid {LINE}; padding: .5rem 0 .45rem; }}
.readout .row span:first-child {{ color: {INK_2}; font-size: .96rem; }}
.readout .row span:last-child {{ font-size: 1.75rem; font-weight: 500; font-stretch: 114%; letter-spacing: -.02em; line-height: 1.15; white-space: nowrap; }}
.readout .hot span:last-child {{ color: {RED}; }}

.key {{ display: flex; flex-wrap: wrap; gap: .4rem 1.25rem; font-size: .9rem; color: {INK_2}; margin: .1rem 0 .55rem; }}
.key i {{ display: inline-block; border-radius: 50%; margin-right: .45rem; vertical-align: -1px; }}

/* widgets */
[data-testid="stWidgetLabel"] p {{ color: {MUTED}; font-size: .88rem; }}
[data-testid="stTab"] p, button[data-baseweb="tab"] p {{ font-size: 1rem; font-weight: 500; }}
[data-testid="stExpander"] details {{ border-color: {LINE}; background: {PAGE}; }}
[data-testid="stExpander"] summary {{ font-weight: 500; }}
[data-testid="stAlert"] > div {{ background: {PANEL} !important; border: 1px solid {LINE}; }}
[data-testid="stAlert"] p {{ color: {INK_2} !important; }}
[data-testid="stAlert"] code {{ color: {INK}; background: {RAISED}; }}
.st-key-mapbox [data-testid="stPlotlyChart"] {{ border: 1px solid {LINE}; border-radius: 8px; overflow: hidden; }}

@media (max-width: 1000px) {{
  [data-testid="stHorizontalBlock"] {{ flex-wrap: wrap; }}
  [data-testid="stColumn"] {{ min-width: min(100%, 22rem); }}
  .readout .row span:last-child {{ font-size: 1.45rem; }}
}}
@media (max-width: 640px) {{
  [data-testid="stMainBlockContainer"] {{ padding-left: 1.1rem; padding-right: 1.1rem; padding-top: 5.2rem; }}
  [data-testid="stToolbar"]::before {{ margin-left: .5rem; padding-right: .8rem; }}
  h1 {{ font-size: 1.95rem !important; }}
  .lede {{ font-size: 1.04rem; }}
}}
@media (prefers-reduced-motion: reduce) {{ * {{ transition: none !important; animation: none !important; }} }}
</style>
"""


def apply_page() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def lede(text: str) -> None:
    st.markdown(f'<p class="lede">{text}</p>', unsafe_allow_html=True)


def note(text: str) -> None:
    st.markdown(f'<p class="note">{text}</p>', unsafe_allow_html=True)


def readout(top: str, rows: list[tuple[str, str]], hot: int | None = None) -> None:
    """Panel of label and figure rows; `hot` is the index of the one row shown in the congestion colour."""
    body = "".join(f'<div class="row{" hot" if k == hot else ""}"><span>{escape(a)}</span><span>{escape(b)}</span></div>' for k, (a, b) in enumerate(rows))
    st.markdown(f'<div class="readout"><div class="top">{escape(top)}</div>{body}</div>', unsafe_allow_html=True)


def key(items: list[tuple[str, str, int]]) -> None:
    """Legend: (label, colour, dot size in px)."""
    st.markdown('<div class="key">' + "".join(
        f'<span><i style="background:{c};width:{s}px;height:{s}px"></i>{escape(t)}</span>' for t, c, s in items) + "</div>",
        unsafe_allow_html=True)


def style(fig: go.Figure, height: int = 380, x: str = "", y: str = "", legend: bool = True, **kw) -> go.Figure:
    axis = dict(gridcolor=GRID, linecolor=LINE, zeroline=False, tickfont=dict(color=MUTED, size=13), title_font=dict(color=INK_2, size=13),
                automargin=True, fixedrange=True)
    fig.update_layout(
        template="none", height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=INK, size=14), margin=dict(l=8, r=12, t=36 if legend else 12, b=8), showlegend=legend,
        legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, font=dict(color=INK_2, size=13)),
        hoverlabel=dict(bgcolor=RAISED, bordercolor=LINE, font=dict(family=FONT, color=INK, size=13)), bargap=0.34, bargroupgap=0.12)
    fig.update_xaxes(**axis, title_text=x, showgrid=False)
    fig.update_yaxes(**axis, title_text=y, showgrid=True)
    fig.update_layout(**kw)
    return fig


def show(fig: go.Figure, key: str, **kw):
    return st.plotly_chart(fig, width="stretch", theme=None, key=key, config={"displayModeBar": False, "responsive": True}, **kw)


def hbar(names: list[str], values: list[float], colours: list[str], fmt: str = "{:.2f}", unit: str = "", height: int | None = None,
         lo: list[float] | None = None, hi: list[float] | None = None, hover: str = "") -> go.Figure:
    """Ranked horizontal bars, the first item on top, each labelled with its own value."""
    err = dict(type="data", symmetric=False, array=[h - v for h, v in zip(hi, values)], arrayminus=[v - a for v, a in zip(values, lo)],
               color=INK, thickness=1.4, width=4) if lo and hi else None
    fig = go.Figure(go.Bar(x=values, y=names, orientation="h", marker=dict(color=colours, cornerradius=3), error_x=err,
                           hovertemplate="%{y}: %{x:.2f}" + unit + hover + "<extra></extra>"))
    top = max(hi) if hi else max(values)
    for n, v, e in zip(names, values, hi or values):
        fig.add_annotation(x=max(v, e), y=n, text=fmt.format(v) + unit, showarrow=False, xanchor="left", xshift=7, font=dict(color=INK, size=14))
    style(fig, height or 46 * len(names) + 40, legend=False)
    fig.update_yaxes(autorange="reversed", showgrid=False, tickfont=dict(color=INK, size=14))
    fig.update_xaxes(range=[0, top * 1.24], showticklabels=False)
    return fig
