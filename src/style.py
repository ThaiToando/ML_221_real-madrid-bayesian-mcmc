"""One look for every chart and the slide deck, picked by THEME below (or env var RM_THEME):
  "editorial" (default) sports-desk report: warm paper, near-black ink, serif headlines, gold only for Real Madrid
  "club"                match-analysis report: deep navy, metallic gold for Real Madrid, condensed sports type
  "terminal"            near-black panels, gold/amber headers, monospaced numbers, thin grid
  "paper"               white background, black ink: journal / research-paper figures
Real Madrid is always GOLD; rivals and "everyone else" are NAVY (light steel blue on the dark theme) and GREY.
Matplotlib uses setup() (called on import); Plotly uses plotly_template(); the deck uses DECK.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

THEME = os.environ.get("RM_THEME", "editorial")
SERIF = ["Source Serif 4", "Georgia", "Cambria", "serif"]
MONO = ["IBM Plex Mono", "JetBrains Mono", "Menlo", "Consolas", "DejaVu Sans Mono", "monospace"]
SANS = ["IBM Plex Sans", "Inter", "Helvetica Neue", "Arial", "DejaVu Sans", "sans-serif"]
SOURCE = "Data: Transfermarkt via transfermarkt-datasets; authors' calculations"

if THEME == "editorial":
    MONO = ["Times New Roman", "Times", "serif"]          # one classic typeface for everything
    SANS = MONO
    SERIF = MONO
    BG, PANEL, GRIDC = "#2b211b", "#352a22", "#4d3f33"          # warm dark brown, a lighter brown panel
    INK, INK2, MUTED, ACCENT = "#f5ede2", "#e0d3c1", "#b3a28c", "#ef5350"   # cream text, red accent
    GOLD, NAVY, GREY, LIGHT = "#e53935", "#8db4dc", "#b2a99c", "#5c4c3f"    # GOLD = the Real Madrid colour (red here)
    RIVALS = {131: "#a78bda", 13: "#4db6ac"}            # Barcelona purple, Atletico teal (Real Madrid is red)
    UP, DOWN = "#66bb6a", "#7fa7d6"                     # DOWN is slate blue so it never looks like Real Madrid
    SEQ = [[0, "#2b211b"], [0.35, "#5d3b33"], [0.7, "#b4544c"], [1, "#f5ddc8"]]
elif THEME == "club":
    MONO = ["Barlow Condensed", "Bahnschrift", "Arial Narrow", "Segoe UI", "Arial", "sans-serif"]   # numbers + labels
    SANS = ["Barlow", "Segoe UI", "Helvetica Neue", "Arial", "sans-serif"]
    BG, PANEL, GRIDC = "#0b1626", "#111f33", "#22334b"
    INK, INK2, MUTED, ACCENT = "#f3f5f9", "#b9c4d4", "#7f8ca1", "#d4af37"
    GOLD, NAVY, GREY, LIGHT = "#d4af37", "#6fa3d8", "#8c99ab", "#3b4b63"
    RIVALS = {131: "#b5487a", 13: "#e0634a"}            # Barcelona, Atletico
    UP, DOWN = "#3cb371", "#e05252"
    SEQ = [[0, "#0e1b2e"], [0.35, "#1f4a78"], [0.7, "#6f93bf"], [1, "#d4af37"]]
elif THEME == "terminal":
    BG, PANEL, GRIDC = "#0a0d12", "#111720", "#222a35"
    INK, INK2, MUTED, ACCENT = "#e8e8e8", "#b4bac4", "#78818d", "#FEBE10"
    GOLD, NAVY, GREY, LIGHT = "#FEBE10", "#5fa8ff", "#8a94a3", "#3a4452"
    RIVALS = {131: "#e0457b", 13: "#ff6b4a"}            # Barcelona, Atletico
    UP, DOWN = "#2fd08b", "#ff4d4d"
    SEQ = [[0, "#0d1622"], [0.35, "#1d4e7e"], [0.7, "#6b8fb8"], [1, "#FEBE10"]]
else:
    BG, PANEL, GRIDC = "#ffffff", "#ffffff", "#e6e5e1"
    INK, INK2, MUTED, ACCENT = "#0b0b0b", "#52514e", "#8a8984", "#1B2A49"
    GOLD, NAVY, GREY, LIGHT = "#FEBE10", "#1B2A49", "#455A64", "#B0BEC5"
    RIVALS = {131: "#A50044", 13: "#CB3524"}
    UP, DOWN = "#2E7D32", "#C62828"
    SEQ = [[0, "#fff7dc"], [0.45, "#FEBE10"], [1, "#1B2A49"]]
CMAP = LinearSegmentedColormap.from_list("rm_seq", [c for _, c in SEQ])


def setup():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 220, "savefig.bbox": "tight",
        "figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG,
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": GRIDC, "axes.linewidth": 0.8,
        "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
        "axes.titlecolor": ACCENT, "axes.grid": True, "grid.color": GRIDC, "grid.linewidth": 0.6,
        "axes.titleweight": "bold", "axes.titlesize": 11.5, "axes.titlelocation": "left", "lines.linewidth": 1.8,
        "legend.frameon": False, "legend.labelcolor": INK2, "font.size": 9.5,
        "font.family": "monospace" if THEME == "terminal" else "sans-serif", "font.monospace": MONO, "font.sans-serif": SANS,
        "axes.prop_cycle": matplotlib.cycler(color=[NAVY, GREY, UP, DOWN, GOLD]),   # gold is reserved for Real Madrid
        "patch.edgecolor": BG, "boxplot.flierprops.markeredgecolor": INK2,
    })


def source(fig):
    """Source line, bottom-right, on every saved figure."""
    fig.text(0.995, 0.003, SOURCE, ha="right", va="bottom", fontsize=6.5, color=MUTED)


def css_family(fonts):
    """A CSS font-family list; names with spaces are quoted (an unquoted 'Source Sans 3' is invalid CSS)."""
    return ", ".join(f"'{f}'" if " " in f else f for f in fonts)


def plotly_template():
    import plotly.graph_objects as go
    fam = css_family(MONO if THEME in ("terminal", "club", "editorial") else SANS)
    axis = dict(gridcolor=GRIDC, zerolinecolor=GRIDC, linecolor=GRIDC, tickfont=dict(color=INK2), title=dict(font=dict(color=INK2)))
    return go.layout.Template(layout=dict(
        paper_bgcolor=BG, plot_bgcolor=PANEL, colorway=[GOLD, NAVY, GREY, UP, DOWN],
        font=dict(family=fam, size=17, color=INK), title=dict(font=dict(color=INK if THEME in ("club", "editorial") else ACCENT, size=21), x=0.015, xanchor="left"),
        xaxis=axis, yaxis=axis, legend=dict(font=dict(color=INK2), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor=PANEL, bordercolor=ACCENT, font=dict(family=fam, color=INK)),
        updatemenudefaults=dict(bgcolor=PANEL, bordercolor=GRIDC, font=dict(color=ACCENT)),
        sliderdefaults=dict(bgcolor=GRIDC, activebgcolor=ACCENT, bordercolor=GRIDC, font=dict(color=INK2)),
    ))


DECK = dict(serif=SERIF, bg=BG, panel=PANEL, grid=GRIDC, ink=INK, ink2=INK2, muted=MUTED, accent=ACCENT,
            c1=GOLD, c2=NAVY, c3=GREY, mono=MONO, sans=SANS, theme=THEME)

setup()
