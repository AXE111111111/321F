"""Shared Matplotlib style and colour palette for the slide figures (Palatino-like serif; colours match polish_preamble.tex).
Draw each figure at the size it will appear on the slide (scale ~1), so 9pt here = 9pt on the slide."""
from pathlib import Path
import matplotlib as mpl
from matplotlib import font_manager

INK, ACCENT, RED, MUTED, HAIR = "#1F2933", "#23426B", "#B03A2E", "#6B7280", "#C9CDD2"
SERIES = [ACCENT, RED, "#4E7F52", "#8A6D3B"]   # navy, brick, sage, ochre — max 2-3 per plot

_GYRE = Path("/usr/share/texmf/fonts/opentype/public/tex-gyre")  # TeX Gyre Pagella = Palatino clone
for f in ("regular", "italic", "bold", "bolditalic"):
    p = _GYRE / f"texgyrepagella-{f}.otf"
    if p.exists():
        font_manager.fontManager.addfont(str(p))

# slide is 160mm x 90mm; a figure in 0.60\textwidth column is ~3.35in wide
FIG_W, FIG_H = 3.35, 2.25

RC = {
    "figure.figsize": (FIG_W, FIG_H),
    "font.family": "TeX Gyre Pagella",
    "mathtext.fontset": "custom", "mathtext.rm": "TeX Gyre Pagella",
    "mathtext.it": "TeX Gyre Pagella:italic", "mathtext.bf": "TeX Gyre Pagella:bold",
    "mathtext.fallback": "stix",
    "font.size": 9.5, "axes.labelsize": 9.5, "legend.fontsize": 9,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,       # >= 8.5pt: readable from the back row
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED,            # ticks grey, data dark
    "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "xtick.major.size": 3, "ytick.major.size": 3, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
    "xtick.direction": "out", "ytick.direction": "out",
    "axes.grid": False,
    "axes.prop_cycle": mpl.cycler(color=SERIES),
    "lines.linewidth": 1.4, "lines.markersize": 4, "lines.markeredgewidth": 0.8,
    "errorbar.capsize": 0,
    "legend.frameon": False, "legend.handlelength": 1.6,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "savefig.transparent": False, "figure.dpi": 150,
}
mpl.rcParams.update(RC)
