"""Shared Matplotlib style for every report figure (Palatino-like serif, 300 dpi).

TeX Gyre Pagella is used when it is installed (TeX Live / TinyTeX or ~/.local/share/fonts);
otherwise Matplotlib's bundled DejaVu Serif is used, so the scripts run on any machine.
"""
from __future__ import annotations
import glob
import logging
import os
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager as fm

# Okabe-Ito based, colour-blind safe; closures also differ by marker and line style.
NAVY, BRICK, SAGE, GREY, DARK = "#0072B2", "#D55E00", "#009E73", "#6e6e6e", "#333333"

# Font sizes in points at the printed size (figures are drawn at their final width).
FS_TICK, FS_LEGEND, FS_LABEL, FS_TITLE = 8.5, 8.5, 9.5, 10.0

_PATTERNS = (
    "/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyrepagella-*.otf",
    "/usr/share/texlive/texmf-dist/fonts/opentype/public/tex-gyre/texgyrepagella-*.otf",
    "~/.TinyTeX/texmf-dist/fonts/opentype/public/tex-gyre/texgyrepagella-*.otf",
    "~/.local/share/fonts/texgyrepagella-*.otf",
)


def setup():
    """Register Pagella if available and apply the report rcParams. Returns the family used."""
    # Fixed PDF creation date so that re-running the scripts gives byte-identical figures.
    os.environ.setdefault("SOURCE_DATE_EPOCH", "1794528000")   # 2026-11-13 00:00 UTC
    found = []
    for pat in _PATTERNS:
        found += glob.glob(os.path.expanduser(pat))
    for f in found:
        fm.fontManager.addfont(f)
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    if found:
        family = "TeX Gyre Pagella"
        math = {"mathtext.fontset": "custom", "mathtext.rm": family,
                "mathtext.it": family + ":italic", "mathtext.bf": family + ":bold"}
    else:
        family = "DejaVu Serif"
        math = {"mathtext.fontset": "dejavuserif"}
    matplotlib.rcParams.update({
        "font.family": "serif", "font.serif": [family, "DejaVu Serif"],
        **math,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": FS_LABEL, "axes.titlesize": FS_TITLE, "axes.labelsize": FS_LABEL,
        "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK, "legend.fontsize": FS_LEGEND,
        "axes.linewidth": 0.6, "lines.linewidth": 1.0, "lines.markersize": 4.5,
        "legend.frameon": False, "pdf.fonttype": 42,
        "savefig.dpi": 300, "figure.dpi": 100,
    })
    return family


def color_legend_text(leg):
    """Colour each legend label exactly like its handle."""
    for h, t in zip(leg.legend_handles, leg.get_texts()):
        c = h.get_color() if hasattr(h, "get_color") else h.get_edgecolor()
        if not isinstance(c, str) and hasattr(c, "__len__") and len(c) and hasattr(c[0], "__len__"):
            c = c[0]
        t.set_color(c)
    return leg
