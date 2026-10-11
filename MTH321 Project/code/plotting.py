"""Publication-ready PDF figures rendered solely from saved CSV data.

The five report figures are drawn in the report's typography: text is typeset by LaTeX in Palatino (``mathpazo``,
the report body font) at the printed size (1:1), navy ``#23426B`` = EM / reference N=4096, brick ``#B03A2E`` =
Milstein / reference N=8192, circles vs. squares, filled markers = points used in the slope fit, open markers =
excluded points, 95% error bars, dashed fitted-slope guides, no grid, direct labels instead of a legend box.

``text.usetex`` needs a LaTeX installation with ``mathpazo``. If it is unavailable (or the environment variable
``TEAMF_PLOTS_NO_TEX=1`` is set) the figures are drawn with Matplotlib's own mathtext and a Palatino-like serif
font instead and a notice is printed; the plotted data are the same, only the glyphs differ. The report figures
are the default (LaTeX) output.
"""
from __future__ import annotations

import csv
import io
import os
import shutil
from fractions import Fraction
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter, NullLocator
from matplotlib.transforms import ScaledTranslation
from scipy.stats import lognorm

INK, NAVY, RED, MUTED = "#1F2933", "#23426B", "#B03A2E", "#6B7280"
TICK, AXIS, LAB, NOTE = "#374151", "#1F2933", "#4B5563", "#4B5563"
W = 4.80             # figure width in inches; the report includes the PDFs at natural size (1:1)
FS, FSN = 9.2, 9.6   # direct-label / run-note sizes in pt
DX = 2.0             # pt; overlapping series are drawn DX left/right of their true h (display shift, data unchanged)
TEX_PREAMBLE = r"\usepackage[T1]{fontenc}\usepackage[sc]{mathpazo}\usepackage{amsmath,amssymb}"
FALLBACK_SERIF = ["TeX Gyre Pagella", "Palatino", "Palatino Linotype", "URW Palladio L", "P052", "Book Antiqua",
                  "DejaVu Serif"]

_USETEX: bool | None = None


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _latex_works() -> bool:
    """True if Matplotlib can typeset a test string with LaTeX + mathpazo on this machine."""
    if os.environ.get("TEAMF_PLOTS_NO_TEX", "") not in ("", "0"):
        return False
    if shutil.which("latex") is None:
        return False
    try:
        with plt.rc_context({"text.usetex": True, "text.latex.preamble": TEX_PREAMBLE}):
            fig = plt.figure(figsize=(1, 1))
            fig.text(0.5, 0.5, r"$\mathbb{E}\,|S_N-S_T|$ 95\% CI")
            fig.savefig(io.BytesIO(), format="pdf")
            fig.canvas.draw()
            plt.close(fig)
        return True
    except Exception:  # noqa: BLE001 - any LaTeX/dvi failure means "fall back"
        plt.close("all")
        return False


def _style() -> bool:
    """Apply the figure style; returns True when LaTeX typesetting is used."""
    global _USETEX
    if _USETEX is None:
        _USETEX = _latex_works()
        if not _USETEX:
            print("plotting.py: LaTeX with mathpazo is not available (or TEAMF_PLOTS_NO_TEX is set); drawing the "
                  "figures with Matplotlib mathtext and a Palatino-like serif instead. The data are identical, but "
                  "the glyphs differ from the report figures, which use the default LaTeX rendering.", flush=True)
    plt.rcdefaults()
    matplotlib.use("Agg")
    rc = {
        "font.family": "serif", "pdf.fonttype": 42,
        "font.size": FS, "axes.labelsize": 9.6, "xtick.labelsize": 9.0, "ytick.labelsize": 9.0,
        "axes.labelcolor": AXIS, "xtick.labelcolor": TICK, "ytick.labelcolor": TICK, "text.color": INK,
        "axes.edgecolor": LAB, "xtick.color": LAB, "ytick.color": LAB, "axes.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 3.2, "ytick.major.size": 3.2,
        "xtick.minor.width": 0.45, "ytick.minor.width": 0.45, "xtick.minor.size": 1.8, "ytick.minor.size": 1.8,
        "xtick.major.pad": 2.5, "ytick.major.pad": 2.5, "axes.labelpad": 3.5, "axes.grid": False,
        "lines.linewidth": 1.0, "lines.markersize": 4.2, "lines.markeredgewidth": 0.8,
        "legend.frameon": False, "axes.ymargin": 0.08,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.03, "figure.dpi": 150,
    }
    if _USETEX:
        rc.update({"text.usetex": True, "text.latex.preamble": TEX_PREAMBLE})
    else:
        rc.update({"text.usetex": False, "font.serif": FALLBACK_SERIF, "mathtext.fontset": "custom",
                   "mathtext.rm": "serif", "mathtext.it": "serif:italic", "mathtext.bf": "serif:bold",
                   "mathtext.fallback": "stix"})
    plt.rcParams.update(rc)
    return _USETEX


def _pct() -> str:
    return r"\%" if _USETEX else "%"


def _series(data, key, value):
    sel = sorted((r for r in data if r[key] == value), key=lambda r: float(r["h"]))
    col = lambda k: np.array([float(r[k]) for r in sel])
    return sel, col("h"), col("estimate"), col("se"), np.array([r["fit_included"] == "1" for r in sel], dtype=bool)


def _frac(x, _=None):
    q = Fraction(x).limit_denominator(100000)
    return r"$1/%d$" % q.denominator if q.numerator == 1 else r"$%s$" % q


def _h_axis(ax, hs, right=1.3):
    """log2 step-size axis labelled 1/2 ... 1/512 (major tick on every other grid if there are more than 6)."""
    ax.set_xscale("log", base=2)
    t = sorted(set(hs))
    ax.xaxis.set_major_locator(FixedLocator(t[0::2] if len(t) > 6 else t))
    ax.xaxis.set_minor_locator(FixedLocator(t[1::2]) if len(t) > 6 else NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(_frac))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r"Time step $h=\Delta t$")
    ax.set_xlim(t[0] / 1.3, t[-1] * right)


def _group_label(sel, group, key):
    """Series label: group name, fitted slope and number of fitted points (or why no slope is reported)."""
    lab = f"reference $N={group}$" if key == "reference_N" else group
    slope = sel[0].get("fitted_slope", "")
    fitn = int(sel[0].get("fit_point_count", "0") or 0)
    elig = int(sel[0].get("eligible_point_count", "0") or 0)
    if slope and fitn >= 4:
        return lab + f" (fit slope {float(slope):.3f}; {fitn} points)"
    if elig >= 2:
        return lab + f" (unresolved; {elig} points)"
    return lab + " (no resolved fit)"


def _label(ax, k, text, color, corner="tl", pitch=11.5):
    x, ha = (0.02, "left") if corner[1] == "l" else (0.98, "right")
    if corner[0] == "t":
        ax.annotate(text, (x, 0.98), xycoords="axes fraction", xytext=(0, -pitch * k), textcoords="offset points",
                    color=color, fontsize=FS, ha=ha, va="top")
    else:
        ax.annotate(text, (x, 0.03), xycoords="axes fraction", xytext=(0, pitch * k), textcoords="offset points",
                    color=color, fontsize=FS, ha=ha, va="bottom")


def _note(fig, ax, lines):
    """Run notes (M, seed, marker convention) 7 pt below the x-axis label, flush left."""
    fig.canvas.draw()
    bb = ax.xaxis.label.get_window_extent().transformed(fig.transFigure.inverted())
    y = bb.y0 - 7.0 / 72.0 / fig.get_figheight()
    fig.text(0.0, y, "\n".join(lines), fontsize=FSN, color=NOTE, va="top", ha="left", linespacing=1.3)


def _shifted(ax, dx_pt, artists):
    if dx_pt:
        off = ScaledTranslation(dx_pt / 72.0, 0, ax.figure.dpi_scale_trans)
        for a in artists:
            a.set_transform(a.get_transform() + off)
    return artists


def _draw_points(ax, h, y, se, fit, color, marker, bars_on="fitted", ms=None, dx=0.0):
    """Connecting line, 95% error bars (fitted points or all points), filled = fitted / open = not fitted."""
    kw = {} if ms is None else {"ms": ms}
    arts = list(ax.plot(h, y, "-", color=color, lw=1.0, alpha=0.9, zorder=2))
    sel = fit if bars_on == "fitted" else np.ones_like(fit)
    if np.any(sel):
        eb = ax.errorbar(h[sel], y[sel], yerr=1.96 * se[sel], fmt="none", ecolor=color, elinewidth=0.8, capsize=0,
                         zorder=2)
        arts += [a for a in [eb.lines[0], *eb.lines[1], *eb.lines[2]] if a is not None]
    arts += ax.plot(h[fit], y[fit], marker, color=color, mec=color, ls="none", zorder=3, **kw)
    arts += ax.plot(h[~fit], y[~fit], marker, mfc="white", mec=color, ls="none", zorder=3, **kw)
    _shifted(ax, dx, arts)


def _fit_guide(ax, h, y, fit, sel, color, dx=0.0):
    """Dashed guide with the fitted slope, anchored at the geometric centre of the fitted points."""
    slope = sel[0].get("fitted_slope", "")
    if slope and np.count_nonzero(fit) >= 4:
        p, xf = float(slope), h[fit]
        c = float(np.mean(np.log(y[fit]) - p * np.log(xf)))
        _shifted(ax, dx, ax.plot(xf, np.exp(c) * xf ** p, ls=(0, (4, 2.5)), color=color, lw=0.9, alpha=0.75,
                                 zorder=1))


def _save(fig, pdf_path: Path) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, metadata={"CreationDate": None})
    plt.close(fig)


def _run_notes(rows, notes):
    return [n.format(M=f"{int(rows[0]['M']):,}", seed=rows[0]["seed"], pct=_pct()) for n in notes]


def strong_figure(csv_path: Path, pdf_path: Path, *, key, groups, ylabel, notes, ylim=None, yticks=None, dx=0.0):
    _style()
    d = read_rows(csv_path)
    fig, ax = plt.subplots(figsize=(W, 3.05))
    for k, (g, col, mk) in enumerate(zip(groups, (NAVY, RED), ("o", "s"))):
        sel, h, y, se, fit = _series(d, key, g)
        _draw_points(ax, h, y, se, fit, col, mk, dx=(-dx, dx)[k])
        _fit_guide(ax, h, y, fit, sel, col, dx=(-dx, dx)[k])
        _label(ax, 1 - k, _group_label(sel, g, key), col, "br")
    ax.set_yscale("log")
    if yticks is not None:
        ax.yaxis.set_major_locator(FixedLocator(yticks))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: r"$%g$" % v))
    if ylim:
        ax.set_ylim(*ylim)
    _h_axis(ax, [float(r["h"]) for r in d])
    ax.set_ylabel(ylabel)
    _note(fig, ax, _run_notes(d, notes))
    _save(fig, pdf_path)


def weak_figure(csv_path: Path, pdf_path: Path, *, key, groups, ylabel, notes, analytic, yticks=None, bticks=None,
                hr=(2.1, 1)):
    _style()
    d = read_rows(csv_path)
    fig, (ax, bx) = plt.subplots(2, 1, sharex=True, figsize=(W, 4.3),
                                 gridspec_kw={"height_ratios": hr, "hspace": 0.10})
    for k, (g, col, mk) in enumerate(zip(groups, (NAVY, RED), ("o", "s"))):
        sel, h, y, se, fit = _series(d, key, g)
        dx = (-DX, DX)[k]
        _draw_points(ax, h, np.abs(y), se, fit, col, mk, dx=dx)                 # top: |estimate|, bars on fitted
        _fit_guide(ax, h, np.abs(y), fit, sel, col, dx=dx)
        _draw_points(bx, h, y, se, fit, col, mk, bars_on="all", ms=3.2, dx=dx)  # bottom: signed, all 95% CIs
        _label(ax, k, _group_label(sel, g, key), col, "tl")
    if analytic and "analytic_bias" in d[0]:
        first = sorted((r for r in d if r[key] == groups[0]), key=lambda r: float(r["h"]))
        hh = np.array([float(r["h"]) for r in first])
        ab = np.array([float(r["analytic_bias"]) for r in first])
        ax.plot(hh, np.abs(ab), ls=(0, (1, 1.6)), color=INK, lw=1.2, zorder=1)
        bx.plot(hh, ab, ls=(0, (1, 1.6)), color=INK, lw=1.2, zorder=1)
        _label(ax, 2, r"analytic $|$bias$|$", INK, "tl")
    bx.axhline(0.0, color=MUTED, lw=0.6, zorder=0)
    ax.set_yscale("log")
    if yticks is not None:
        ax.yaxis.set_major_locator(FixedLocator(yticks))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: r"$%g$" % v))
    else:
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
    if bticks is not None:
        bx.yaxis.set_major_locator(FixedLocator(bticks))
        bx.yaxis.set_major_formatter(FuncFormatter(lambda v, _: r"$%g$" % round(v, 4)))
    ax.set_ylabel(ylabel)
    bx.set_ylabel("Signed paired difference")
    _h_axis(bx, [float(r["h"]) for r in d])
    fig.align_ylabels((ax, bx))
    _note(fig, bx, _run_notes(d, notes))
    _save(fig, pdf_path)


def distribution_figure(csv_path: Path, pdf_path: Path) -> None:
    _style()
    d = read_rows(csv_path)
    fig, ax = plt.subplots(figsize=(4.4, 2.85))
    n = d[0]["N"]
    for method, col, ls in (("EM", NAVY, "-"), ("Milstein", RED, (0, (3, 1.5)))):
        hist = [r for r in d if r["record_type"] == "histogram" and r["method"] == method]
        edges = np.array([float(r["bin_left"]) for r in hist] + [float(hist[-1]["bin_right"])])
        heights = np.array([float(r["density"]) for r in hist])
        ax.stairs(heights, edges, color=col, lw=0.9, ls=ls)
    s0, mu, sigma, t = (float(d[0][k]) for k in ("s0", "mu", "sigma", "T"))
    x = np.linspace(max(edges[0], 1e-10), edges[-1], 700)
    scale = np.exp(np.log(s0) + (mu - 0.5 * sigma ** 2) * t)
    ax.plot(x, lognorm.pdf(x, s=sigma * np.sqrt(t), scale=scale), color=INK, lw=1.5, alpha=0.55, zorder=0)
    _label(ax, 0, f"EM ($N={n}$)", NAVY, "tr")
    _label(ax, 1, f"Milstein ($N={n}$)", RED, "tr")
    _label(ax, 2, "Exact lognormal density", LAB, "tr")
    ax.set_xlabel(r"Terminal asset price $S_T$")
    ax.set_ylabel("Probability density")
    ax.set_ylim(0, None)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: r"$%g$" % round(v, 4)))
    _note(fig, ax, _run_notes(d, ["GBM terminal distribution; M={M}; seed={seed}"]))
    _save(fig, pdf_path)


def make_all_figures(results: Path, figures: Path) -> list[Path]:
    figures.mkdir(parents=True, exist_ok=True)
    strong_note = "M={M}; seed={seed}. Open = excluded from fit; bars = 95{pct} CI."
    weak_notes = ["M={M}; seed={seed}. Top: absolute estimates (open = unresolved).",
                  "Bottom: signed estimates with 95{pct} CIs."]
    ref_note = "References are numerical EM solutions, not exact paths; slopes are empirical."
    made = []

    def out(name: str) -> Path:
        made.append(figures / name)
        return made[-1]

    strong_figure(results / "gbm_strong.csv", out("gbm_strong_convergence.pdf"), key="method",
                  groups=("EM", "Milstein"), ylabel=r"$\mathbb{E}\,|S_N-S_T|$", notes=[strong_note],
                  ylim=(1.5e-3, 3.0))
    weak_figure(results / "gbm_weak.csv", out("gbm_weak_convergence.pdf"), key="method", groups=("EM", "Milstein"),
                ylabel=r"$|\mathbb{E}[S_N-S_T]|$", notes=weak_notes, analytic=True)
    strong_figure(results / "nonaffine_strong.csv", out("nonaffine_strong_convergence.pdf"), key="reference_N",
                  groups=("4096", "8192"), ylabel=r"$\mathbb{E}\,|S_N-S_{N_{\mathrm{ref}}}|$",
                  notes=[strong_note, ref_note], ylim=(0.075, 0.85), yticks=[0.1, 0.15, 0.2, 0.3, 0.4, 0.6], dx=DX)
    weak_figure(results / "nonaffine_weak.csv", out("nonaffine_weak_convergence.pdf"), key="reference_N",
                groups=("4096", "8192"), ylabel=r"$|\mathbb{E}[\phi(S_N)-\phi(S_{N_{\mathrm{ref}}})]|$",
                notes=weak_notes + [ref_note], analytic=False, yticks=[5e-4, 1e-3, 2e-3, 4e-3],
                bticks=[-0.005, 0.0, 0.005, 0.010], hr=(1.6, 1.2))
    distribution_figure(results / "gbm_distribution.csv", out("gbm_distribution.pdf"))
    return made
