"""Slide figures for TeamF_Slides.tex (the four convergence figures in slides/figures/).

Reads   ../results/gbm_strong.csv, gbm_weak.csv, nonaffine_strong.csv, nonaffine_weak.csv (the shipped full-run CSVs,
        path relative to this file); writes figures/*.pdf next to this file. No simulation; data are not modified.
Style   figure text typeset by LaTeX in URW Palladio (mathpazo, the deck's Palatino), real searchable Type 1 text with the
        deck's 0.15 pt fill+stroke weight (heavy()); drawn ~1:1 for the 47 mm frame height: ticks 8.6 pt, axis labels 9.1 pt,
        inline labels 8.85 pt (x0.98 on the slide); near-ink text colours, 0.8 pt dark-grey spines/ticks; no grid, no
        legend boxes, direct labels, grey dashed reference slopes, navy circle = EM / ref. 4096, brick square = Milstein /
        ref. 8192; filled = fitted, open = not fitted; p7-p9 series offset +-2.5 pt horizontally for visibility.
Run     python make_slide_figures.py   (from any directory; needs matplotlib, numpy and a TeX installation with
        mathpazo and cm-super, because text.usetex = True)
"""
import csv
from fractions import Fraction
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator, NullFormatter, LogLocator
from matplotlib.transforms import ScaledTranslation

import slide_rcparams as S
# Figure text typeset by LaTeX in URW Palladio (= the deck's mathpazo Palatino, Type 1, searchable),
# drawn at ~1:1 scale on the slide (figure height ~ the 47 mm frame height), ticks 8.5 pt, labels 9 pt, inline labels
# 8.75 pt, text in near-ink colours, and given the deck's 0.15 pt fill+stroke weight by a post-processing step (heavy()).
TICK, AXIS, LAB = "#374151", "#1F2933", "#4B5563"          # tick labels, axis labels, spines/ticks + grey labels
plt.rcParams.update({"pdf.fonttype": 42, "figure.figsize": (2.80, 1.92), "font.size": 8.85,
                     "text.usetex": True, "font.family": "serif",
                     "text.latex.preamble": r"\usepackage[sc]{mathpazo}\usepackage{amsmath,amssymb}",
                     "axes.labelsize": 9.1, "xtick.labelsize": 8.6, "ytick.labelsize": 8.6,
                     "axes.labelcolor": AXIS, "xtick.labelcolor": TICK, "ytick.labelcolor": TICK,
                     "axes.edgecolor": LAB, "xtick.color": LAB, "ytick.color": LAB, "axes.linewidth": 0.8,
                     "xtick.major.width": 0.8, "ytick.major.width": 0.8, "xtick.major.size": 3.5, "ytick.major.size": 3.5,
                     "xtick.minor.width": 0.55, "ytick.minor.width": 0.55, "xtick.minor.size": 2.0, "ytick.minor.size": 2.0,
                     "xtick.major.pad": 2.5, "ytick.major.pad": 2.5, "axes.labelpad": 3.0,
                     "lines.linewidth": 1.2, "lines.markersize": 3.6,
                     "axes.ymargin": 0.10})   # no marker or bar end touches a spine

DATA = Path(__file__).resolve().parent.parent / "results"
OUT = Path(__file__).resolve().parent / "figures"
NAVY, RED, MUTED, INK = S.ACCENT, S.RED, S.MUTED, S.INK
FS = 8.85
STROKE = 0.15   # pt, = \\strokeweight of the deck (figures are placed at ~1:1)
# Marker convention (all four figures): colour + shape = series (navy circle = EM / reference 4096,
# brick square = Milstein / reference 8192); filled = point used in the slope fit (fit_included == 1),
# open = not fitted. Where two series share an x position (p7, p8, p9) each series is drawn DX points
# left/right of its true h purely for visibility (display-space shift; data x = h exactly).
DX = 2.5


def shifted(ax, dx_pt, artists):
    """Shift artists horizontally by dx_pt points in display space (visibility only; data unchanged)."""
    off = ScaledTranslation(dx_pt / 72.0, 0, ax.figure.dpi_scale_trans)
    for a in artists:
        a.set_transform(a.get_transform() + off)
    return artists


def eb_artists(cont):
    """All drawable artists of an ErrorbarContainer."""
    line, caps, cols = cont.lines
    return [a for a in [line, *caps, *cols] if a is not None]


def rows(name):
    with (DATA / name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def series(data, key, value):
    sel = sorted((r for r in data if r[key] == value), key=lambda r: float(r["h"]))
    g = lambda k: np.array([float(r[k]) for r in sel])
    return sel, g("h"), g("estimate"), g("se"), np.array([r["fit_included"] == "1" for r in sel])


def frac(x, _=None):
    q = Fraction(x).limit_denominator(100000)
    return r"$1/%d$" % q.denominator if q.numerator == 1 else r"$%s$" % q


def h_axis(ax, hs, right=1.35):
    ax.set_xscale("log", base=2)
    t = sorted(set(hs))
    ax.xaxis.set_major_locator(FixedLocator(t[0::2] if len(t) > 6 else t))
    ax.xaxis.set_minor_locator(FixedLocator(t[1::2]) if len(t) > 6 else NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(frac))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r"step size $h$")
    ax.set_xlim(t[0] / 1.35, t[-1] * right)


def marks(ax, h, y, se, fit, color, marker, dx=0.0, line=True, ms=None, yerr=None):
    """Error bars (95%: 1.96 se, or explicit yerr), optional connecting line, filled = fitted / open = not fitted."""
    kw = {} if ms is None else {"ms": ms}
    arts = eb_artists(ax.errorbar(h, y, yerr=1.96 * se if yerr is None else yerr, fmt="none", ecolor=color,
                                  elinewidth=0.8, zorder=2))
    if line:
        arts += ax.plot(h, y, "-", color=color, lw=1.0, zorder=2)
    arts += ax.plot(h[fit], y[fit], marker, color=color, mec=color, ls="none", zorder=3, **kw)
    arts += ax.plot(h[~fit], y[~fit], marker, mfc="white", mec=color, ls="none", zorder=3, **kw)
    if dx:
        shifted(ax, dx, arts)


def ref_slope(ax, x0, x1, y0, p, label):
    """Grey dashed reference segment of slope p from (x0, y0) to x1, label at its right end."""
    xs = np.array([x0, x1]); ys = y0 * (xs / x0) ** p
    ax.plot(xs, ys, ls=(0, (4, 2.5)), color=MUTED, lw=0.8, zorder=1)
    ax.text(x1 * 1.12, ys[-1], label, color=LAB, fontsize=FS, va="center", ha="left")


def heavy(path, width=STROKE):
    """Give every text object the deck's fill+stroke weight: wrap each BT..ET in
    q <w> w <fill colour as stroke colour> 2 Tr ... Q (text stays real Type 1 text; lines untouched)."""
    import re, zlib
    raw = path.read_bytes()
    def fix(stream):
        out, pos, fill = [], 0, "0 G"
        for m in re.finditer(rb"([\d.]+ [\d.]+ [\d.]+) rg|([\d.]+) g\b|BT\b.*?ET", stream, re.S):
            if m.group(1): fill = m.group(1).decode() + " RG"; continue
            if m.group(2): fill = m.group(2).decode() + " G"; continue
            out.append(stream[pos:m.start()])
            out.append(("q %g w 1 j %s 2 Tr\n" % (width, fill)).encode() + m.group(0) + b"\nQ")
            pos = m.end()
        out.append(stream[pos:]); return b"".join(out)
    # locate the page content stream exactly (by /Contents and /Length), never by pattern-matching binary data
    cont = re.search(rb"/Contents (\d+) 0 R", raw).group(1)
    o = raw.index(b"\n" + cont + b" 0 obj") + 1
    d0 = raw.index(b"<<", o); s0 = raw.index(b"stream", d0)
    head = raw[d0:s0]
    L = re.search(rb"/Length (\d+)( 0 R)?", head)
    if L.group(2):
        lo = raw.index(b"\n" + L.group(1) + b" 0 obj") + 1
        length = int(re.match(rb"\d+ 0 obj\s*(\d+)", raw[lo:]).group(1))
    else:
        length = int(L.group(1))
    b0 = s0 + len(b"stream") + (2 if raw[s0 + 6:s0 + 8] == b"\r\n" else 1)
    data = zlib.decompress(raw[b0:b0 + length])
    newdata = zlib.compress(fix(data))
    e0 = raw.index(b"endstream", b0 + length)
    head = re.sub(rb"/Length \d+(?: 0 R)?", b"/Length %d" % len(newdata), head)
    new = raw[:d0] + head + b"stream\n" + newdata + b"\n" + raw[e0:]
    # rebuild xref offsets
    objs = list(re.finditer(rb"(?m)^(\d+) 0 obj", new))
    xs = new.rindex(b"\nxref\n") + 1
    n = max(int(m.group(1)) for m in objs) + 1
    off = {int(m.group(1)): m.start() for m in objs}
    xref = [b"xref\n0 %d\n0000000000 65535 f \n" % n]
    for k in range(1, n):
        xref.append(b"%010d 00000 n \n" % off[k] if k in off else b"0000000000 65535 f \n")
    tail = new[new.index(b"trailer", xs):]
    tail = re.sub(rb"startxref\s+\d+", b"startxref\n%d" % xs, tail)
    path.write_bytes(new[:xs] + b"".join(xref) + tail)


def label(ax, k, text, color, corner="tl"):
    """Inline series label k (0, 1, 2) stacked from a corner at a fixed 10.6 pt line pitch."""
    x, ha = (0.03, "left") if corner[1] == "l" else (0.97, "right")
    if corner[0] == "t":
        ax.annotate(text, (x, 0.97), xycoords="axes fraction", xytext=(0, -10.6 * k), textcoords="offset points",
                    color=color, fontsize=FS, ha=ha, va="top")
    else:   # bottom: k = 0 is the upper line
        ax.annotate(text, (x, 0.04), xycoords="axes fraction", xytext=(0, 10.6 * k), textcoords="offset points",
                    color=color, fontsize=FS, ha=ha, va="bottom")


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / name)
    plt.close(fig)
    heavy(OUT / name)


def gbm_strong():
    d = rows("gbm_strong.csv")
    fig, ax = plt.subplots()
    for k, (m, col, mk, lab) in enumerate((("EM", NAVY, "o", "Euler--Maruyama"), ("Milstein", RED, "s", "Milstein"))):
        sel, h, y, se, fit = series(d, "method", m)
        marks(ax, h, y, se, fit, col, mk)
        label(ax, 1 - k, lab + r", slope $%.4f$" % float(sel[0]["fitted_slope"]), col, "br")
    ref_slope(ax, 1 / 512, 1 / 2, 0.045, 0.5, "1/2")
    ref_slope(ax, 1 / 512, 1 / 2, 0.0007, 1.0, "1")
    ax.set_yscale("log")
    h_axis(ax, h, right=2.1)
    ax.set_ylim(2e-4, 3.0)   # room for the labels below the slope-1 guide
    ax.set_ylabel(r"$\mathbb{E}\,|S_N-S_T|$")
    save(fig, "gbm_strong.pdf")


def gbm_weak():
    d = rows("gbm_weak.csv")
    fig, (ax, bx) = plt.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [2.1, 1], "hspace": 0.12})
    for k, (m, col, mk, lab, dx) in enumerate((("EM", NAVY, "o", "Euler--Maruyama", -DX),
                                                ("Milstein", RED, "s", "Milstein", +DX))):
        sel, h, y, se, fit = series(d, "method", m)
        a = np.abs(y)
        # top: |bias| with |y +- 1.96 se| bars; bottom: signed bias +- 1.96 se. Same filled/open rule in both.
        marks(ax, h, a, se, fit, col, mk, dx=dx, line=False,
              yerr=[a - np.abs(y + 1.96 * se), np.abs(y - 1.96 * se) - a])
        marks(bx, h, y, se, fit, col, mk, dx=dx, line=False, ms=2.6)
        label(ax, k, lab + r", slope $%.4f$" % float(sel[0]["fitted_slope"]), col)
    hh = np.array(sorted({float(r["h"]) for r in d}))
    ab = np.array([float(next(r for r in d if float(r["h"]) == x)["analytic_bias"]) for x in hh])
    ax.plot(hh, np.abs(ab), "-", color=MUTED, lw=0.9, zorder=1)
    label(ax, 0, r"analytic $|\delta(h)|$ (grey)", LAB, "br")   # bottom-right: clear of the data
    bx.plot(hh, ab, "-", color=MUTED, lw=0.9, zorder=1)
    bx.axhline(0, color=MUTED, lw=0.6, ls=(0, (4, 2.5)))
    ax.set_yscale("log")
    ax.set_ylabel(r"$|\mathbb{E}\,S_N-\mathbb{E}\,S_T|$")
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=12))   # every decade (margins must not thin the ticks)
    bx.set_ylabel("signed bias")
    bx.set_ylim(-0.078, 0.009)   # clear of both spines
    h_axis(bx, hh)
    fig.align_ylabels((ax, bx))
    save(fig, "gbm_weak.pdf")


def nonaffine_strong():
    d = rows("nonaffine_strong.csv")
    fig, ax = plt.subplots()
    for k, (ref, col, mk) in enumerate((("4096", NAVY, "o"), ("8192", RED, "s"))):
        sel, h, y, se, fit = series(d, "reference_N", ref)
        marks(ax, h, y, se, fit, col, mk, dx=(-DX, +DX)[k])
        label(ax, 1 - k, r"ref.\ $N=%s$, slope $%.4f$" % (ref, float(sel[0]["fitted_slope"])), col, "br")
    ref_slope(ax, 1 / 512, 1 / 32, 0.115, 0.5, "1/2")
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(FixedLocator([0.15, 0.2, 0.3, 0.4, 0.6]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: r"$%g$" % v))
    h_axis(ax, h, right=1.9)
    ax.set_ylim(0.088, 0.95)   # room for the labels below the slope guide
    ax.set_ylabel(r"$\mathbb{E}\,|S_N-S_{N_{\mathrm{ref}}}|$")
    save(fig, "nonaffine_strong.pdf")


def nonaffine_weak():
    d = rows("nonaffine_weak.csv")
    fig, (ax, bx) = plt.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [1.6, 1.2], "hspace": 0.12})
    for k, (ref, col, mk, dx) in enumerate((("4096", NAVY, "o", -DX), ("8192", RED, "s", +DX))):
        sel, h, y, se, fit = series(d, "reference_N", ref)   # fit_included is 0 everywhere: all open
        a = np.abs(y)
        shifted(ax, dx, ax.plot(h, a, "-", color=col, lw=1.0, zorder=2)
                + ax.plot(h[fit], a[fit], mk, color=col, mec=col, ls="none", zorder=3)
                + ax.plot(h[~fit], a[~fit], mk, mfc="white", mec=col, ls="none", zorder=3))
        marks(bx, h, y, se, fit, col, mk, dx=dx, line=False, ms=2.6)
        label(ax, k, r"ref.\ $N=%s$" % ref, col)
    bx.axhline(0, color=MUTED, lw=0.6, ls=(0, (4, 2.5)))   # same zero line as gbm_weak
    ax.set_yscale("log")
    ax.set_ylabel(r"$|\Delta\phi|$")
    ax.yaxis.set_major_locator(FixedLocator([5e-4, 1e-3, 2e-3, 4e-3]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: r"$%g$" % v))
    bx.set_ylabel(r"$\Delta\phi$, 95\% CI")
    bx.yaxis.set_major_locator(FixedLocator([0.0, 0.005, 0.010]))
    bx.yaxis.set_major_formatter(FuncFormatter(lambda v, _: r"$%g$" % round(v, 4)))
    h_axis(bx, h)
    fig.align_ylabels((ax, bx))
    save(fig, "nonaffine_weak.pdf")


if __name__ == "__main__":
    for f in (gbm_strong, gbm_weak, nonaffine_strong, nonaffine_weak):
        f()
    print("wrote", sorted(p.name for p in OUT.glob("*.pdf")))
