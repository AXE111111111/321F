"""Incompatible-corner experiment: same five-point scheme, exact u*=Im(z Log z).

This runner is independent of the smooth e^{x+y} pipeline. It writes only
incompatible-* files under ../results/ and ../report/figures/ plus
../report/tables/conv_incompatible_*.tex. It never overwrites the smooth
baseline CSVs or figure PDFs.

Usage (from this directory, project venv):
    python run_incompatible.py

Experiment (declared before the run, not tuned afterwards):
    domain, stencil, closures and corner ownership as in solver.py;
    manufactured data from mms_incompatible.py;
    N = 16, 32, 64, 128, 256, 512 (the main-experiment set plus one finer level);
    Neumann (β=0) and Robin (β=2), both closures;
    global L2_h and nodal L∞ as in solver.discrete_norms;
    split of the error at the fixed radius r = 1/4 from (1,0)
    (near: r < 1/4; far: r ≥ 1/4);
    location of the nodal maximum error;
    computed residual τ = A u*_h − F of the grid function (not a Taylor
    truncation proof: u* is not C^4);
    scaled residual of the computed solve (algebraic check).

Observed orders are reported as last-pair ratios and as least-squares fits
over (i) all computed levels and (ii) the a-priori window N ≥ 64. No order
is assumed in advance.
"""
from __future__ import annotations
import csv, os, sys, time
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

import plotstyle
from plotstyle import NAVY, BRICK, GREY, DARK, color_legend_text, FS_LEGEND, FS_LABEL
from solver import assemble, solve_direct, discrete_norms, scaled_residual
from mms_incompatible import u_star, f_star, g_star, r_star
import convergence_plot_template as tpl

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "..", "report", "figures")
TAB = os.path.join(HERE, "..", "report", "tables")
os.makedirs(RES, exist_ok=True)
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

NS = [16, 32, 64, 128, 256, 512]
R_SPLIT = 0.25  # fixed geometric split, not tuned
FIT_NMIN = 64   # a-priori finer-mesh window
PROTECTED_FIGS = {
    "convergence.pdf", "cost_accuracy.pdf", "solution_surface.pdf",
    "spy.pdf", "truncation.pdf", "xjtlu_crest.pdf",
}
PROTECTED_CSV = {
    "convergence.csv", "convergence_fit.csv", "cost_accuracy.csv",
    "cost_accuracy_all.csv", "cost_accuracy_log.txt", "d2_extra.csv",
    "d2_extra_log.txt", "mms_checks.csv", "mms_checks_log.txt",
    "solution_surface_log.txt",
}
PROTECTED_TEX = {"conv_Neumann.tex", "conv_Robin.tex", "cost.tex", "trunc.tex"}

plotstyle.setup()
TEXTW = 6.299   # report text width in inches


def _assert_safe_write(path, protected, kind):
    name = os.path.basename(path)
    if name in protected:
        raise RuntimeError(f"refusing to overwrite protected {kind} {path}")


def region_masks(I, J, h, r_split=R_SPLIT):
    r = np.hypot(1.0 - I * h, J * h)
    near = r < r_split
    far = ~near
    return r, near, far


def region_norms(e, I, J, h, r_split=R_SPLIT):
    """Partial L2_h (same trapezoidal weights) and nodal L∞ on each region."""
    r, near, far = region_masks(I, J, h, r_split)
    w = np.where(I == I.max(), 0.5, 1.0)

    def pack(mask):
        if not np.any(mask):
            return 0.0, float("nan"), 0
        L2 = float(np.sqrt(h**2 * np.sum(w[mask] * e[mask]**2)))
        Linf = float(np.abs(e[mask]).max())
        return L2, Linf, int(mask.sum())

    L2n, Lin, nn = pack(near)
    L2f, Lif, nf = pack(far)
    return dict(
        r=r, near=near, far=far,
        L2_near=L2n, Linf_near=Lin, n_near=nn,
        L2_far=L2f, Linf_far=Lif, n_far=nf,
    )


def maxabs(arr, mask=None):
    if mask is None:
        return float(np.abs(arr).max()) if arr.size else float("nan")
    if not np.any(mask):
        return float("nan")
    return float(np.abs(arr[mask]).max())


def run_study():
    rows = []
    t0 = time.perf_counter()
    for beta, lab in [(0.0, "Neumann"), (2.0, "Robin")]:
        for cl in ["first", "ghost"]:
            prev = None
            for N in NS:
                t1 = time.perf_counter()
                A, F, h, I, J = assemble(
                    N, beta, cl, f_star, g_star, lambda y: r_star(y, beta)
                )
                U = solve_direct(A, F)
                Ue = u_star(I * h, J * h)
                e = U - Ue
                L2, Linf, Lb, Li = discrete_norms(e, I, h)
                spl = region_norms(e, I, J, h)
                kmax = int(np.abs(e).argmax())
                xmax = float(I[kmax] * h)
                ymax = float(J[kmax] * h)
                rmax = float(spl["r"][kmax])
                tau = A @ Ue - F
                bmask = I == N
                sres = scaled_residual(A, U, F)
                row = dict(
                    bc=lab, closure=cl, N=int(N), h=h,
                    L2=L2, Linf=Linf,
                    Linf_boundary=Lb, Linf_interior=Li,
                    L2_near=spl["L2_near"], Linf_near=spl["Linf_near"],
                    n_near=spl["n_near"],
                    L2_far=spl["L2_far"], Linf_far=spl["Linf_far"],
                    n_far=spl["n_far"],
                    argmax_x=xmax, argmax_y=ymax, argmax_r=rmax,
                    tau_max=maxabs(tau),
                    tau_near=maxabs(tau, spl["near"]),
                    tau_far=maxabs(tau, spl["far"]),
                    tau_boundary=maxabs(tau, bmask),
                    tau_interior=maxabs(tau, ~bmask),
                    scaled_res=sres,
                    solve_s=time.perf_counter() - t1,
                )
                for key in ("L2", "Linf", "L2_near", "Linf_near", "L2_far", "Linf_far"):
                    pk = "p_" + key
                    if prev is not None and prev[key] > 0 and row[key] > 0:
                        row[pk] = float(np.log2(prev[key] / row[key]))
                    else:
                        row[pk] = float("nan")
                rows.append(row)
                prev = row
                print(
                    f"{lab:7s} {cl:5s} N={N:3d}  "
                    f"L2={L2:.3e} p={row['p_L2']:.2f}  "
                    f"Linf={Linf:.3e} p={row['p_Linf']:.2f}  "
                    f"near={spl['Linf_near']:.3e} p={row['p_Linf_near']:.2f}  "
                    f"far={spl['Linf_far']:.3e} p={row['p_Linf_far']:.2f}  "
                    f"max@({xmax:.4f},{ymax:.4f}) r={rmax:.4f}  "
                    f"sres={sres:.1e}  {row['solve_s']:.2f}s"
                )
    print(f"solves done in {time.perf_counter()-t0:.1f}s")
    return rows


def write_csv(rows):
    path = os.path.join(RES, "incompatible_convergence.csv")
    _assert_safe_write(path, PROTECTED_CSV, "csv")
    fields = list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return path


def _sci(v):
    m, ex = f"{float(v):.3e}".split("e")
    return f"${m}\\times10^{{{int(ex)}}}$"


def _p(v):
    if v is None or (isinstance(v, float) and not np.isfinite(v)) or v in ("nan", "", None):
        return "---"
    return f"{float(v):.2f}"


def write_tables(rows):
    by = {(r["bc"], r["closure"], r["N"]): r for r in rows}
    for bc in ("Neumann", "Robin"):
        # global norms
        L = [
            r"\begin{tabular}{@{}llrcccccc@{}}\toprule",
            r"Closure & $N$ & $h$ & $\|e\|_{L^2_h}$ & $p$ & $\|e\|_{\infty}$ & $p$ & $\|e\|_{\infty,\Gamma_B}$ & $\|e\|_{\infty,\mathrm{int}}$\\\midrule",
        ]
        for cl in ("first", "ghost"):
            for N in NS:
                r = by[(bc, cl, N)]
                lab = "first-order" if cl == "first" else "ghost-point"
                L.append(
                    f"{lab} & {N} & 1/{N} & {_sci(r['L2'])} & {_p(r['p_L2'])} & "
                    f"{_sci(r['Linf'])} & {_p(r['p_Linf'])} & "
                    f"{_sci(r['Linf_boundary'])} & {_sci(r['Linf_interior'])}\\\\"
                )
            L.append(r"\midrule" if cl == "first" else r"\bottomrule")
        L.append(r"\end{tabular}")
        path = os.path.join(TAB, f"conv_incompatible_{bc}.tex")
        _assert_safe_write(path, PROTECTED_TEX, "tex")
        open(path, "w").write("\n".join(L) + "\n")

        # split + location
        L = [
            r"\begin{tabular}{@{}lrcccccc@{}}\toprule",
            r"Closure & $N$ & $\|e\|_{\infty,\mathrm{near}}$ & $p$ & $\|e\|_{\infty,\mathrm{far}}$ & $p$"
            r" & $(x,y)_{\max}$ & $r_{\max}$\\\midrule",
        ]
        for cl in ("first", "ghost"):
            for N in NS:
                r = by[(bc, cl, N)]
                lab = "first-order" if cl == "first" else "ghost-point"
                # In every computed run the nodal max is at the nearest unknown (1,h).
                if abs(r["argmax_x"] - 1.0) < 1e-12 and abs(r["argmax_y"] - r["h"]) < 1e-12:
                    xy = r"$(1,h)$"
                    rmax = r"$h$"
                else:
                    xy = f"$({r['argmax_x']:.4f},{r['argmax_y']:.4f})$"
                    rmax = f"{r['argmax_r']:.4f}"
                L.append(
                    f"{lab} & {N} & {_sci(r['Linf_near'])} & {_p(r['p_Linf_near'])} & "
                    f"{_sci(r['Linf_far'])} & {_p(r['p_Linf_far'])} & "
                    f"{xy} & {rmax}\\\\"
                )
            L.append(r"\midrule" if cl == "first" else r"\bottomrule")
        L.append(r"\end{tabular}")
        path = os.path.join(TAB, f"split_incompatible_{bc}.tex")
        _assert_safe_write(path, PROTECTED_TEX, "tex")
        open(path, "w").write("\n".join(L) + "\n")
    print("tables written to", TAB)


def _sel(rows, **kw):
    return [r for r in rows if all(r[a] == b for a, b in kw.items())]


def fit_block(rows):
    """Least-squares fits; windows are declared a priori (all levels, N≥64)."""
    lines = ["bc,closure,norm,window,n_levels,fitted_order,max_log_dev"]
    recs = []
    for bc in ("Neumann", "Robin"):
        for cl in ("first", "ghost"):
            s = _sel(rows, bc=bc, closure=cl)
            s = sorted(s, key=lambda r: r["N"])
            hs = np.array([r["h"] for r in s], dtype=float)
            Ns = np.array([r["N"] for r in s], dtype=int)
            fine = Ns >= FIT_NMIN
            for nm, key in (
                ("L2", "L2"), ("Linf", "Linf"),
                ("Linf_near", "Linf_near"), ("Linf_far", "Linf_far"),
            ):
                e = np.array([r[key] for r in s], dtype=float)
                for label, mask in (("all levels", np.ones(len(s), bool)),
                                    (f"N>={FIT_NMIN}", fine)):
                    if mask.sum() < 2:
                        continue
                    slope, dev = tpl.fit_observed_order(hs[mask], e[mask])
                    recs.append(dict(bc=bc, closure=cl, norm=nm,
                                     window=label, n=int(mask.sum()),
                                     slope=slope, dev=dev))
                    lines.append(
                        f"{bc},{cl},{nm},{label},{int(mask.sum())},{slope:.4f},{dev:.4f}"
                    )
                    print(f"{bc}/{cl}/{nm} [{label}]: fit {slope:.3f}  max ln-dev {dev:.3f}")
    path = os.path.join(RES, "incompatible_convergence_fit.csv")
    _assert_safe_write(path, PROTECTED_CSV, "csv")
    open(path, "w").write("\n".join(lines) + "\n")
    return recs


def _two_panel_refinement(rows, series, fname, ylabel, title_fmt, fit_head):
    """Shared layout for the two incompatible refinement figures (drawn at text width).
    series: list of (closure, key, linestyle, open_marker, legend label)."""
    fig, ax = plt.subplots(1, 2, figsize=(TEXTW, 3.15), sharey=True)
    for a, bc in zip(ax, ["Neumann", "Robin"]):
        fits = {}
        for cl, key, ls, hollow, _ in series:
            c, m = (BRICK, "o") if cl == "first" else (NAVY, "s")
            s = sorted(_sel(rows, bc=bc, closure=cl), key=lambda r: r["N"])
            hs = np.array([r["h"] for r in s], float)
            e = np.array([r[key] for r in s], float)
            fits.setdefault(cl, []).append(tpl.fit_observed_order(hs, e)[0])
            a.loglog(hs, e, m + ls, color=c, mfc="white" if hollow else c)
        if fname == "incompatible-convergence.pdf":
            # visual guides only (smooth-baseline orders); not claimed here
            for cl, p, col in (("first", 1.0, GREY), ("ghost", 2.0, DARK)):
                s0 = sorted(_sel(rows, bc=bc, closure=cl), key=lambda r: r["N"])
                a.loglog(hs, s0[0]["L2"] * (hs / s0[0]["h"]) ** p, ":", color=col, lw=1.0)
        a.set_xticks(hs)
        a.set_xticklabels([f"1/{int(round(1/v))}" for v in hs])
        a.minorticks_off()
        a.set_title(title_fmt.format(bc=bc, beta=0 if bc == "Neumann" else 2))
        a.set_xlabel("mesh size $h$")
        items = [(BRICK, "first-order: " + " / ".join(f"{v:.2f}" for v in fits["first"])),
                 (NAVY, "ghost: " + " / ".join(f"{v:.2f}" for v in fits["ghost"]))]
        for k, (c, t) in enumerate(items[::-1]):
            a.text(0.97, 0.04 + 0.085 * k, t, transform=a.transAxes,
                   fontsize=FS_LEGEND, ha="right", va="bottom", color=c)
        a.text(0.97, 0.04 + 0.085 * 2, fit_head, transform=a.transAxes,
               fontsize=FS_LEGEND, ha="right", va="bottom", color=DARK)
    ax[0].set_ylabel(ylabel)
    hd, lb = [], []
    for cl, key, ls, hollow, lab in series:
        c, m = (BRICK, "o") if cl == "first" else (NAVY, "s")
        hd.append(Line2D([], [], color=c, marker=m, ls=ls, mfc="white" if hollow else c))
        lb.append(lab)
    if fname == "incompatible-convergence.pdf":
        hd.insert(2, Line2D([], [], color=GREY, ls=":")); lb.insert(2, "guide slope 1")
        hd.append(Line2D([], [], color=DARK, ls=":")); lb.append("guide slope 2")
    color_legend_text(fig.legend(hd, lb, loc="lower center", ncol=3, handlelength=2.2,
                                 columnspacing=1.4, bbox_to_anchor=(0.53, 0.0)))
    fig.subplots_adjust(left=0.095, right=0.99, top=0.92, bottom=0.30, wspace=0.08)
    path = os.path.join(FIG, fname)
    _assert_safe_write(path, PROTECTED_FIGS, "figure")
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


def plot_convergence(rows):
    _two_panel_refinement(
        rows,
        [("first", "L2", "-", True, r"first-order, $L^2_h$"),
         ("first", "Linf", "--", False, r"first-order, $L^\infty$"),
         ("ghost", "L2", "-", True, r"ghost, $L^2_h$"),
         ("ghost", "Linf", "--", False, r"ghost, $L^\infty$")],
        "incompatible-convergence.pdf", "error norm",
        r"{bc} on $x=1$ ($\beta={beta}$)", r"fitted order $L^2_h$ / $L^\infty$")


def plot_split(rows):
    _two_panel_refinement(
        rows,
        [("first", "Linf_near", "-", True, r"first-order, near ($r<1/4$)"),
         ("first", "Linf_far", "--", False, r"first-order, far ($r\geq1/4$)"),
         ("ghost", "Linf_near", "-", True, r"ghost, near ($r<1/4$)"),
         ("ghost", "Linf_far", "--", False, r"ghost, far ($r\geq1/4$)")],
        "incompatible-split.pdf", r"$\|e\|_\infty$ on region",
        r"{bc} ($\beta={beta}$): $L^\infty$ split at $r=1/4$", "fitted order near / far")


def _tex_sci(v):
    m, ex = f"{v:.2e}".split("e")
    return rf"{m}\times10^{{{int(ex)}}}"


def _circle(ax, r=R_SPLIT, n=200, **kw):
    t = np.linspace(0.0, 0.5 * np.pi, n)
    ax.plot(1.0 - r * np.cos(t), r * np.sin(t), **kw)


def plot_fields(rows, N=64):
    # dense plot of u* (independent of the unknown grid)
    ng = 241
    gx = np.linspace(0.0, 1.0, ng)
    gy = np.linspace(0.0, 1.0, ng)
    GX, GY = np.meshgrid(gx, gy, indexing="xy")
    Uex = u_star(GX, GY)

    cases = [
        (0.0, "Neumann", "first"),
        (0.0, "Neumann", "ghost"),
        (2.0, "Robin", "first"),
        (2.0, "Robin", "ghost"),
    ]
    fields = []
    for beta, lab, cl in cases:
        A, F, h, I, J = assemble(N, beta, cl, f_star, g_star, lambda y: r_star(y, beta))
        U = solve_direct(A, F)
        X = (I * h).reshape(N - 1, N)
        Y = (J * h).reshape(N - 1, N)
        E = np.abs(U.reshape(N - 1, N) - u_star(X, Y))
        # argmax in this solve
        k = int(np.abs(U - u_star(I * h, J * h)).argmax())
        fields.append(dict(
            lab=lab, cl=cl, X=X, Y=Y, E=E, h=h,
            mx=float((I * h)[k]), my=float((J * h)[k]),
            emax=float(E.max()),
        ))

    fig, axes = plt.subplots(2, 3, figsize=(TEXTW, 4.3))
    # (0,0): u*
    a0 = axes[0, 0]
    im0 = a0.pcolormesh(GX, GY, Uex, cmap="cividis", shading="auto", rasterized=True)
    _circle(a0, color="white", lw=0.9, ls="--")
    a0.plot(1.0, 0.0, "o", color="white", ms=4, mew=0.6, markeredgecolor="black")
    a0.set_title(r"exact $u^*=\mathrm{Im}(z\,\mathrm{Log}\,z)$", fontsize=FS_LABEL - 1)
    a0.set_xlabel("$x$", fontsize=8)
    a0.set_ylabel("$y$", fontsize=8)
    cb0 = fig.colorbar(im0, ax=a0, fraction=0.046, pad=0.03)
    cb0.ax.tick_params(labelsize=7.5)
    cb0.set_label("$u^*$", fontsize=8)

    # (1,0): zoom of u* near (1,0)
    aZ = axes[1, 0]
    zoom = 0.35
    imz = aZ.pcolormesh(GX, GY, Uex, cmap="cividis", shading="auto", rasterized=True)
    aZ.set_xlim(1.0 - zoom, 1.0)
    aZ.set_ylim(0.0, zoom)
    _circle(aZ, color="white", lw=0.9, ls="--")
    aZ.plot(1.0, 0.0, "o", color="white", ms=4, mew=0.6, markeredgecolor="black")
    aZ.set_title(r"$u^*$ near $(1,0)$", fontsize=FS_LABEL - 1)
    aZ.set_xlabel("$x$", fontsize=8)
    aZ.set_ylabel("$y$", fontsize=8)
    cbz = fig.colorbar(imz, ax=aZ, fraction=0.046, pad=0.03)
    cbz.ax.tick_params(labelsize=7.5)
    cbz.set_label("$u^*$", fontsize=8)

    # error panels: log scale, individual limits (ghost errors are much smaller)
    panel = [(0, 1), (0, 2), (1, 1), (1, 2)]
    for (i, j), f in zip(panel, fields):
        a = axes[i, j]
        E = np.maximum(f["E"], np.nanmin(f["E"][f["E"] > 0]) * 0.5)
        vmin = max(np.nanmin(E), f["emax"] * 1e-3)
        im = a.pcolormesh(
            f["X"], f["Y"], E, cmap="cividis", shading="auto",
            norm=LogNorm(vmin=vmin, vmax=f["emax"]), rasterized=True,
        )
        _circle(a, color="white", lw=0.9, ls="--")
        a.plot(f["mx"], f["my"], marker="*", color="white", ms=8,
               markeredgecolor="black", markeredgewidth=0.6, zorder=5)
        a.plot(1.0, 0.0, "o", color="white", ms=3.5, mew=0.5, markeredgecolor="black")
        cln = "first-order" if f["cl"] == "first" else "ghost-point"
        a.set_title(
            f"{f['lab']}, {cln}, $N={N}$\n"
            rf"$\max|e|={_tex_sci(f['emax'])}$ at $(1,h)$",
            fontsize=FS_LABEL - 1.5,
        )
        a.set_xlabel("$x$", fontsize=8)
        a.set_ylabel("$y$", fontsize=8)
        a.set_xlim(0, 1)
        a.set_ylim(0, 1)
        a.set_aspect("equal")
        cb = fig.colorbar(im, ax=a, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=7.5)
        cb.set_label(r"$|U-u^*|$", fontsize=8)
    for a in axes.ravel():
        a.set_aspect("equal")
        a.tick_params(labelsize=7.5)
        a.set_xlim(a.get_xlim())
    # zoom axes limits already set
    axes[1, 0].set_xlim(1.0 - zoom, 1.0)
    axes[1, 0].set_ylim(0.0, zoom)
    fig.subplots_adjust(left=0.065, right=0.925, top=0.93, bottom=0.08, wspace=0.75, hspace=0.42)
    path = os.path.join(FIG, "incompatible-error-field.pdf")
    _assert_safe_write(path, PROTECTED_FIGS, "figure")
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)
    # also a log of the N=64 maxima
    logp = os.path.join(RES, "incompatible_fields_log.txt")
    _assert_safe_write(logp, PROTECTED_CSV, "csv")
    with open(logp, "w") as fh:
        fh.write(f"error-field figure, N={N}, r_split={R_SPLIT}\n")
        for f in fields:
            fh.write(
                f"{f['lab']} {f['cl']}: max|U-u*|={f['emax']:.6e} at "
                f"(x,y)=({f['mx']:.4f},{f['my']:.4f})\n"
            )


def write_log(rows, recs):
    path = os.path.join(RES, "incompatible_convergence_log.txt")
    _assert_safe_write(path, PROTECTED_CSV, "csv")
    with open(path, "w") as fh:
        fh.write("Incompatible-corner experiment log\n")
        fh.write(f"R_SPLIT={R_SPLIT} (near: r < {R_SPLIT}, far: r >= {R_SPLIT})\n")
        fh.write(f"NS={NS}\n")
        fh.write(f"FIT windows: all levels, and N>={FIT_NMIN} (declared a priori)\n\n")
        for r in rows:
            fh.write(
                f"{r['bc']:7s} {r['closure']:5s} N={r['N']:3d} h={r['h']:.6f}  "
                f"L2={r['L2']:.6e} p_L2={r['p_L2']:.4f}  "
                f"Linf={r['Linf']:.6e} p_Linf={r['p_Linf']:.4f}  "
                f"L2_near={r['L2_near']:.6e} Linf_near={r['Linf_near']:.6e} p={r['p_Linf_near']:.4f}  "
                f"L2_far={r['L2_far']:.6e} Linf_far={r['Linf_far']:.6e} p={r['p_Linf_far']:.4f}  "
                f"n_near={r['n_near']} n_far={r['n_far']}  "
                f"max@({r['argmax_x']:.6f},{r['argmax_y']:.6f}) r={r['argmax_r']:.6f}  "
                f"tau_max={r['tau_max']:.6e} tau_near={r['tau_near']:.6e} tau_far={r['tau_far']:.6e}  "
                f"tau_B={r['tau_boundary']:.6e} tau_I={r['tau_interior']:.6e}  "
                f"scaled_res={r['scaled_res']:.3e}  {r['solve_s']:.3f}s\n"
            )
        fh.write("\nFitted orders\n")
        for rec in recs:
            fh.write(
                f"{rec['bc']:7s} {rec['closure']:5s} {rec['norm']:10s} "
                f"[{rec['window']}] n={rec['n']}  slope={rec['slope']:.4f}  "
                f"max_ln_dev={rec['dev']:.4f}\n"
            )
    print("wrote", path)


def run():
    os.chdir(HERE)
    sys.path.insert(0, HERE)
    rows = run_study()
    write_csv(rows)
    recs = fit_block(rows)
    write_tables(rows)
    plot_convergence(rows)
    plot_split(rows)
    plot_fields(rows, N=64)
    write_log(rows, recs)
    return rows


if __name__ == "__main__":
    run()
