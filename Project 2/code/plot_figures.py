"""Draw the report figures from the CSV results (writes only to ../report/figures/).

Every figure is drawn at its printed width (text width 6.30 in, or 0.48 of it), so the
font sizes set in plotstyle.py are the sizes on the page (ticks/legends 8.5 pt, labels
9.5 pt). Vector PDFs; any rasterised layer is written at 300 dpi.
"""
from __future__ import annotations
import csv, os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import plotstyle
from plotstyle import NAVY, BRICK, SAGE, GREY, DARK, color_legend_text
from plotstyle import FS_TICK, FS_LEGEND, FS_LABEL
from solver import assemble, solve_direct
from mms import u_star, f_star, g_star, r_star

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "..", "report", "figures")
os.makedirs(FIG, exist_ok=True)
plotstyle.setup()
TEXTW = 6.299   # report text width in inches (A4, 2.5 cm margins)


class T:
    def __init__(self, rows):
        self.r = rows

    def __getattr__(self, k):
        return np.array([x[k] for x in self.r])

    def sel(self, **kw):
        return T([x for x in self.r if all(x[a] == b for a, b in kw.items())])


def load_csv(path, float_keys=None):
    rows = []
    with open(path) as fh:
        for x in csv.DictReader(fh):
            y = {}
            for k, v in x.items():
                if k in ("bc", "closure", "counted", "not_counted") or v in ("", None):
                    y[k] = v
                else:
                    try:
                        y[k] = float(v) if ("." in v or "e" in v.lower() or k in (float_keys or ())) else (
                            int(v) if v.lstrip("-").isdigit() else v
                        )
                    except ValueError:
                        y[k] = v
            # force numeric where expected
            for k in ("N", "h", "L2", "Linf", "Linf_boundary", "Linf_interior",
                      "tauB", "tauI", "etaA", "etaS", "scaled_res", "t_spsolve_s", "M", "p_L2", "p_Linf"):
                if k in y and y[k] not in (None, "", "nan"):
                    try:
                        y[k] = float(y[k])
                        if k in ("N", "M") and y[k] == int(y[k]):
                            y[k] = int(y[k])
                    except (TypeError, ValueError):
                        pass
            rows.append(y)
    return T(rows)


def _h_ticks(a, hs):
    a.set_xticks(hs)
    a.set_xticklabels([f"1/{int(round(1/v))}" for v in hs])
    a.minorticks_off()


def plot_convergence():
    """Refinement log-log following convergence_plot_template.py: measured errors,
    reference slopes, and fitted orders (template's fit_observed_order) annotated."""
    import convergence_plot_template as tpl
    d = load_csv(os.path.join(RES, "convergence.csv"))
    fig, ax = plt.subplots(1, 2, figsize=(TEXTW, 3.15), sharey=True)
    lines = ["bc,closure,norm,expected,fitted_order,max_log_dev,window"]
    for a, bc in zip(ax, ["Neumann", "Robin"]):
        fit_txt = []
        for cl, c, m, p in [("first", BRICK, "o", 1.0), ("ghost", NAVY, "s", 2.0)]:
            s = d.sel(bc=bc, closure=cl)
            hs, e2, ei = s.h.astype(float), s.L2.astype(float), s.Linf.astype(float)
            for nm, e in (("L2", e2), ("Linf", ei)):
                slope, dev = tpl.fit_observed_order(hs, e)
                lines.append(f"{bc},{cl},{nm},{p:g},{slope:.4f},{dev:.4f},all levels")
                print(f"{bc}/{cl}/{nm}:"); tpl.refinement_table(hs, e, p)
            sl2, _ = tpl.fit_observed_order(hs, e2)
            sli, _ = tpl.fit_observed_order(hs, ei)
            a.loglog(hs, e2, m + "-", color=c, mfc="white")
            a.loglog(hs, ei, m + "--", color=c)
            ref = e2[0] * (hs / hs[0]) ** p * 0.5
            a.loglog(hs, ref, ":", color=GREY if p == 1 else DARK, lw=1.0)
            fit_txt.append((c, f"{'first-order' if cl == 'first' else 'ghost'}: {sl2:.2f} / {sli:.2f}"))
        _h_ticks(a, hs)
        a.set_ylim(1e-7, 2e-1)
        a.set_title(fr"{bc} on $x=1$ ($\beta={0 if bc == 'Neumann' else 2}$)")
        a.set_xlabel("mesh size $h$")
        for k, (c, t) in enumerate(fit_txt[::-1]):
            a.text(0.97, 0.04 + 0.085 * k, t, transform=a.transAxes,
                   fontsize=FS_LEGEND, ha="right", va="bottom", color=c)
        a.text(0.97, 0.04 + 0.085 * 2, r"fitted order $L^2_h$ / $L^\infty$", transform=a.transAxes,
               fontsize=FS_LEGEND, ha="right", va="bottom", color=DARK)
    ax[0].set_ylabel("error norm")
    hd = [Line2D([], [], color=BRICK, marker="o", mfc="white", ls="-"),
          Line2D([], [], color=BRICK, marker="o", ls="--"),
          Line2D([], [], color=GREY, ls=":"),
          Line2D([], [], color=NAVY, marker="s", mfc="white", ls="-"),
          Line2D([], [], color=NAVY, marker="s", ls="--"),
          Line2D([], [], color=DARK, ls=":")]
    lb = [r"first-order, $L^2_h$", r"first-order, $L^\infty$", "reference slope 1",
          r"ghost, $L^2_h$", r"ghost, $L^\infty$", "reference slope 2"]
    color_legend_text(fig.legend(hd, lb, loc="lower center", ncol=3, handlelength=2.2,
                                 columnspacing=1.4, bbox_to_anchor=(0.53, 0.0)))
    fig.subplots_adjust(left=0.095, right=0.99, top=0.92, bottom=0.30, wspace=0.08)
    fig.savefig(os.path.join(FIG, "convergence.pdf"))
    plt.close(fig)
    with open(os.path.join(RES, "convergence_fit.csv"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


def plot_truncation():
    t = load_csv(os.path.join(RES, "mms_checks.csv"))
    fig, a = plt.subplots(figsize=(0.48 * TEXTW, 2.55))
    for cl, c, m in [("first", BRICK, "o"), ("ghost", NAVY, "s")]:
        s = t.sel(bc="Robin", closure=cl)
        a.loglog(s.h, s.tauB, m + "-", color=c, label=f"boundary rows, {cl}")
    s = t.sel(bc="Robin", closure="ghost")
    a.loglog(s.h, s.tauI, "^-", color=SAGE, label="interior rows")
    _h_ticks(a, s.h)
    a.set_xlabel("$h$")
    a.set_ylabel(r"$\max|\tau|$")
    a.set_ylim(1.5e-6, 0.4)
    color_legend_text(a.legend(loc="lower right", handlelength=1.6, borderaxespad=0.1))
    fig.subplots_adjust(left=0.22, right=0.97, top=0.97, bottom=0.17)
    fig.savefig(os.path.join(FIG, "truncation.pdf"))
    plt.close(fig)


def plot_spy():
    """Left: the N=8 grid (which nodes are unknowns, where the ghost points sit).
    Centre/right: sparsity patterns of A for the same grid, both closures."""
    N = 8
    h = 1.0 / N
    fig = plt.figure(figsize=(TEXTW, 2.75))
    ag = fig.add_axes([0.065, 0.30, 0.25, 0.58])
    xs = np.arange(N + 1) * h
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    dir_mask = (X == 0) | (Y == 0) | (Y == 1)
    edge_mask = (np.isclose(X, 1)) & ~dir_mask
    int_mask = ~dir_mask & ~edge_mask
    ag.plot(X[dir_mask], Y[dir_mask], "s", ms=3.6, mfc="white", mec=GREY, mew=0.8, ls="none")
    ag.plot(X[int_mask], Y[int_mask], "o", ms=2.6, color=DARK, ls="none")
    ag.plot(X[edge_mask], Y[edge_mask], "D", ms=3.4, color=NAVY, ls="none")
    yg = np.arange(1, N) * h
    ag.plot(np.full_like(yg, 1 + h), yg, "x", ms=3.4, color=BRICK, mew=0.9, ls="none")
    for v in xs:
        ag.plot([v, v], [0, 1], color="#d0d0d0", lw=0.4, zorder=0)
        ag.plot([0, 1], [v, v], color="#d0d0d0", lw=0.4, zorder=0)
    ag.set_xlim(-0.06, 1 + 1.9 * h); ag.set_ylim(-0.06, 1.06)
    ag.set_aspect("equal")
    ag.set_xticks([0, 0.5, 1]); ag.set_yticks([0, 0.5, 1])
    ag.set_xlabel("$x$", labelpad=1); ag.set_ylabel("$y$", labelpad=1)
    ag.set_title(f"grid, $N={N}$, $h=1/{N}$", fontsize=FS_LABEL)
    hd = [Line2D([], [], marker="s", mfc="white", mec=GREY, ls="none", ms=4),
          Line2D([], [], marker="o", color=DARK, ls="none", ms=3.2),
          Line2D([], [], marker="D", color=NAVY, ls="none", ms=3.8),
          Line2D([], [], marker="x", color=BRICK, ls="none", ms=4, mew=0.9)]
    lb = ["Dirichlet (data)", "interior unknown", r"unknown on $\Gamma_B$",
          "ghost (eliminated)"]
    leg = fig.legend(hd, lb, loc="lower left", ncol=2, fontsize=FS_LEGEND - 0.5,
                     handletextpad=0.2, columnspacing=0.6, bbox_to_anchor=(0.0, -0.02))
    leg.get_texts()[2].set_color(NAVY); leg.get_texts()[3].set_color(BRICK)
    for k, (cl, c, t) in enumerate([("ghost", NAVY, "ghost closure"),
                                    ("first", BRICK, "first-order closure")]):
        a = fig.add_axes([0.40 + 0.31 * k, 0.12, 0.26, 0.70])
        A, F, hh, I, J = assemble(N, 2.0, cl, f_star, g_star, lambda y: r_star(y, 2.0))
        a.spy(A, markersize=1.6, color=c)
        n0, n1 = A.shape
        a.set_xlim(-1.5, n1 + 0.5); a.set_ylim(n0 + 0.5, -1.5)
        a.set_xticks([0, 20, 40]); a.set_yticks([0, 20, 40])
        a.set_title(f"{t}\n{A.shape[0]} unknowns, nnz $={A.nnz}$", fontsize=FS_LABEL, color=c)
    fig.savefig(os.path.join(FIG, "spy.pdf"))
    plt.close(fig)


def plot_solution_surface(N=64, beta=2.0):
    """Exact u*, numerical U and |U-u*| from a real ghost Robin solve, with colour bars.
    Arrays follow solver.assemble: meshgrid(Jg, Ig, indexing="ij"), so axis 0 is y (j)
    and axis 1 is x (i)."""
    from matplotlib import cm, colors
    A, F, h, I, J = assemble(N, beta, "ghost", f_star, g_star, lambda y: r_star(y, beta))
    U = solve_direct(A, F)
    X, Y = (I * h).reshape(N - 1, N), (J * h).reshape(N - 1, N)
    Ue = u_star(X, Y); Un = U.reshape(N - 1, N); E = np.abs(Un - Ue)
    fig = plt.figure(figsize=(TEXTW, 2.55))
    unorm = colors.Normalize(vmin=min(Ue.min(), Un.min()), vmax=max(Ue.max(), Un.max()))
    enorm = colors.Normalize(vmin=0.0, vmax=E.max() * 1e5)
    panels = [(Ue, r"exact $u^*=e^{x+y}$", "viridis", unorm),
              (Un, f"numerical $U$, $N={N}$", "viridis", unorm),
              (E * 1e5, r"error $|U-u^*|$", "cividis", enorm)]
    lefts = [0.0, 0.25, 0.615]
    for k, (Z, t, cmap, nrm) in enumerate(panels):
        a = fig.add_axes([lefts[k], 0.02, 0.30, 0.90], projection="3d")
        a.plot_surface(X, Y, Z, cmap=cmap, norm=nrm, linewidth=0, antialiased=True,
                       rstride=2, cstride=2, rasterized=True)
        a.set_title(t, fontsize=FS_LABEL, pad=-2)
        a.set_xlabel("$x$", fontsize=FS_LABEL, labelpad=-7)
        a.set_ylabel("$y$", fontsize=FS_LABEL, labelpad=-7)
        a.tick_params(labelsize=7.5, pad=-3)
        a.set_xticks([0, 0.5, 1]); a.set_yticks([0, 0.5, 1])
        a.set_zticklabels([])   # values are read from the colour bars
        a.set_box_aspect((1, 1, 0.8))
        a.view_init(elev=28, azim=-125)
    cax1 = fig.add_axes([0.565, 0.22, 0.012, 0.56])
    cb1 = fig.colorbar(cm.ScalarMappable(norm=unorm, cmap="viridis"), cax=cax1)
    cb1.set_label("$u^*$ and $U$", fontsize=FS_TICK, labelpad=2)
    cb1.ax.tick_params(labelsize=7.5)
    cax2 = fig.add_axes([0.91, 0.22, 0.012, 0.56])
    cb2 = fig.colorbar(cm.ScalarMappable(norm=enorm, cmap="cividis"), cax=cax2)
    cb2.set_label(r"$|U-u^*|\ \times10^{5}$", fontsize=FS_TICK, labelpad=2)
    cb2.ax.tick_params(labelsize=7.5)
    fig.savefig(os.path.join(FIG, "solution_surface.pdf"), dpi=300)
    plt.close(fig)
    with open(os.path.join(RES, "solution_surface_log.txt"), "w") as fh:
        fh.write(f"ghost Robin beta={beta} N={N}: max|U-u*|={E.max():.6e} at "
                 f"(x,y)=({X.flat[E.argmax()]:.4f},{Y.flat[E.argmax()]:.4f})\n")


def plot_cost(tol=5e-3):
    path = os.path.join(RES, "cost_accuracy_all.csv")
    if not os.path.isfile(path):
        print("no cost_accuracy_all.csv — skip cost figure")
        return
    rows = list(csv.DictReader(open(path)))
    fig, ax = plt.subplots(1, 2, figsize=(TEXTW, 2.95), sharey=True)
    for a, bc in zip(ax, ["Neumann", "Robin"]):
        for cl, c, m in [("first", BRICK, "o"), ("ghost", NAVY, "s")]:
            s = [r for r in rows if r["bc"] == bc and r["closure"] == cl]
            t = np.array([float(r["t_spsolve_s"]) for r in s])
            e = np.array([float(r["Linf"]) for r in s])
            lo = t - np.array([float(r["t_min_s"]) for r in s])
            hi = np.array([float(r["t_max_s"]) for r in s]) - t
            a.errorbar(t, e, xerr=[lo, hi], fmt=m + "-", color=c, ecolor=c, elinewidth=0.8,
                       capsize=2.0, mfc="white" if cl == "first" else c)
        a.axhline(tol, color=DARK, ls=":", lw=1.0)
        a.set_xscale("log"); a.set_yscale("log")
        a.set_title(fr"{bc} ($\beta={0 if bc == 'Neumann' else 2}$), $N=16,\dots,512$")
        a.set_xlabel("median spsolve time (s)")
    ax[0].set_ylabel(r"$\|e\|_\infty$")
    ylo, yhi = ax[0].get_ylim()
    ax[0].set_ylim(ylo, yhi * 3)
    hd = [Line2D([], [], color=BRICK, marker="o", mfc="white"),
          Line2D([], [], color=NAVY, marker="s"),
          Line2D([], [], color=DARK, ls=":")]
    lb = ["first-order closure", "ghost closure",
          rf"matched target $\|e\|_\infty\leq{tol*1e3:g}\times10^{{-3}}$"]
    color_legend_text(fig.legend(hd, lb, loc="lower center", ncol=3, handlelength=2.0,
                                 columnspacing=1.6, bbox_to_anchor=(0.53, 0.0)))
    fig.subplots_adjust(left=0.10, right=0.99, top=0.91, bottom=0.27, wspace=0.08)
    fig.savefig(os.path.join(FIG, "cost_accuracy.pdf"))
    plt.close(fig)
    print("wrote", os.path.join(FIG, "cost_accuracy.pdf"))


def main():
    plot_convergence()
    plot_truncation()
    plot_spy()
    plot_solution_surface()
    plot_cost()
    print("figures written to", FIG, "(slides/figures is never touched)")


if __name__ == "__main__":
    main()
