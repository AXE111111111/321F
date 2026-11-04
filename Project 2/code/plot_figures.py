"""Draw Palatino-style figures for the report/slides from CSV results."""
from __future__ import annotations
import csv, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager as fm, pyplot as plt
from solver import assemble, solve_direct
from mms import u_star, f_star, g_star, r_star

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "..", "report", "figures")
os.makedirs(FIG, exist_ok=True)

for f in (
    glob.glob("/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyrepagella-*.otf")
    + glob.glob("/usr/share/texlive/texmf-dist/fonts/opentype/public/tex-gyre/texgyrepagella-*.otf")
    + glob.glob(os.path.expanduser("~/.local/share/fonts/texgyrepagella-*.otf"))
):
    fm.fontManager.addfont(f)
matplotlib.rcParams.update({
    "font.family": "serif", "font.serif": ["TeX Gyre Pagella"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "TeX Gyre Pagella", "mathtext.it": "TeX Gyre Pagella:italic",
    "mathtext.bf": "TeX Gyre Pagella:bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "axes.linewidth": 0.6, "lines.linewidth": 1.0,
    "lines.markersize": 4.5, "legend.frameon": False, "pdf.fonttype": 42,
})
# Okabe-Ito based, colorblind-safe; text colours always copied from artists
NAVY, BRICK, SAGE, GREY = "#0072B2", "#D55E00", "#009E73", "#6e6e6e"
DARK = "#333333"


def color_legend_text(leg):
    """Colour each legend label exactly like its handle."""
    for h, t in zip(leg.legend_handles, leg.get_texts()):
        c = h.get_color() if hasattr(h, "get_color") else h.get_edgecolor()
        t.set_color(c)
    return leg


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


def plot_convergence():
    """Refinement log-log following convergence_plot_template.py:
    measured error, reference slope of the expected order, and the fitted
    order with its max log-deviation (template's fit_observed_order)."""
    import convergence_plot_template as tpl
    d = load_csv(os.path.join(RES, "convergence.csv"))
    fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.45), sharey=True)
    lines = ["bc,closure,norm,expected,fitted_order,max_log_dev,window"]
    for a, bc in zip(ax, ["Neumann", "Robin"]):
        txt = []
        for cl, c, m, p in [("first", BRICK, "o", 1.0), ("ghost", NAVY, "s", 2.0)]:
            s = d.sel(bc=bc, closure=cl)
            hs, e2, ei = s.h.astype(float), s.L2.astype(float), s.Linf.astype(float)
            for nm, e in (("L2", e2), ("Linf", ei)):
                slope, dev = tpl.fit_observed_order(hs, e)
                lines.append(f"{bc},{cl},{nm},{p:g},{slope:.4f},{dev:.4f},all levels")
                print(f"{bc}/{cl}/{nm}:"); tpl.refinement_table(hs, e, p)
            sl2, dv2 = tpl.fit_observed_order(hs, e2)
            sli, _ = tpl.fit_observed_order(hs, ei)
            a.loglog(hs, e2, m + "-", color=c, mfc="white", label=f"{cl}, $L^2_h$ (fit {sl2:.2f})")
            a.loglog(hs, ei, m + "--", color=c, label=f"{cl}, $L^\\infty$ (fit {sli:.2f})")
            ref = e2[0] * (hs / hs[0]) ** p * 0.5
            a.loglog(hs, ref, ":", color=GREY if p == 1 else DARK, lw=1.0,
                     label=f"ref. slope {p:g}")
        a.set_xticks(hs)
        a.set_xticklabels([f"1/{int(round(1/v))}" for v in hs])
        a.minorticks_off()
        a.set_ylim(bottom=1e-7)
        a.set_title(f"{bc} on $x=1$", fontsize=10.5)
        a.set_xlabel("mesh size $h$", fontsize=9.5)
        a.tick_params(labelsize=8.5)
    ax[0].set_ylabel("error norm", fontsize=9.5)
    hd, lb = ax[0].get_legend_handles_labels()
    for a in ax:
        h2, l2 = a.get_legend_handles_labels()
        color_legend_text(a.legend(h2, l2, loc="upper center", ncol=3, fontsize=7.4,
                                   handlelength=1.9, columnspacing=0.6, handletextpad=0.3,
                                   bbox_to_anchor=(0.5, -0.17)))
    fig.subplots_adjust(left=0.07, right=0.995, top=0.90, bottom=0.30, wspace=0.18)
    fig.savefig(os.path.join(FIG, "convergence.pdf"))
    plt.close(fig)
    with open(os.path.join(RES, "convergence_fit.csv"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


def plot_truncation():
    t = load_csv(os.path.join(RES, "mms_checks.csv"))
    fig, a = plt.subplots(figsize=(3.6, 2.85))
    for cl, c, m in [("first", BRICK, "o"), ("ghost", NAVY, "s")]:
        s = t.sel(bc="Robin", closure=cl)
        a.loglog(s.h, s.tauB, m + "-", color=c, label=f"boundary rows, {cl}")
    s = t.sel(bc="Robin", closure="ghost")
    a.loglog(s.h, s.tauI, "^-", color=SAGE, label="interior rows")
    a.set_xticks(s.h)
    a.set_xticklabels([f"1/{int(round(1/v))}" for v in s.h])
    a.minorticks_off()
    a.set_xlabel("$h$")
    a.set_ylabel(r"$\max|\tau|$")
    color_legend_text(a.legend(fontsize=7))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "truncation.pdf"))
    plt.close(fig)


def plot_spy():
    fig, ax = plt.subplots(1, 2, figsize=(4.5, 2.55))
    for a, cl, c, t in [(ax[0], "ghost", NAVY, "ghost closure"), (ax[1], "first", BRICK, "first-order closure")]:
        A, F, h, I, J = assemble(8, 2.0, cl, f_star, g_star, lambda y: r_star(y, 2.0))
        a.spy(A, markersize=1.9, color=c)
        n0, n1 = A.shape
        a.set_xlim(-1.5, n1 + 0.5); a.set_ylim(n0 + 0.5, -1.5)  # keep markers off the frame
        a.set_title(f"{t}, $N=8$\n{A.shape[0]} unknowns, nnz $={A.nnz}$", fontsize=9.5, color=c)
        a.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "spy.pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_solution_surface(N=64, beta=2.0):
    """Exact u*, numerical U and |U-u*| from a real ghost Robin solve."""
    A, F, h, I, J = assemble(N, beta, "ghost", f_star, g_star, lambda y: r_star(y, beta))
    U = solve_direct(A, F)
    X, Y = (I * h).reshape(N - 1, N), (J * h).reshape(N - 1, N)
    Ue = u_star(X, Y); Un = U.reshape(N - 1, N); E = np.abs(Un - Ue)
    fig = plt.figure(figsize=(6.8, 2.5))
    for k, (Z, t, cm) in enumerate([(Ue, "exact $u^*=e^{x+y}$", "Greys"),
                                    (Un, f"numerical $U$, $N={N}$", "Greys"),
                                    (E * 1e5, r"error $|U-u^*|$ ($\times10^{-5}$)", "cividis")]):
        a = fig.add_subplot(1, 3, k + 1, projection="3d")
        a.plot_surface(X, Y, Z, cmap=cm, linewidth=0, antialiased=True,
                       rstride=2, cstride=2, alpha=0.95)
        a.set_title(t, fontsize=8)
        a.set_xlabel("$x$", fontsize=7, labelpad=-8); a.set_ylabel("$y$", fontsize=7, labelpad=-8)
        a.tick_params(labelsize=5.5, pad=-3)
        a.view_init(elev=28, azim=-125)
    fig.subplots_adjust(left=0.0, right=0.97, wspace=0.08)
    fig.savefig(os.path.join(FIG, "solution_surface.pdf"), bbox_inches="tight")
    plt.close(fig)
    with open(os.path.join(RES, "solution_surface_log.txt"), "w") as fh:
        fh.write(f"ghost Robin beta={beta} N={N}: max|U-u*|={E.max():.6e} at "
                 f"(x,y)=({X.flat[E.argmax()]:.4f},{Y.flat[E.argmax()]:.4f})\n")


def plot_cost():
    path = os.path.join(RES, "cost_accuracy_all.csv")
    if not os.path.isfile(path):
        print("no cost_accuracy_all.csv — skip cost figure")
        return
    d = load_csv(path)
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.15), sharey=True)
    for a, bc in zip(ax, ["Neumann", "Robin"]):
        for cl, c, m in [("first", BRICK, "o"), ("ghost", NAVY, "s")]:
            s = d.sel(bc=bc, closure=cl)
            a.loglog(s.t_spsolve_s, s.Linf, m + "-", color=c, label=cl)
        a.set_title(f"{bc}", fontsize=10)
        a.axhline(1e-3, color=GREY, ls=":", lw=0.9, label=r"reference $10^{-3}$")
        a.set_xlabel("spsolve time (s)", fontsize=9)
        a.tick_params(labelsize=8)
    ax[0].set_ylabel(r"$\|e\|_\infty$", fontsize=9)
    hd, lb = ax[0].get_legend_handles_labels()
    color_legend_text(fig.legend(hd, lb, loc="lower center", ncol=3, fontsize=8.0,
               handlelength=1.9, columnspacing=1.1, borderaxespad=0.15,
               bbox_to_anchor=(0.5, -0.005)))
    fig.subplots_adjust(left=0.065, right=0.995, top=0.90, bottom=0.18, wspace=0.14)
    fig.savefig(os.path.join(FIG, "cost_accuracy.pdf"))
    plt.close(fig)
    print("wrote", os.path.join(FIG, "cost_accuracy.pdf"))


def main():
    plot_convergence()
    plot_truncation()
    plot_spy()
    plot_solution_surface()
    plot_cost()
    # copy to slides/figures
    import shutil
    slid = os.path.join(HERE, "..", "slides", "figures")
    os.makedirs(slid, exist_ok=True)
    for name in ("convergence.pdf", "truncation.pdf", "spy.pdf", "solution_surface.pdf", "cost_accuracy.pdf"):
        src = os.path.join(FIG, name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(slid, name))
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
