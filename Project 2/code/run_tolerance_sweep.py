"""Tolerance sweep: separate algebraic (solver) error from discretisation error.

Report Section 4.2 / course checklist item "tolerance sweep on one representative mesh".
Conjugate gradients are applied to the symmetrised SPD system DA U = DF of
Proposition 2.x (never to the nonsymmetric A), started from U0 = 0 with the stopping
rule ||DF - DA U_k||_2 <= tol ||DF||_2 (SciPy rtol, atol = 0). Deterministic: no random
start vectors.

Part 1 (sweep): Robin (beta = 2), N = 64, both closures, tol = 1e-4, ..., 1e-12.
  For each run: iterations, relative residual of the scaled system, componentwise
  scaled residual rho of A U = F (eq. sres), ||U_CG - U_direct||_inf and the true
  errors ||U_CG - u*||_inf, ||U_CG - u*||_{L2_h}.
Part 2 (order check): the same CG at tol = 1e-8 and 1e-10 on N = 16, ..., 256;
  observed orders are compared with those of the sparse direct solve.

Outputs: ../results/tolerance_sweep.csv, ../results/tolerance_order_check.csv,
         ../results/tolerance_sweep_log.txt, ../report/tables/tol_sweep.tex,
         ../report/tables/tol_order.tex
"""
from __future__ import annotations
import csv, os
import numpy as np
from solver import (assemble, solve_direct, solve_cg, symmetrise, scaled_residual,
                    discrete_norms)
from mms import u_star, f_star, g_star, r_star
import convergence_plot_template as tpl

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
TAB = os.path.join(HERE, "..", "report", "tables")

BETA, LABEL = 2.0, "Robin"
N_SWEEP = 64
TOLS = [1e-4, 1e-6, 1e-8, 1e-10, 1e-12]
NS_ORDER = [16, 32, 64, 128, 256]
TOLS_ORDER = [1e-8, 1e-10]
CLOSURES = ["ghost", "first"]


def _system(N, cl):
    A, F, h, I, J = assemble(N, BETA, cl, f_star, g_star, lambda y: r_star(y, BETA))
    S, G, _ = symmetrise(A, F, I, h, cl)
    return A, F, S, G, h, I, u_star(I * h, J * h)


def sweep():
    rows = []
    for cl in CLOSURES:
        A, F, S, G, h, I, Ue = _system(N_SWEEP, cl)
        Ud = solve_direct(A, F)
        L2d, Linfd, _, _ = discrete_norms(Ud - Ue, I, h)
        for tol in TOLS:
            x, iters, relres, info = solve_cg(S, G, tol=tol)
            L2, Linf, _, _ = discrete_norms(x - Ue, I, h)
            rows.append(dict(
                bc=LABEL, closure=cl, N=N_SWEEP, tol=tol, info=info, iters=iters,
                relres_DA=relres, scaled_res=scaled_residual(A, x, F),
                diff_direct_inf=float(np.abs(x - Ud).max()),
                err_inf=Linf, err_L2=L2, direct_err_inf=Linfd, direct_err_L2=L2d,
            ))
    return rows


def order_check():
    rows = []
    for cl in CLOSURES:
        for N in NS_ORDER:
            A, F, S, G, h, I, Ue = _system(N, cl)
            Ud = solve_direct(A, F)
            L2, Linf, _, _ = discrete_norms(Ud - Ue, I, h)
            rows.append(dict(bc=LABEL, closure=cl, N=N, h=h, solver="direct", tol="",
                             iters="", err_inf=Linf, err_L2=L2, diff_direct_inf=0.0))
            for tol in TOLS_ORDER:
                x, iters, relres, info = solve_cg(S, G, tol=tol)
                L2, Linf, _, _ = discrete_norms(x - Ue, I, h)
                rows.append(dict(bc=LABEL, closure=cl, N=N, h=h, solver="cg", tol=tol,
                                 iters=iters, err_inf=Linf, err_L2=L2,
                                 diff_direct_inf=float(np.abs(x - Ud).max())))
    return rows


def fits(rows):
    out = []
    for cl in CLOSURES:
        for solver, tol in [("direct", "")] + [("cg", t) for t in TOLS_ORDER]:
            s = [r for r in rows if r["closure"] == cl and r["solver"] == solver and r["tol"] == tol]
            s.sort(key=lambda r: r["N"])
            hs = np.array([r["h"] for r in s])
            ei = np.array([r["err_inf"] for r in s])
            e2 = np.array([r["err_L2"] for r in s])
            out.append(dict(
                closure=cl, solver=solver, tol=tol,
                fit_inf=tpl.fit_observed_order(hs, ei)[0],
                fit_L2=tpl.fit_observed_order(hs, e2)[0],
                last_inf=float(np.log2(ei[-2] / ei[-1])),
                last_L2=float(np.log2(e2[-2] / e2[-1])),
                err_inf_finest=float(ei[-1]),
                iters_finest=s[-1]["iters"],
            ))
    return out


def _sci(v, d=2):
    m, ex = f"{float(v):.{d}e}".split("e")
    return f"${m}\\times10^{{{int(ex)}}}$"


def _tol(t):
    return f"$10^{{{int(round(np.log10(t)))}}}$"


def write_tables(rows, frows):
    os.makedirs(TAB, exist_ok=True)
    L = [r"\begin{tabular}{@{}llrcccc@{}}\toprule",
         r"Closure & tol & iters & $\|DF-DAU\|_2/\|DF\|_2$ & $\rho$ & "
         r"$\|U_{\mathrm{CG}}-U_{\mathrm{dir}}\|_\infty$ & $\|U_{\mathrm{CG}}-u^*\|_\infty$\\\midrule"]
    for cl in CLOSURES:
        s = [r for r in rows if r["closure"] == cl]
        lab = "ghost-point" if cl == "ghost" else "first-order"
        for r in s:
            L.append(f"{lab} & {_tol(r['tol'])} & {r['iters']} & {_sci(r['relres_DA'])} & "
                     f"{_sci(r['scaled_res'])} & {_sci(r['diff_direct_inf'])} & "
                     f"{_sci(r['err_inf'], 3)}\\\\")
        L.append(f"\\multicolumn{{6}}{{@{{}}l}}{{\\quad direct \\texttt{{spsolve}}, same mesh}} & "
                 f"{_sci(s[0]['direct_err_inf'], 3)}\\\\")
        L.append(r"\midrule" if cl == CLOSURES[0] else r"\bottomrule")
    L.append(r"\end{tabular}")
    open(os.path.join(TAB, "tol_sweep.tex"), "w").write("\n".join(L) + "\n")

    L = [r"\begin{tabular}{@{}llcccc@{}}\toprule",
         r"Closure & solver & $\|e\|_\infty$ at $N=256$ & last-pair $p_\infty$ & "
         r"fit $p_\infty$ & fit $p_{L^2_h}$\\\midrule"]
    for cl in CLOSURES:
        lab = "ghost-point" if cl == "ghost" else "first-order"
        for f in [f for f in frows if f["closure"] == cl]:
            sv = r"direct \texttt{spsolve}" if f["solver"] == "direct" else \
                f"CG, tol {_tol(f['tol'])} ({f['iters_finest']} it.)"
            L.append(f"{lab} & {sv} & {_sci(f['err_inf_finest'], 3)} & {f['last_inf']:.2f} & "
                     f"{f['fit_inf']:.2f} & {f['fit_L2']:.2f}\\\\")
        L.append(r"\midrule" if cl == CLOSURES[0] else r"\bottomrule")
    L.append(r"\end{tabular}")
    open(os.path.join(TAB, "tol_order.tex"), "w").write("\n".join(L) + "\n")


def _write_csv(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def run():
    os.makedirs(RES, exist_ok=True)
    rows = sweep()
    orows = order_check()
    frows = fits(orows)
    _write_csv(os.path.join(RES, "tolerance_sweep.csv"), rows)
    _write_csv(os.path.join(RES, "tolerance_order_check.csv"), orows)
    _write_csv(os.path.join(RES, "tolerance_order_fit.csv"), frows)
    with open(os.path.join(RES, "tolerance_sweep_log.txt"), "w") as log:
        def P(s):
            print(s)
            print(s, file=log)
        P(f"CG on DA U = DF, {LABEL} beta={BETA}, x0=0, rtol=tol, atol=0")
        for r in rows:
            P(f"{r['closure']:5s} N={r['N']} tol={r['tol']:.0e} info={r['info']} it={r['iters']:4d} "
              f"relres={r['relres_DA']:.3e} rho={r['scaled_res']:.3e} "
              f"|cg-dir|={r['diff_direct_inf']:.3e} |cg-u*|_inf={r['err_inf']:.6e} "
              f"(direct {r['direct_err_inf']:.6e})")
        for f in frows:
            P(f"order {f['closure']:5s} {f['solver']:6s} tol={f['tol']!s:6s} "
              f"fit_inf={f['fit_inf']:.4f} fit_L2={f['fit_L2']:.4f} last_inf={f['last_inf']:.4f} "
              f"last_L2={f['last_L2']:.4f} err_inf(N=256)={f['err_inf_finest']:.6e}")
    write_tables(rows, frows)
    return rows, orows, frows


if __name__ == "__main__":
    run()
