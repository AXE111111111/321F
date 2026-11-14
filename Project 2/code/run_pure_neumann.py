"""Optional depth: pure-Neumann problem (all four edges ∂_n u = q), both closures.

Writes results/pure_neumann_{nullspace,defect,convergence,fit,uncorrected}.csv,
results/pure_neumann_log.txt, report/tables/pure_neumann.tex and pure_neumann_stats.tex.
Nothing is random; no timings are used.
"""
from __future__ import annotations
import csv, math, os, warnings
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import neumann_pure as P

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
TAB = os.path.join(HERE, "..", "report", "tables")
NS = [16, 32, 64, 128, 256, 512]
NS_SVD = [16, 32]
sing_msg = [""]
E2M1 = math.e**2 - 1.0                      # predicted first-order defect constant
GHOST_C = -(math.e - 1.0) ** 2 / 6.0        # predicted ghost defect constant


def _write(name, rows):
    with open(os.path.join(RES, name), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def nullspace(log):
    rows = []
    for cl in P.CLOSURES:
        for N in NS_SVD:
            A, F, h, I, J = P.assemble(N, cl, P.f_star, **P.Q_STAR)
            w = P.left_null(N, cl)
            U_, s, Vt = np.linalg.svd(A.toarray())
            v = Vt[-1]; u = U_[:, -1]
            v = v * np.sign(v.sum()); u = u * np.sign(u @ w)
            ones = np.ones(A.shape[0])
            anorm = float(abs(A).sum(axis=1).max())
            rows.append(dict(
                closure=cl, N=N, n=A.shape[0],
                A1_rel=float(np.abs(A @ ones).max() / anorm),
                ATw_rel=float(np.abs(A.T @ w).max() / (anorm * np.abs(w).max())),
                sigma_max=float(s[0]), sigma_min=float(s[-1]), sigma_2=float(s[-2]),
                sigma_min_rel=float(s[-1] / s[0]), sigma_2_rel=float(s[-2] / s[0]),
                rank_deficiency=int(np.sum(s < s[0] * A.shape[0] * np.finfo(float).eps)),
                v_minus_const=float(np.abs(v - 1 / math.sqrt(len(v))).max()),
                u_vs_w=float(np.abs(u - w / np.linalg.norm(w)).max()),
            ))
            r = rows[-1]
            log.append(f"{cl:5s} N={N:3d} n={r['n']} |A1|/|A|={r['A1_rel']:.1e} |A^Tw|={r['ATw_rel']:.1e} "
                       f"s_min/s_max={r['sigma_min_rel']:.2e} s_2/s_max={r['sigma_2_rel']:.2e} "
                       f"defic={r['rank_deficiency']} |v-1/sqrt n|={r['v_minus_const']:.1e} |u-w/|w||={r['u_vs_w']:.1e}")
    return rows


def main():
    os.makedirs(RES, exist_ok=True); os.makedirs(TAB, exist_ok=True)
    log = ["# pure-Neumann study, u* = exp(x+y), all four edges flux, (N+1)^2 unknowns", "", "## null space (dense SVD)"]
    NSR = nullspace(log)
    _write("pure_neumann_nullspace.csv", NSR)

    log += ["", "## compatibility defect, gauge, convergence"]
    DEF, CONV, UNC = [], [], []
    for cl in P.CLOSURES:
        prev = None
        for N in NS:
            A, F, h, I, J = P.assemble(N, cl, P.f_star, **P.Q_STAR)
            w = P.left_null(N, cl)
            us = P.u_star(I * h, J * h)
            ones = np.ones(A.shape[0])
            A1 = float(np.abs(A @ ones).max() / abs(A).sum(axis=1).max())
            dint = float(h**2 * (w @ F))
            p_def = float("nan") if prev is None else math.log2(prev["defect_int"] / dint)
            DEF.append(dict(closure=cl, N=N, h=h, delta_rel=P.defect(w, F), defect_int=dint,
                            p_defect=p_def,
                            scaled=dint / h if cl == "first" else dint / h**2,
                            predicted=E2M1 if cl == "first" else GHOST_C, A1_rel=A1))
            Fc = P.project(F, w)
            Um, lam = P.solve_mean(A, Fc, w, N)
            e = Um - (us - P.wmean(us, N))
            L2, Li, Lb, Lin = P.norms(e, N)
            p = (N // 2) * (N + 1) + N // 2                     # centre node, w_p = 1
            Up = P.solve_pin(A, Fc, p)
            gauge_diff = float(np.abs((Up - P.wmean(Up, N)) - Um).max())
            ep = (Up + us[p]) - us
            L2p, Lip, _, _ = P.norms(ep, N)
            row = dict(closure=cl, N=N, h=h, L2=L2, Linf=Li, Linf_boundary=Lb, Linf_interior=Lin,
                       p_L2=float("nan") if prev is None else math.log2(prev["L2"] / L2),
                       p_Linf=float("nan") if prev is None else math.log2(prev["Linf"] / Li),
                       lambda_border=lam, gauge_diff=gauge_diff, L2_pin=L2p, Linf_pin=Lip,
                       defect_int=dint, resid_rel=float(np.abs(Fc - A @ Um).max() / np.abs(Fc).max()))
            CONV.append(row); prev = row
            # brief: what happens without the projection
            Ub, lamb = P.solve_mean(A, F, w, N)
            eb = Ub - (us - P.wmean(us, N))
            Upr = P.solve_pin(A, F, p)
            epr = (Upr - P.wmean(Upr, N)) - (us - P.wmean(us, N))
            UNC.append(dict(closure=cl, N=N, lambda_raw=lamb,
                            Linf_border_raw=float(np.abs(eb).max()),
                            Linf_pin_raw=float(np.abs(epr).max()),
                            pinrow_resid=float(abs(F[p] - (A @ Upr)[p])),
                            Linf_projected=Li))
            log.append(f"{cl:5s} N={N:3d} h^2 w^TF={dint:.4e} p={p_def:.2f} delta={DEF[-1]['delta_rel']:.3e} "
                       f"L2={L2:.3e} p={row['p_L2']:.2f} Linf={Li:.3e} p={row['p_Linf']:.2f} "
                       f"bnd={Lb:.2e} int={Lin:.2e} lambda={lam:.1e} gauge_diff={gauge_diff:.1e} "
                       f"| raw: lambda={lamb:.2e} Linf(border)={UNC[-1]['Linf_border_raw']:.2e} "
                       f"Linf(pin)={UNC[-1]['Linf_pin_raw']:.2e} pin-row residual={UNC[-1]['pinrow_resid']:.2e}")

    # direct spsolve on the singular A, and CG on the symmetric ghost DA (N=64)
    N = 64
    A, F, h, I, J = P.assemble(N, "ghost", P.f_star, **P.Q_STAR)
    w = P.left_null(N, "ghost")
    with warnings.catch_warnings(record=True) as W:
        warnings.simplefilter("always")
        try:
            Us = spla.spsolve(A.tocsc(), F)
            sing = "finite" if np.all(np.isfinite(Us)) else "non-finite"
        except Exception as ex:      # pragma: no cover
            sing = f"exception {type(ex).__name__}"
        msg = "; ".join(str(x.message) for x in W) or "no warning"
    log += ["", f"## spsolve on singular ghost A (N={N}): result {sing}; {msg}"]
    D = sp.diags(w); DA = (D @ A).tocsr()
    cg = []
    for lab, rhs in (("projected", D @ P.project(F, w)), ("raw", D @ F)):
        it = [0]
        x, info = spla.cg(DA, rhs, rtol=1e-10, atol=0.0, maxiter=5000,
                          callback=lambda _: it.__setitem__(0, it[0] + 1))
        rel = float(np.linalg.norm(rhs - DA @ x) / np.linalg.norm(rhs))
        us = P.u_star(I * h, J * h)
        e = (x - P.wmean(x, N)) - (us - P.wmean(us, N))
        cg.append(dict(rhs=lab, iters=it[0], info=info, relres=rel, Linf=float(np.abs(e).max())))
        log.append(f"CG on DA (ghost, N={N}, tol 1e-10) {lab}: iters={it[0]} info={info} relres={rel:.2e} Linf(mean-gauged)={cg[-1]['Linf']:.3e}")
    _write("pure_neumann_cg.csv", cg)

    # fits
    FIT = []
    for cl in P.CLOSURES:
        R = [r for r in CONV if r["closure"] == cl]
        Dd = [r for r in DEF if r["closure"] == cl]
        hs = np.array([r["h"] for r in R])
        for nm, vals in (("L2", [r["L2"] for r in R]), ("Linf", [r["Linf"] for r in R]),
                         ("defect", [abs(r["defect_int"]) for r in Dd])):
            pfit = float(np.polyfit(np.log(hs), np.log(vals), 1)[0])
            FIT.append(dict(closure=cl, quantity=nm, fitted_order=pfit, window="N=16..512"))
            log.append(f"fit {cl:5s} {nm:6s} order {pfit:.4f}")
        Ur = [u for u in UNC if u["closure"] == cl]
        pfit = float(np.polyfit(np.log(hs), np.log([u["Linf_pin_raw"] for u in Ur]), 1)[0])
        FIT.append(dict(closure=cl, quantity="Linf_pin_unprojected", fitted_order=pfit, window="N=16..512"))
        log.append(f"fit {cl:5s} Linf of pin without projection: order {pfit:.4f}")
    _write("pure_neumann_defect.csv", DEF)
    _write("pure_neumann_convergence.csv", CONV)
    _write("pure_neumann_uncorrected.csv", UNC)
    _write("pure_neumann_fit.csv", FIT)
    open(os.path.join(RES, "pure_neumann_log.txt"), "w").write("\n".join(log) + "\n")
    print("\n".join(log))
    sing_msg[0] = sing
    tables(DEF, CONV, FIT, NSR, UNC, cg)


def _e(v, d=3):
    m, ex = f"{float(v):.{d}e}".split("e")
    return f"${m}\\times10^{{{int(ex)}}}$"


def _e_up(v, d=1):
    """Scientific TeX with the mantissa rounded up, for use as a strict upper bound."""
    av = abs(float(v))
    if av == 0:
        return _e(0, d)
    exp = math.floor(math.log10(av))
    mant = av / (10 ** exp)
    step = 10 ** (-d)
    mant_up = math.ceil(mant / step - 1e-15) * step
    if mant_up >= 10:
        mant_up = 1.0
        exp += 1
    return f"${mant_up:.{d}f}\\times10^{{{exp}}}$"


def _p(v):
    return "---" if math.isnan(v) else f"{v:.2f}"


def tables(DEF, CONV, FIT, NSR, UNC, cg):
    D = {(r["closure"], r["N"]): r for r in DEF}
    L = [r"\begin{tabular}{@{}lrcccccccc@{}}\toprule",
         r"Closure & $N$ & $h^2w^{\mathsf T}F$ & $p$ & $\|e\|_{L^2_h}$ & $p$ & $\|e\|_{\infty}$ & $p$"
         r" & $\|e\|_{\infty,\partial\Omega}$ & $\|e\|_{\infty,\mathrm{int}}$\\\midrule"]
    for cl in P.CLOSURES:
        lab = "first-order" if cl == "first" else "ghost-point"
        for r in [r for r in CONV if r["closure"] == cl]:
            d = D[(cl, r["N"])]
            L.append(f"{lab} & {r['N']} & {_e(d['defect_int'])} & {_p(d['p_defect'])} & {_e(r['L2'])} & "
                     f"{_p(r['p_L2'])} & {_e(r['Linf'])} & {_p(r['p_Linf'])} & {_e(r['Linf_boundary'])} & "
                     f"{_e(r['Linf_interior'])}\\\\")
        L.append(r"\midrule" if cl == "first" else r"\bottomrule")
    L.append(r"\end{tabular}")
    open(os.path.join(TAB, "pure_neumann.tex"), "w").write("\n".join(L) + "\n")

    F_ = {(r["closure"], r["quantity"]): r["fitted_order"] for r in FIT}
    last = {cl: [r for r in CONV if r["closure"] == cl][-1] for cl in P.CLOSURES}
    S = {(r["closure"], r["N"]): r for r in NSR}
    U = {(r["closure"], r["N"]): r for r in UNC}
    Dl = {cl: [r for r in DEF if r["closure"] == cl] for cl in P.CLOSURES}
    mx = lambda key, rows: max(abs(r[key]) for r in rows)
    M = ["% written by run_pure_neumann.py from results/pure_neumann_*.csv"]
    def m(name, val):
        val = str(val)
        if val.startswith("$") and val.endswith("$"):
            val = "\\ensuremath{" + val[1:-1] + "}"
        M.append(f"\\newcommand{{\\{name}}}{{{val}}}")
    m("PNAoneMax", _e(max(r["A1_rel"] for r in DEF), 1))
    m("PNATwMax", _e(max(r["ATw_rel"] for r in NSR), 1))
    for cl, tag in (("first", "F"), ("ghost", "G")):
        s = S[(cl, 32)]
        m(f"PN{tag}sminrel", _e(s["sigma_min_rel"], 1))
        m(f"PN{tag}stworel", _e(s["sigma_2_rel"], 1))
        m(f"PN{tag}vconst", _e(max(S[(cl, n)]["v_minus_const"] for n in NS_SVD), 1))
        m(f"PN{tag}uw", _e(max(S[(cl, n)]["u_vs_w"] for n in NS_SVD), 1))
        m(f"PN{tag}stwoA", f"{S[(cl, 16)]['sigma_2']:.2f}")
        m(f"PN{tag}stwoB", f"{S[(cl, 32)]['sigma_2']:.2f}")
        m(f"PN{tag}sminrelMax", _e_up(max(S[(cl, n)]["sigma_min_rel"] for n in NS_SVD), 1))
        m(f"PN{tag}fitL", f"{F_[(cl, 'L2')]:.2f}")
        m(f"PN{tag}fitI", f"{F_[(cl, 'Linf')]:.2f}")
        m(f"PN{tag}fitD", f"{F_[(cl, 'defect')]:.2f}")
        m(f"PN{tag}lastL", f"{last[cl]['p_L2']:.2f}")
        m(f"PN{tag}lastI", f"{last[cl]['p_Linf']:.2f}")
        m(f"PN{tag}defLast", f"{Dl[cl][-1]['scaled']:.3f}")
        m(f"PN{tag}defFirst", f"{Dl[cl][0]['scaled']:.3f}")
        m(f"PN{tag}lam", _e(mx("lambda_border", [r for r in CONV if r["closure"] == cl]), 0))
        m(f"PN{tag}gauge", _e(mx("gauge_diff", [r for r in CONV if r["closure"] == cl]), 0))
        m(f"PN{tag}rawPin", _e(U[(cl, 64)]["Linf_pin_raw"], 2))
        m(f"PN{tag}rawPinFit", f"{F_[(cl, 'Linf_pin_unprojected')]:.2f}")
        m(f"PN{tag}rawPinRatioMin", f"{min(U[(cl, n)]['Linf_pin_raw'] / U[(cl, n)]['Linf_projected'] for n in NS):.2f}")
        m(f"PN{tag}rawPinRatioMax", f"{max(U[(cl, n)]['Linf_pin_raw'] / U[(cl, n)]['Linf_projected'] for n in NS):.2f}")
        m(f"PN{tag}rawProj", _e(U[(cl, 64)]["Linf_projected"], 2))
        m(f"PN{tag}rawBorder", _e(U[(cl, 64)]["Linf_border_raw"], 2))
        m(f"PN{tag}rawLam", _e(U[(cl, 64)]["lambda_raw"], 2))
    m("PNdefFpred", f"{E2M1:.3f}")
    m("PNdefGpred", f"{GHOST_C:.4f}")
    m("PNGdefLastFour", f"{Dl['ghost'][-1]['scaled']:.4f}")
    cp, cr = cg
    m("PNspsolve", sing_msg[0])
    m("PNcgIt", f"{cp['iters']}")
    m("PNcgRawIt", f"{cr['iters']}")
    m("PNcgRawRes", _e(cr["relres"], 1))
    open(os.path.join(TAB, "pure_neumann_stats.tex"), "w").write("\n".join(M) + "\n")
    print("wrote pure_neumann.tex, pure_neumann_stats.tex")


def run():
    main()


if __name__ == "__main__":
    main()
