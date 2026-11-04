"""MMS checks: truncation constants, symmetrisation, scaled residual, CG on DA,
condition-number estimate on one mesh.

Writes ../results/mms_checks.csv and appends a human log to ../results/mms_checks_log.txt.
Also refreshes ../results/d2_extra.csv for make_tables.py compatibility.
"""
from __future__ import annotations
import csv, os, time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from solver import (
    assemble, solve_direct, solve_cg, symmetrise, asymmetry,
    scaled_residual, discrete_norms,
)
from mms import (
    u_star, f_star, g_star, r_star,
    TAU_I_OVER_H2, TAU_B_FIRST_OVER_H, TAU_B_GHOST_OVER_H,
)

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
os.makedirs(RES, exist_ok=True)
NS = [16, 32, 64, 128, 256]


def run():
    log_path = os.path.join(RES, "mms_checks_log.txt")
    out = open(log_path, "w")

    def P(*a):
        print(*a)
        print(*a, file=out)

    rows = []
    for beta, lab in [(2.0, "Robin"), (0.0, "Neumann")]:
        for cl in ["first", "ghost"]:
            for N in NS:
                A, F, h, I, J = assemble(N, beta, cl, f_star, g_star, lambda y: r_star(y, beta))
                Ue = u_star(I * h, J * h)
                tau = A @ Ue - F
                bmask = I == N
                U = solve_direct(A, F)
                S, G, D = symmetrise(A, F, I, h, cl)
                rows.append(dict(
                    bc=lab, closure=cl, N=N, h=h,
                    tauB=float(np.abs(tau[bmask]).max()),
                    tauI=float(np.abs(tau[~bmask]).max()),
                    etaA=asymmetry(A),
                    etaS=asymmetry(S),
                    scaled_res=scaled_residual(A, U, F),
                ))
                r = rows[-1]
                P(
                    f"{lab:7s} {cl:5s} N={N:3d} max|tau_B|={r['tauB']:.3e} "
                    f"(/h={r['tauB']/h:.4f}) max|tau_I|={r['tauI']:.3e} "
                    f"(/h^2={r['tauI']/h**2:.4f}) eta(A)={r['etaA']:.4f} "
                    f"eta(DA)={r['etaS']:.1e} scaled_res={r['scaled_res']:.1e}"
                )

    P(
        "predicted |tau_B|/h: first", TAU_B_FIRST_OVER_H,
        " ghost", TAU_B_GHOST_OVER_H,
        "; |tau_I|/h^2", TAU_I_OVER_H2,
    )

    # CG on DA, N=64
    for beta, lab in [(2.0, "Robin"), (0.0, "Neumann")]:
        for cl in ["first", "ghost"]:
            A, F, h, I, J = assemble(64, beta, cl, f_star, g_star, lambda y: r_star(y, beta))
            S, G, D = symmetrise(A, F, I, h, cl)
            Ud = solve_direct(A, F)
            x, iters, relres, info = solve_cg(S, G, tol=1e-12)
            lmin = float(spla.eigsh(S, k=1, sigma=0, which="LM", return_eigenvectors=False)[0])
            P(
                f"CG {lab} {cl} N=64: info={info} iters={iters} "
                f"|x-direct|_inf={np.abs(x-Ud).max():.2e} lambda_min(DA)={lmin:.4f}"
            )

    # Conditioning estimate on one representative mesh (Robin ghost, N=64)
    # Brief B6: backward error + conditioning for spsolve, or tolerance sweep.
    N_c = 64
    beta_c, cl_c = 2.0, "ghost"
    A, F, h, I, J = assemble(N_c, beta_c, cl_c, f_star, g_star, lambda y: r_star(y, beta_c))
    U = solve_direct(A, F)
    sres = scaled_residual(A, U, F)
    # κ₂ estimate via extreme eigenvalues of the SPD matrix DA
    S, G, D = symmetrise(A, F, I, h, cl_c)
    lmax = float(spla.eigsh(S, k=1, which="LA", return_eigenvectors=False)[0])
    lmin = float(spla.eigsh(S, k=1, sigma=0, which="LM", return_eigenvectors=False)[0])
    kappa = lmax / lmin
    # crude 1-norm condition estimate of A via onenormest of A and of A^{-1} (via spsolve)
    Anorm = spla.onenormest(A)
    AT = A.T.tocsr()
    def inv_mv(v):
        return solve_direct(A, v)
    def inv_rmv(v):
        return solve_direct(AT, v)
    Ainv = spla.LinearOperator(A.shape, matvec=inv_mv, rmatvec=inv_rmv)
    Ainv_norm = spla.onenormest(Ainv)
    kappa1 = Anorm * Ainv_norm
    P(
        f"COND Robin ghost N={N_c}: scaled_res={sres:.2e} "
        f"kappa2(DA)={kappa:.3e} (lmax={lmax:.4e}, lmin={lmin:.4e}) "
        f"kappa1(A)~{kappa1:.3e} (||A||1~{Anorm:.3e}, ||A^-1||1~{Ainv_norm:.3e})"
    )
    # also N=128 for a second point
    for N_c2 in (128, 256):
        A2, F2, h2, I2, J2 = assemble(N_c2, beta_c, cl_c, f_star, g_star, lambda y: r_star(y, beta_c))
        S2, G2, D2 = symmetrise(A2, F2, I2, h2, cl_c)
        lmax2 = float(spla.eigsh(S2, k=1, which="LA", return_eigenvectors=False)[0])
        lmin2 = float(spla.eigsh(S2, k=1, sigma=0, which="LM", return_eigenvectors=False)[0])
        P(f"COND Robin ghost N={N_c2}: kappa2(DA)={lmax2/lmin2:.3e}")

    for path in (
        os.path.join(RES, "mms_checks.csv"),
        os.path.join(RES, "d2_extra.csv"),
    ):
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)

    # keep a copy of the log under the old name too
    with open(os.path.join(RES, "d2_extra_log.txt"), "w") as fh:
        out.flush()
        fh.write(open(log_path).read())
    out.close()
    return rows


if __name__ == "__main__":
    run()
