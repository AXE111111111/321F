"""Refinement study: both closures × {Neumann, Robin}, N=16..256.

Writes ../results/convergence.csv and prints observed orders.
Also writes ticket_d2_results.csv in this folder for backward compatibility
with make_tables.py / d2_extra.py.
"""
from __future__ import annotations
import csv, os, sys
import numpy as np
from solver import assemble, solve_direct, discrete_norms, asymmetry
from mms import u_star, f_star, g_star, r_star

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
os.makedirs(RES, exist_ok=True)

NS = [16, 32, 64, 128, 256]


def run():
    rows = []
    for beta, lab in [(0.0, "Neumann"), (2.0, "Robin")]:
        for cl in ["first", "ghost"]:
            prev = None
            for N in NS:
                A, F, h, I, J = assemble(N, beta, cl, f_star, g_star, lambda y: r_star(y, beta))
                U = solve_direct(A, F)
                Ue = u_star(I * h, J * h)
                e = U - Ue
                L2, Linf, Lb, Li = discrete_norms(e, I, h)
                kmax = int(np.abs(e).argmax())
                row = dict(
                    bc=lab, closure=cl, N=N, h=h,
                    L2=L2, Linf=Linf, Linf_boundary=Lb, Linf_interior=Li,
                    argmax_x=float(I[kmax] * h), argmax_y=float(J[kmax] * h),
                    asym_eta=asymmetry(A),
                )
                if prev is not None:
                    row["p_L2"] = float(np.log2(prev["L2"] / L2))
                    row["p_Linf"] = float(np.log2(prev["Linf"] / Linf))
                else:
                    row["p_L2"] = float("nan")
                    row["p_Linf"] = float("nan")
                rows.append(row)
                prev = row
                print(
                    f"{lab:7s} {cl:5s} N={N:3d} L2={L2:.3e} p={row['p_L2']:.2f} "
                    f"Linf={Linf:.3e} p={row['p_Linf']:.2f} "
                    f"bnd={Lb:.2e} int={Li:.2e} eta={row['asym_eta']:.3f}"
                )
    fields = list(rows[0].keys())
    for path in (
        os.path.join(RES, "convergence.csv"),
        os.path.join(HERE, "ticket_d2_results.csv"),
    ):
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    return rows


if __name__ == "__main__":
    run()
