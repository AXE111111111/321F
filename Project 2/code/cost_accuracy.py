"""Cost vs accuracy at a matched target (brief ★★ / rubric D).

Target: ||e||_∞ ≤ TOL. For each (bc, closure) find the smallest N in the list
with measured L∞ ≤ TOL, then time spsolve (median of REPEATS runs, assembly
excluded). What is NOT counted: assembly, I/O, Python overhead outside spsolve,
CG iterations (direct solve only).

Writes ../results/cost_accuracy.csv and a short log.
"""
from __future__ import annotations
import csv, os, time
import numpy as np
from solver import assemble, solve_direct, discrete_norms
from mms import u_star, f_star, g_star, r_star

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
os.makedirs(RES, exist_ok=True)

# Target chosen so both closures hit it inside N≤512 (first-order needs ~N∝1/ε).
# For ε=1e-4 first-order would need N~O(10^3–10^4); documented in the log.
TOL = 5.0e-3
NS = [16, 32, 64, 128, 256, 512]
REPEATS = 5


def timed_solve(A, F, repeats=REPEATS):
    # warm-up
    solve_direct(A, F)
    times = []
    U = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        U = solve_direct(A, F)
        times.append(time.perf_counter() - t0)
    return U, float(np.median(times)), times


def run(tol=TOL):
    log = open(os.path.join(RES, "cost_accuracy_log.txt"), "w")

    def P(*a):
        print(*a)
        print(*a, file=log)

    P(f"Target ||e||_inf <= {tol:g}")
    P("Cost measure: median wall time of spsolve over", REPEATS, "runs (assembly not counted).")
    P("N candidates:", NS)
    rows = []
    # Full error table first (so we can also plot error vs cost)
    all_pts = []
    for beta, lab in [(0.0, "Neumann"), (2.0, "Robin")]:
        for cl in ["first", "ghost"]:
            hit = None
            for N in NS:
                A, F, h, I, J = assemble(N, beta, cl, f_star, g_star, lambda y: r_star(y, beta))
                U, tmed, ts = timed_solve(A, F)
                Ue = u_star(I * h, J * h)
                L2, Linf, Lb, Li = discrete_norms(U - Ue, I, h)
                M = N * (N - 1)
                all_pts.append(dict(
                    bc=lab, closure=cl, N=N, h=h, M=M,
                    L2=L2, Linf=Linf, t_spsolve_s=tmed,
                ))
                P(
                    f"{lab:7s} {cl:5s} N={N:3d} M={M:6d} Linf={Linf:.3e} "
                    f"t_med={tmed:.4e}s  times={[f'{t:.3e}' for t in ts]}"
                )
                if hit is None and Linf <= tol:
                    hit = dict(
                        bc=lab, closure=cl, N=N, h=h, M=M,
                        Linf=Linf, t_spsolve_s=tmed, tol=tol,
                        counted="median spsolve wall time only",
                        not_counted="assembly, I/O, CG, plotting, Python outside spsolve",
                    )
            if hit is None:
                hit = dict(
                    bc=lab, closure=cl, N=None, h=None, M=None,
                    Linf=None, t_spsolve_s=None, tol=tol,
                    counted="median spsolve wall time only",
                    not_counted="assembly, I/O, CG, plotting, Python outside spsolve; "
                                f"no N≤{NS[-1]} reached tol={tol:g}",
                )
                P(f"{lab:7s} {cl:5s} DID NOT reach tol={tol:g} by N={NS[-1]}")
            rows.append(hit)
            P(f">>> smallest N for {lab}/{cl}: {hit}")

    with open(os.path.join(RES, "cost_accuracy.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(RES, "cost_accuracy_all.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_pts[0].keys()))
        w.writeheader()
        w.writerows(all_pts)
    log.close()
    return rows, all_pts


if __name__ == "__main__":
    run()
