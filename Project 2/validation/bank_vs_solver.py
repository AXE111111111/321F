"""Run course test_problem_bank.py exact solutions through the team solver (code/solver.py).
Solver geometry: Dirichlet on x=0,y=0,y=1; flux u_x+beta*u=r on x=1.
P1,P2,P5: exact Dirichlet data elsewhere, Neumann data r=u_x(1,y) (beta=0) on x=1.
P3: rotated (x<->y) so its Neumann edge y=1 becomes x=1.  P4 uses variable coefficient a=1+x (solver a= argument).
Max-norm error on unknowns, 4 meshes, fitted order.

Run from Project 2 root (sibling of code/ and tests/), e.g.:
  python3 validation/bank_vs_solver.py
or from this directory:
  python3 bank_vs_solver.py
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))
sys.path.insert(0, os.path.join(ROOT, "tests"))
import solver as S
import test_problem_bank as B

d = 1e-6


def ux(u):
    return lambda x, y: (u(x + d, y) - u(x - d, y)) / (2 * d)


cases = {}
for P in (B.P1, B.P2, B.P5):
    cases[P.name] = (P.exact, P.forcing, ux(P.exact), None)
u3 = lambda x, y: B.P3.exact(y, x)
f3 = lambda x, y: B.P3.forcing(y, x)
cases[B.P3.name + " [rotated x<->y]"] = (
    u3,
    f3,
    lambda x, y: np.sin(np.pi * y) * np.e,
    None,
)
cases[B.P4.name] = (B.P4.exact, B.P4.forcing, ux(B.P4.exact), B.P4.coefficient)
for name, (u, f, uxf, a) in cases.items():
    for cl in ("first", "ghost"):
        Ns = [16, 32, 64, 128]
        E = []
        for N in Ns:
            A, F, h, I, J = S.assemble(N, 0.0, cl, f, u, lambda y: uxf(1.0, y), a=a)
            U = S.solve_direct(A, F)
            E.append(np.max(np.abs(U - u(I * h, J * h))))
        p = np.polyfit(
            np.log(1 / np.array(Ns)), np.log(np.maximum(E, 1e-300)), 1
        )[0]
        print(
            f"{name:55s} {cl:5s} errs="
            + " ".join(f"{e:.2e}" for e in E)
            + f"  order={p:.2f}"
        )
