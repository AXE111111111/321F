"""Tests for the CG tolerance sweep (code/run_tolerance_sweep.py, report Section 4.2).
Run from the package root:  python -m pytest -q tests/
"""
import os, sys
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))
import solver as S
from mms import f_star, g_star, r_star, u_star

TOLS = [1e-4, 1e-6, 1e-8, 1e-10, 1e-12]


@pytest.mark.parametrize("closure", ["ghost", "first"])
@pytest.mark.parametrize("beta", [0.0, 2.0])
def test_cg_on_DA_meets_tolerance_and_approaches_direct(closure, beta):
    N = 16
    A, F, h, I, J = S.assemble(N, beta, closure, f_star, g_star, lambda y: r_star(y, beta))
    DA, DF, _ = S.symmetrise(A, F, I, h, closure)
    Ud = S.solve_direct(A, F)
    diffs = []
    for tol in TOLS:
        x, iters, relres, info = S.solve_cg(DA, DF, tol=tol)
        assert info == 0
        assert relres <= tol            # stopping rule is the residual of DA U = DF
        diffs.append(np.abs(x - Ud).max())
    assert all(b < a for a, b in zip(diffs, diffs[1:]))
    assert diffs[-1] < 1e-9


def test_tight_tolerance_leaves_discretisation_error_unchanged():
    N, beta = 32, 2.0
    A, F, h, I, J = S.assemble(N, beta, "ghost", f_star, g_star, lambda y: r_star(y, beta))
    DA, DF, _ = S.symmetrise(A, F, I, h, "ghost")
    Ue = u_star(I * h, J * h)
    e_dir = np.abs(S.solve_direct(A, F) - Ue).max()
    x, *_ = S.solve_cg(DA, DF, tol=1e-12)
    assert abs(np.abs(x - Ue).max() - e_dir) < 1e-6 * e_dir
