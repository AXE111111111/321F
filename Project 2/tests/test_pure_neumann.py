"""Tests for the pure-Neumann study (code/neumann_pure.py, report Section 6.3).
Run from the package root:  python -m pytest -q tests/
"""
import os, sys
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))
import neumann_pure as P

CL = list(P.CLOSURES)


def build(N, cl):
    A, F, h, I, J = P.assemble(N, cl, P.f_star, **P.Q_STAR)
    return A, F, h, I, J, P.left_null(N, cl)


@pytest.mark.parametrize("cl", CL)
def test_constant_is_right_null_vector(cl):
    A, *_ = build(16, cl)
    assert np.abs(A @ np.ones(A.shape[0])).max() <= 1e-12 * abs(A).sum(axis=1).max()


@pytest.mark.parametrize("cl", CL)
def test_left_null_vector(cl):
    A, F, h, I, J, w = build(16, cl)
    assert np.abs(A.T @ w).max() <= 1e-12 * np.abs(A).max() * np.abs(w).max()


def test_ghost_DA_symmetric():
    A, F, h, I, J, w = build(16, "ghost")
    DA = (np.diag(w) @ A.toarray())
    assert np.abs(DA - DA.T).max() <= 1e-12 * np.abs(DA).max()


@pytest.mark.parametrize("cl", CL)
def test_rank_deficiency_exactly_one(cl):
    A, *_ = build(16, cl)
    s = np.linalg.svd(A.toarray(), compute_uv=False)
    assert s[-1] < 1e-12 * s[0]
    assert s[-2] > 1e-6 * s[0]


@pytest.mark.parametrize("cl,order", [("first", 1.0), ("ghost", 2.0)])
def test_defect_decreases_with_expected_order(cl, order):
    d = []
    for N in (16, 32, 64):
        A, F, h, I, J, w = build(N, cl)
        d.append(abs(h**2 * (w @ F)))
    assert d[0] > d[1] > d[2]
    assert abs(np.log2(d[1] / d[2]) - order) < 0.1


@pytest.mark.parametrize("cl", CL)
def test_projection_makes_rhs_compatible(cl):
    A, F, h, I, J, w = build(32, cl)
    Fc = P.project(F, w)
    assert abs(w @ Fc) <= 1e-12 * (np.abs(w) @ np.abs(Fc))
    U, lam = P.solve_mean(A, Fc, w, 32)
    assert abs(lam) < 1e-10
    assert np.abs(Fc - A @ U).max() < 1e-8 * np.abs(Fc).max()


def test_bordered_lambda_equals_ghost_defect():
    A, F, h, I, J, w = build(32, "ghost")
    U, lam = P.solve_mean(A, F, w, 32)
    assert lam == pytest.approx(h**2 * (w @ F), rel=1e-8)


@pytest.mark.parametrize("cl", CL)
def test_gauge_fixed_solution_unique(cl):
    N = 32
    A, F, h, I, J, w = build(N, cl)
    Fc = P.project(F, w)
    Um, _ = P.solve_mean(A, Fc, w, N)
    for p in ((N // 2) * (N + 1) + N // 2, 3 * (N + 1) + 5):
        Up = P.solve_pin(A, Fc, p)
        assert np.abs((Up - P.wmean(Up, N)) - Um).max() < 1e-9
    assert abs(P.wmean(Um, N)) < 1e-12


@pytest.mark.parametrize("cl,order", [("first", 1.0), ("ghost", 2.0)])
def test_observed_order(cl, order):
    E = []
    for N in (16, 32, 64):
        A, F, h, I, J, w = build(N, cl)
        U, _ = P.solve_mean(A, P.project(F, w), w, N)
        us = P.u_star(I * h, J * h)
        E.append(np.abs(U - (us - P.wmean(us, N))).max())
    p = np.log2(E[1] / E[2])
    assert abs(p - order) < 0.15
