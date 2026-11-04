"""pytest suite for code/solver.py: course test_problem_bank + edge cases.
Run from project root:  python -m pytest -q tests/
"""
import os, sys
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import solver as S
import _solver_loop_reference as REF
from mms import f_star, g_star, r_star, u_star

try:  # course bank shipped beside this file as tests/test_problem_bank.py
    import test_problem_bank as B
except ImportError:  # pragma: no cover - package incomplete
    pytest.skip(
        "tests/test_problem_bank.py missing; cannot import course bank",
        allow_module_level=True,
    )

NS = [16, 32, 64, 128]
D = 1e-6


def ux(u):
    return lambda x, y: (u(x + D, y) - u(x - D, y)) / (2 * D)


def err_inf(u, f, r, closure, N, beta=0.0, a=None):
    A, F, h, I, J = S.assemble(N, beta, closure, f, u, r, a=a)
    U = S.solve_direct(A, F)
    return np.max(np.abs(U - u(I * h, J * h)))


def order(errs):
    return np.polyfit(np.log(1.0 / np.array(NS)), np.log(errs), 1)[0]


# ---------- bank (P1, P2, P3 rotated, P5; P4 with variable coefficient) ----------
u3 = lambda x, y: B.P3.exact(y, x)
f3 = lambda x, y: B.P3.forcing(y, x)
BANK = {
    "P1": (B.P1.exact, B.P1.forcing, None),
    "P3rot": (u3, f3, None),
    "P5": (B.P5.exact, B.P5.forcing, None),
    "P4": (B.P4.exact, B.P4.forcing, B.P4.coefficient),
}
EXPECT = {"first": 1.0, "ghost": 2.0}


@pytest.mark.parametrize("name", list(BANK))
@pytest.mark.parametrize("closure", ["first", "ghost"])
def test_bank_observed_order(name, closure):
    u, f, a = BANK[name]
    du = ux(u)
    errs = [err_inf(u, f, lambda y: du(1.0, y), closure, N, a=a) for N in NS]
    p = order(errs)
    if name in ("P1", "P4") and closure == "first":  # P4 reuses P1 exact solution
        # u_xx(1,y)=0 kills the leading O(h) boundary truncation term -> order 2 observed
        assert p > 1.8
    else:
        assert abs(p - EXPECT[closure]) < 0.2, (name, closure, p, errs)


def test_bank_forcing_selfcheck():
    B.verify_bank()


# ---------- boundary sign ----------
def test_boundary_sign_quadratic_exact():
    """Ghost closure + five-point stencil are exact on x^2-y^2: error must be round-off."""
    u = B.P2.exact
    du = ux(u)
    e = err_inf(u, B.P2.forcing, lambda y: du(1.0, y), "ghost", 32)
    assert e < 1e-9


def test_boundary_sign_flip_detected():
    """Flipping the sign of the flux data r must destroy accuracy (catches sign errors)."""
    u = B.P2.exact
    du = ux(u)
    good = err_inf(u, B.P2.forcing, lambda y: du(1.0, y), "ghost", 32)
    bad = err_inf(u, B.P2.forcing, lambda y: -du(1.0, y), "ghost", 32)
    assert bad > 1e6 * max(good, 1e-15)


def test_robin_beta_mms():
    for cl, p_exp in EXPECT.items():
        errs = [err_inf(u_star, f_star, lambda y: r_star(y, 2.0), cl, N, beta=2.0) for N in NS]
        assert abs(order(errs) - p_exp) < 0.2


# ---------- index map ----------
@pytest.mark.parametrize("N", [3, 4, 7])
def test_index_map_explicit(N):
    A, F, h, I, J = S.assemble(N, 0.0, "ghost", f_star, g_star, lambda y: r_star(y, 0.0))
    k = np.array([S.index(i, j, N) for i, j in zip(I, J)])
    assert (k == np.arange(N * (N - 1))).all()
    assert A.shape == (N * (N - 1),) * 2


def test_index_transposition_detected():
    """Unknown grid is N x (N-1) and u = x^2 y is asymmetric: a transposed reshape must fail."""
    u = lambda x, y: x**2 * y - y**3 / 3.0  # harmonic? -Δ = -(2y - 2y) = 0
    f = lambda x, y: 0.0 * x
    du = ux(u)
    N = 16
    A, F, h, I, J = S.assemble(N, 0.0, "ghost", f, u, lambda y: du(1.0, y))
    U = S.solve_direct(A, F)
    exact = u(I * h, J * h)
    assert np.max(np.abs(U - exact)) < 1e-9
    Ut = U.reshape(N - 1, N).T.ravel()[: U.size]  # deliberately wrong ordering
    assert np.max(np.abs(Ut - exact)) > 1e-2


# ---------- tolerance contamination ----------
def test_tolerance_contamination():
    """Loose CG tolerance contaminates the error; tight CG agrees with spsolve far below discretisation error."""
    N, beta, cl = 64, 2.0, "ghost"
    A, F, h, I, J = S.assemble(N, beta, cl, f_star, g_star, lambda y: r_star(y, beta))
    As, Fs, _ = S.symmetrise(A, F, I, h, cl)
    exact = u_star(I * h, J * h)
    Ud = S.solve_direct(A, F)
    disc = np.max(np.abs(Ud - exact))
    Ut, *_ = S.solve_cg(As, Fs, tol=1e-12)
    Ul, *_ = S.solve_cg(As, Fs, tol=1e-2)
    assert np.max(np.abs(Ut - Ud)) < 1e-3 * disc
    assert np.max(np.abs(Ul - exact)) > 2 * disc
    assert S.scaled_residual(A, Ud, F) < 1e-13


# ---------- structure / degenerate coefficient / refactor regression ----------
@pytest.mark.parametrize("cl", ["first", "ghost"])
def test_symmetrised_matrix_symmetric(cl):
    A, F, h, I, J = S.assemble(32, 2.0, cl, f_star, g_star, lambda y: r_star(y, 2.0))
    As, _, _ = S.symmetrise(A, F, I, h, cl)
    assert S.asymmetry(As) < 1e-14


def test_degenerate_coefficient_scaling():
    """Constant a = c rescales A by c exactly (ghost, beta=0)."""
    r = lambda y: r_star(y, 0.0)
    A1, *_ = S.assemble(16, 0.0, "ghost", f_star, g_star, r)
    Ac, *_ = S.assemble(16, 0.0, "ghost", f_star, g_star, r, a=lambda x, y: 1e-6 + 0 * x)
    assert abs(Ac - 1e-6 * A1).max() < 1e-12 * abs(Ac).max()


def test_unknown_closure_rejected():
    with pytest.raises(ValueError):
        S.assemble(4, 0.0, "bogus", f_star, g_star, lambda y: 0.0)


@pytest.mark.parametrize("N", [4, 9, 32])
@pytest.mark.parametrize("beta", [0.0, 2.0])
@pytest.mark.parametrize("cl", ["first", "ghost"])
def test_vectorised_matches_loop_reference(N, beta, cl):
    r = lambda y: r_star(y, beta)
    A0, F0, *_ = REF.assemble(N, beta, cl, f_star, g_star, r)
    A1, F1, *_ = S.assemble(N, beta, cl, f_star, g_star, r)
    assert abs(A1 - A0).max() <= 1e-12 * abs(A0).max()
    assert np.max(np.abs(F1 - F0)) <= 1e-12 * np.max(np.abs(F0))
