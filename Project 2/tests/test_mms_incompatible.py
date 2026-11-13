"""Tests for the incompatible-corner manufactured solution.

Run from project root:  python -m pytest -q tests/test_mms_incompatible.py
"""
import os, sys
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))
import mms_incompatible as M
import solver as S

# Finite-difference step for derivative checks away from the corner.
D = 1e-7
# Geometric neighbourhood used only in these tests (not the report split).
R_AWAY = 0.2


def _grid(n=21):
    xs = np.linspace(0.0, 1.0, n)
    ys = np.linspace(0.0, 1.0, n)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    r, _, _ = M.polar(X, Y)
    return X, Y, r


def test_corner_value_is_zero_without_log0():
    assert M.u_star(1.0, 0.0) == 0.0
    # A tight approach along several rays: r log r → 0, no NaN/Inf.
    for theta in (0.0, np.pi / 8, np.pi / 4, np.pi / 2):
        for r in (1e-16, 1e-12, 1e-8, 1e-4):
            x = 1.0 - r * np.cos(theta)
            y = r * np.sin(theta)
            u = M.u_star(x, y)
            assert np.isfinite(u), (theta, r, u)
            assert abs(u) <= max(2.0 * r * abs(np.log(r)) + r * theta, 1e-12)


def test_bottom_edge_is_zero():
    xs = np.linspace(0.0, 1.0, 64, endpoint=False)  # x<1
    u = M.u_star(xs, np.zeros_like(xs))
    assert np.allclose(u, 0.0, atol=1e-15)
    assert M.u_star(0.5, 0.0) == 0.0


def test_right_edge_is_y_log_y():
    ys = np.linspace(1e-12, 1.0, 64)
    u = M.u_star(np.ones_like(ys), ys)
    expected = ys * np.log(ys)
    assert np.allclose(u, expected, rtol=1e-14, atol=1e-15)
    # y=1: log 1 = 0
    assert abs(M.u_star(1.0, 1.0)) <= 1e-15


def test_other_corners_finite_and_expected():
    # (1,1): r=1, θ=π/2, u*=0
    assert abs(M.u_star(1.0, 1.0)) <= 1e-15
    # (0,0): r=1, θ=0, u*=0
    assert abs(M.u_star(0.0, 0.0)) <= 1e-15
    # (0,1): r=√2, θ=π/4, u* = log√2 + π/4
    expected = 0.5 * np.log(2.0) + 0.25 * np.pi
    assert abs(M.u_star(0.0, 1.0) - expected) <= 1e-14


def test_forcing_is_identically_zero():
    assert M.f_star(0.3, 0.4) == 0.0
    X, Y, _ = _grid(31)
    f = M.f_star(X, Y)
    assert f.shape == X.shape
    assert np.all(f == 0.0)


def test_u_x_equals_minus_theta():
    rng = np.random.default_rng(0)
    x = rng.uniform(0.0, 1.0, size=200)
    y = rng.uniform(0.0, 1.0, size=200)
    # Keep a small distance from the singular corner.
    r, theta, _ = M.polar(x, y)
    mask = r >= 1e-8
    ux = M.u_x(x[mask], y[mask])
    assert np.allclose(ux, -theta[mask], rtol=1e-14, atol=1e-15)
    # On the flux edge x=1, y>0: u_x = -π/2
    ys = np.linspace(1e-6, 1.0, 40)
    assert np.allclose(M.u_x(np.ones_like(ys), ys), -0.5 * np.pi, rtol=0, atol=1e-15)


def test_u_x_matches_finite_difference_away_from_corner():
    X, Y, r = _grid(25)
    mask = r >= R_AWAY
    ux = M.u_x(X[mask], Y[mask])
    fd = (M.u_star(X[mask] + D, Y[mask]) - M.u_star(X[mask] - D, Y[mask])) / (2 * D)
    assert np.allclose(ux, fd, rtol=1e-6, atol=1e-8)


def test_laplacian_vanishes_away_from_corner():
    """Five-point Laplacian of u* should be tiny once r is bounded away from 0."""
    h = 1e-4
    xs = np.linspace(0.05, 0.95, 18)
    ys = np.linspace(0.05, 0.95, 18)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    r, _, _ = M.polar(X, Y)
    mask = r >= R_AWAY
    lap = (
        M.u_star(X + h, Y) + M.u_star(X - h, Y)
        + M.u_star(X, Y + h) + M.u_star(X, Y - h)
        - 4.0 * M.u_star(X, Y)
    ) / h**2
    assert np.max(np.abs(lap[mask])) < 1e-4


def test_flux_data_neumann_is_constant():
    ys = np.linspace(0.0, 1.0, 50)
    r0 = M.r_star(ys, 0.0)
    assert np.allclose(r0, -0.5 * np.pi, rtol=0, atol=1e-15)
    assert M.r_star(0.0, 0.0) == -0.5 * np.pi


def test_flux_data_robin_and_safe_log0():
    assert M.r_star(0.0, 2.0) == -0.5 * np.pi
    ys = np.array([0.0, 1e-16, 1e-8, 0.2, 1.0 / np.e, 1.0])
    got = M.r_star(ys, 2.0)
    expected = np.array([
        -0.5 * np.pi,
        -0.5 * np.pi + 2.0 * 1e-16 * np.log(1e-16),
        -0.5 * np.pi + 2.0 * 1e-8 * np.log(1e-8),
        -0.5 * np.pi + 2.0 * 0.2 * np.log(0.2),
        -0.5 * np.pi + 2.0 * (1.0 / np.e) * np.log(1.0 / np.e),
        -0.5 * np.pi,
    ])
    assert np.allclose(got, expected, rtol=1e-14, atol=1e-15)
    assert np.all(np.isfinite(got))


def test_array_and_scalar_shapes():
    xs = np.linspace(0.0, 1.0, 7)
    ys = np.linspace(0.0, 1.0, 5)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    U = M.u_star(X, Y)
    assert U.shape == X.shape
    assert U[ -1, 0] == 0.0  # (1,0)
    # scalar in, scalar out
    assert isinstance(M.u_star(0.3, 0.4), float)
    assert isinstance(M.r_star(0.3, 2.0), float)
    assert isinstance(M.f_star(0.3, 0.4), float)


def test_g_star_is_u_star():
    x, y = 0.25, 0.8
    assert M.g_star(x, y) == M.u_star(x, y)
    X, Y, _ = _grid(9)
    assert np.allclose(M.g_star(X, Y), M.u_star(X, Y))


def test_nearest_unknown_to_mixed_corner_is_at_r_equals_h():
    """Unknowns are i=1..N, j=1..N-1; the closest node to (1,0) is (1,h)."""
    for N in (8, 16, 32):
        A, F, h, I, J = S.assemble(
            N, 0.0, "ghost", M.f_star, M.g_star, lambda y: M.r_star(y, 0.0)
        )
        x, y = I * h, J * h
        r = np.hypot(1.0 - x, y)
        k = int(np.argmin(r))
        assert abs(x[k] - 1.0) <= 1e-15
        assert abs(y[k] - h) <= 1e-15
        assert abs(r[k] - h) <= 1e-15
        # The corner itself is not an unknown.
        assert not np.any((np.isclose(x, 1.0) & np.isclose(y, 0.0)))


@pytest.mark.parametrize("closure", ["first", "ghost"])
@pytest.mark.parametrize("beta", [0.0, 2.0])
def test_solver_returns_finite_error(closure, beta):
    N = 16
    A, F, h, I, J = S.assemble(
        N, beta, closure, M.f_star, M.g_star, lambda y: M.r_star(y, beta)
    )
    U = S.solve_direct(A, F)
    Ue = M.u_star(I * h, J * h)
    e = U - Ue
    assert np.all(np.isfinite(U))
    assert np.all(np.isfinite(Ue))
    assert np.all(np.isfinite(e))
    assert np.max(np.abs(e)) < 1.0  # crude sanity bound, not an order claim
