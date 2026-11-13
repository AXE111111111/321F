"""Incompatible manufactured solution at the mixed corner (1,0).

Local complex coordinate z = (1-x) + i y, with
    u*(x,y) = Im(z Log z) = y log r + (1-x) θ,
where r = sqrt((1-x)^2 + y^2) and θ = atan2(y, 1-x). On the unit square,
θ ∈ [0, π/2]. The function is harmonic (f = 0) for r > 0; the only
singularity is at (1,0). At that corner r log r → 0, so u*(1,0) = 0;
log 0 is never evaluated.

Analytic identities (r > 0):
    u_x = -θ,   u_y = 1 + log r,
    on y=0, x<1:  θ=0, u*=0,
    on x=1, y>0:  r=y, θ=π/2, u*= y log y,
    at (1,1):     r=1, θ=π/2, u*=0.
The flux data on x=1 are r_bdry(y) = u_x + β u* = -π/2 + β y log y
(with y log y → 0 as y→0). For Neumann (β=0) this is the constant -π/2.

All callables are array-capable (NumPy broadcasting).
"""
from __future__ import annotations
import numpy as np

_PI_HALF = 0.5 * np.pi


def _as_xy(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x, y = np.broadcast_arrays(x, y)
    return x, y, (np.ndim(np.asarray(x)) == 0 and np.ndim(np.asarray(y)) == 0)


def _maybe_scalar(value, scalar: bool):
    value = np.asarray(value, dtype=float)
    if scalar:
        return float(value)
    return value


def polar(x, y):
    """Return (r, theta, xi) with xi = 1-x, θ = atan2(y, xi).

    At the single point (1,0), NumPy's atan2(0, 0) is 0 and r is 0.
    Derivatives must not be interpreted at that isolated point.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x, y = np.broadcast_arrays(x, y)
    xi = 1.0 - x
    r = np.hypot(xi, y)
    theta = np.arctan2(y, xi)
    return r, theta, xi


def u_star(x, y):
    """u* = y log r + (1-x) θ, with u*(1,0) = 0."""
    scalar = np.ndim(x) == 0 and np.ndim(y) == 0
    r, theta, xi = polar(x, y)
    yb = np.broadcast_to(np.asarray(y, dtype=float), r.shape)
    out = np.zeros(r.shape, dtype=float)
    pos = r > 0
    out[pos] = yb[pos] * np.log(r[pos]) + xi[pos] * theta[pos]
    return _maybe_scalar(out, scalar)


def f_star(x, y):
    """Forcing: -Δu* = 0 away from (1,0)."""
    scalar = np.ndim(x) == 0 and np.ndim(y) == 0
    x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    return _maybe_scalar(np.zeros(x.shape, dtype=float), scalar)


def g_star(x, y):
    return u_star(x, y)


def u_x(x, y):
    """Exact ∂u*/∂x = -θ for r > 0."""
    scalar = np.ndim(x) == 0 and np.ndim(y) == 0
    _, theta, _ = polar(x, y)
    return _maybe_scalar(-theta, scalar)


def u_y(x, y):
    """Exact ∂u*/∂y = 1 + log r for r > 0. Not evaluated at r=0."""
    scalar = np.ndim(x) == 0 and np.ndim(y) == 0
    r, _, _ = polar(x, y)
    out = np.full(np.shape(r), np.nan, dtype=float)
    pos = r > 0
    out[pos] = 1.0 + np.log(r[pos])
    return _maybe_scalar(out, scalar)


def r_star(y, beta: float):
    """Flux data on x=1: r_bdry(y) = -π/2 + β y log y, with 0·log 0 := 0."""
    scalar = np.ndim(y) == 0
    y = np.asarray(y, dtype=float)
    term = np.zeros(y.shape, dtype=float)
    pos = y > 0
    term[pos] = y[pos] * np.log(y[pos])
    return _maybe_scalar(-_PI_HALF + float(beta) * term, scalar)
