"""Geometric Brownian motion exact, EM and Milstein terminal solvers."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def _check(dW: Array, h: float) -> None:
    if dW.ndim != 2 or dW.shape[1] < 1 or h <= 0:
        raise ValueError("dW must have shape (paths, steps) and h>0")


def gbm_exact_terminal(s0: float, mu: float, sigma: float, t: float, dW: Array) -> Array:
    _check(dW, t)
    return np.asarray(s0 * np.exp((mu - 0.5 * sigma**2) * t + sigma * dW.sum(axis=1)), dtype=np.float64)


def gbm_em_terminal(s0: float, mu: float, sigma: float, h: float, dW: Array) -> Array:
    _check(dW, h)
    s = np.full(dW.shape[0], s0, dtype=np.float64)
    for k in range(dW.shape[1]):
        s *= 1.0 + mu * h + sigma * dW[:, k]
    return s


def gbm_milstein_terminal(s0: float, mu: float, sigma: float, h: float, dW: Array) -> Array:
    _check(dW, h)
    s = np.full(dW.shape[0], s0, dtype=np.float64)
    for k in range(dW.shape[1]):
        w = dW[:, k]
        s *= 1.0 + mu * h + sigma * w + 0.5 * sigma**2 * (w * w - h)
    return s


def gbm_exact_mean(s0: float, mu: float, t: float) -> float:
    return float(s0 * np.exp(mu * t))


def gbm_exact_variance(s0: float, mu: float, sigma: float, t: float) -> float:
    return float(s0**2 * np.exp(2.0 * mu * t) * np.expm1(sigma**2 * t))


def gbm_analytic_bias(s0: float, mu: float, t: float, steps: int) -> float:
    # expm1 avoids cancellation at fine step sizes.
    log_numerical = steps * np.log1p(mu * t / steps)
    return float(s0 * np.exp(mu * t) * np.expm1(log_numerical - mu * t))
