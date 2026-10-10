"""Two-noise Euler–Maruyama simulation for non-affine stochastic volatility."""
from __future__ import annotations

import numpy as np
from scipy.special import expit
from numpy.typing import NDArray

from config import NonAffineConfig

Array = NDArray[np.float64]


def volatility(y: Array, params: NonAffineConfig) -> Array:
    """Stable, mathematically unchanged bounded logistic volatility."""
    return params.sigma_min + (params.sigma_max - params.sigma_min) * expit(y)


def nonaffine_em_terminal(params: NonAffineConfig, h: float, dW1: Array,
                          dW2: Array) -> tuple[Array, Array, Array]:
    if dW1.ndim != 2 or dW1.shape != dW2.shape or dW1.shape[1] < 1 or h <= 0:
        raise ValueError("matching 2D increment arrays and h>0 required")
    x = np.full(dW1.shape[0], np.log(params.s0), dtype=np.float64)
    y = np.full(dW1.shape[0], params.y0, dtype=np.float64)
    for k in range(dW1.shape[1]):
        g = volatility(y, params)
        x += (params.mu - 0.5 * g * g) * h + g * dW1[:, k]
        y += params.kappa * (params.theta - y) * h + params.xi * np.sqrt(1.0 + y * y) * dW2[:, k]
    return x, y, np.exp(x)
