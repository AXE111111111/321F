"""Externally generated Brownian paths; solvers never own a random generator."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def normal_increments(rng: np.random.Generator, paths: int, steps: int, t: float) -> Array:
    """Return independent N(0, t/steps) increments, shape (paths, steps)."""
    if paths < 1 or steps < 1 or t <= 0:
        raise ValueError("paths, steps and time must be positive")
    return np.asarray(rng.normal(0.0, np.sqrt(t / steps), size=(paths, steps)), dtype=np.float64)


def aggregate_increments(dB_fine: Array, ratio: int) -> Array:
    """Sum consecutive fine increments without regenerating randomness."""
    if dB_fine.ndim != 2 or ratio < 1 or dB_fine.shape[1] % ratio:
        raise ValueError("fine increments must be 2D and steps divisible by ratio")
    paths, steps = dB_fine.shape
    return np.asarray(dB_fine.reshape(paths, steps // ratio, ratio).sum(axis=2), dtype=np.float64)


def correlate_increments(dB1: Array, dB2: Array, rho: float) -> tuple[Array, Array]:
    """Construct correlated increments from two independent master streams."""
    if dB1.shape != dB2.shape or dB1.ndim != 2 or not (-1.0 < rho < 1.0):
        raise ValueError("matching 2D streams and |rho|<1 required")
    return dB1, np.asarray(rho * dB1 + np.sqrt(1.0 - rho * rho) * dB2, dtype=np.float64)
