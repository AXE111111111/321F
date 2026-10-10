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


def coarsen_increments(dB_fine: Array, steps: int) -> Array:
    """Map a (paths, master_N) Brownian array to a grid with N=steps.

    Require N to divide master_N exactly. For R = master_N / N, coarse
    increment j is the sum of consecutive master increments j*R:(j+1)*R.
    Thus master increments with variance T/master_N give variance h = T/N.

    Always coarsen directly from the master, without drawing random numbers.
    Delegating to the existing block sum (including R=1) preserves its
    floating-point reduction order; successive coarsening can change round-off.
    """
    if dB_fine.ndim != 2 or dB_fine.shape[1] < 1:
        raise ValueError("master increments must be 2D with a nonempty time axis")
    master_steps = dB_fine.shape[1]
    if (not isinstance(steps, (int, np.integer)) or steps < 1 or
            master_steps % steps):
        raise ValueError("coarse steps must be a positive integer dividing the master grid")
    return aggregate_increments(dB_fine, master_steps // steps)


def correlate_increments(dB1: Array, dB2: Array, rho: float) -> tuple[Array, Array]:
    """Construct correlated increments from two independent master streams."""
    if dB1.shape != dB2.shape or dB1.ndim != 2 or not (-1.0 < rho < 1.0):
        raise ValueError("matching 2D streams and |rho|<1 required")
    return dB1, np.asarray(rho * dB1 + np.sqrt(1.0 - rho * rho) * dB2, dtype=np.float64)
