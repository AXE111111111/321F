"""Fixed, reproducible experiment settings for Direction 4."""
from __future__ import annotations

from dataclasses import dataclass


SEED = 32104

# Ordered step counts N = 2**k, k = 1,...,9; every step size is h = T / N.
# GBM uses the full hierarchy. The main non-affine experiment retains its
# original N >= 32 subset; adding levels would change the saved experiment.
CONVERGENCE_STEPS = (2, 4, 8, 16, 32, 64, 128, 256, 512)
NONAFFINE_REFERENCE_STEPS = (4_096, 8_192)


@dataclass(frozen=True)
class GBMConfig:
    s0: float = 100.0
    mu: float = 0.05
    sigma: float = 0.20
    t: float = 1.0
    paths: int = 2_000_000
    batch_size: int = 10_000
    fine_steps: int = CONVERGENCE_STEPS[-1]
    grids: tuple[int, ...] = CONVERGENCE_STEPS
    distribution_steps: int = 256
    strong_fit_min_steps: int = 64


@dataclass(frozen=True)
class NonAffineConfig:
    s0: float = 100.0
    y0: float = 0.0
    mu: float = 0.05
    kappa: float = 2.0
    theta: float = -0.2
    xi: float = 0.6
    rho: float = -0.7
    sigma_min: float = 0.10
    sigma_max: float = 0.50
    t: float = 1.0
    paths: int = 50_000
    batch_size: int = 1_000
    fine_steps: int = NONAFFINE_REFERENCE_STEPS[-1]
    grids: tuple[int, ...] = tuple(n for n in CONVERGENCE_STEPS if n >= 32)
    references: tuple[int, ...] = NONAFFINE_REFERENCE_STEPS
    strike: float = 100.0
