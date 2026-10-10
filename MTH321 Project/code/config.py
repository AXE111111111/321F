"""Fixed, reproducible experiment settings for Direction 4."""
from __future__ import annotations

from dataclasses import dataclass


SEED = 32104


@dataclass(frozen=True)
class GBMConfig:
    s0: float = 100.0
    mu: float = 0.05
    sigma: float = 0.20
    t: float = 1.0
    paths: int = 2_000_000
    batch_size: int = 10_000
    fine_steps: int = 512
    grids: tuple[int, ...] = (2, 4, 8, 16, 32, 64, 128, 256, 512)
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
    fine_steps: int = 8_192
    grids: tuple[int, ...] = (32, 64, 128, 256, 512)
    references: tuple[int, ...] = (4_096, 8_192)
    strike: float = 100.0
