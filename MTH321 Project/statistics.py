"""Batch-safe sample statistics and reproducible log-log fits."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike


@dataclass
class RunningStats:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def add(self, values: ArrayLike) -> None:
        x = np.asarray(values, dtype=np.float64).ravel()
        if not x.size:
            return
        if not np.all(np.isfinite(x)):
            raise ValueError("nonfinite samples cannot enter statistics")
        n = int(x.size)
        batch_mean = float(np.mean(x))
        batch_m2 = float(np.sum((x - batch_mean)**2, dtype=np.float64))
        if self.count == 0:
            self.count, self.mean, self.m2 = n, batch_mean, batch_m2
            return
        total = self.count + n
        delta = batch_mean - self.mean
        self.m2 += batch_m2 + delta * delta * self.count * n / total
        self.mean += delta * n / total
        self.count = total

    @property
    def sd(self) -> float:
        return float(np.sqrt(self.m2 / (self.count - 1))) if self.count > 1 else float("nan")

    @property
    def se(self) -> float:
        return self.sd / np.sqrt(self.count)

    def summary(self) -> dict[str, float | int]:
        halfwidth = 1.96 * self.se
        return {"M": self.count, "estimate": self.mean, "sample_sd": self.sd,
                "se": self.se, "ci_lower": self.mean - halfwidth, "ci_upper": self.mean + halfwidth}


def mean_se_ci(samples: ArrayLike) -> tuple[float, float, float, float]:
    stats = RunningStats()
    stats.add(samples)
    if stats.count < 2:
        raise ValueError("at least two samples required")
    v = stats.summary()
    return float(v["estimate"]), float(v["se"]), float(v["ci_lower"]), float(v["ci_upper"])


def weak_fit_eligible(row: dict[str, object]) -> bool:
    mean = float(row["estimate"])
    lo, hi = float(row["ci_lower"]), float(row["ci_upper"])
    halfwidth = (hi - lo) / 2.0
    return bool((lo > 0 or hi < 0) and halfwidth < 0.5 * abs(mean))


def fit_loglog_slope(step_sizes: ArrayLike, errors: ArrayLike,
                     valid_mask: ArrayLike) -> tuple[float, float]:
    h = np.asarray(step_sizes, dtype=np.float64)
    e = np.asarray(errors, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    if h.shape != e.shape or h.shape != valid.shape or np.count_nonzero(valid) < 2:
        raise ValueError("at least two valid matched points required")
    if not np.all(np.isfinite(h[valid])) or not np.all(np.isfinite(e[valid])) or np.any(h[valid] <= 0) or np.any(e[valid] <= 0):
        raise ValueError("fit points must be finite and positive")
    slope, intercept = np.polyfit(np.log(h[valid]), np.log(e[valid]), 1)
    return float(slope), float(intercept)


def jackknife_slope_se(step_sizes: ArrayLike, batch_sums: ArrayLike,
                       batch_counts: ArrayLike, full_slope: float) -> float:
    """Delete one independent batch at a time, keeping all grids paired.

    Rows of ``batch_sums`` are fitted grids and columns are common path batches.
    For unequal batch sizes, use weighted jackknife pseudo-values.
    """
    h = np.asarray(step_sizes, dtype=np.float64)
    sums = np.asarray(batch_sums, dtype=np.float64)
    counts = np.asarray(batch_counts, dtype=np.float64)
    if (h.ndim != 1 or sums.ndim != 2 or sums.shape[0] != h.size or
        sums.shape[1] != counts.size or counts.size < 2 or
        np.any(counts <= 0) or np.any(counts >= np.sum(counts))):
        raise ValueError("paired grid-by-batch sums and at least two batches required")
    x = np.log(h)
    x_centered = x - np.mean(x)
    denominator = float(np.dot(x_centered, x_centered))
    if denominator <= 0:
        raise ValueError("distinct fitted step sizes required")
    total_count = float(np.sum(counts))
    leave_means = (np.sum(sums, axis=1, keepdims=True) - sums) / (total_count - counts)
    if np.any(~np.isfinite(leave_means)) or np.any(leave_means == 0):
        raise ValueError("nonfinite or zero leave-one-batch fit point")
    slopes = np.sum(x_centered[:, None] * np.log(np.abs(leave_means)), axis=0) / denominator
    if np.all(counts == counts[0]):
        se2 = (counts.size - 1) / counts.size * float(np.sum((slopes - np.mean(slopes))**2))
    else:
        weights = counts / total_count
        pseudo = (total_count * full_slope - (total_count - counts) * slopes) / counts
        pseudo_mean = float(np.dot(weights, pseudo))
        se2 = float(np.sum(weights**2 * (pseudo - pseudo_mean)**2) / (1.0 - np.sum(weights**2)))
    return float(np.sqrt(max(0.0, se2)))
