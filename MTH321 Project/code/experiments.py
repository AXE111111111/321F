"""Coupled, batched Direction 4 experiments and CSV artifacts."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import norm

from config import GBMConfig, NonAffineConfig, SEED
from gbm import (gbm_analytic_bias, gbm_em_terminal, gbm_exact_mean,
                 gbm_exact_terminal, gbm_exact_variance, gbm_milstein_terminal)
from nonaffine import nonaffine_em_terminal
from random_paths import aggregate_increments, correlate_increments, normal_increments
from statistics import RunningStats, fit_loglog_slope, jackknife_slope_se, weak_fit_eligible


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _fit_rows(rows: list[dict[str, object]], group_keys: tuple[str, ...],
              kind: str, strong_min_n: int = 0,
              batch_sums: dict[tuple[object, ...], list[float]] | None = None,
              batch_counts: list[int] | None = None) -> None:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in group_keys)].append(row)
    for group in groups.values():
        for row in group:
            if kind == "weak":
                eligible = weak_fit_eligible(row)
            else:
                estimate = float(row["estimate"])
                halfwidth = (float(row["ci_upper"]) - float(row["ci_lower"])) / 2
                eligible = int(row["N"]) >= strong_min_n and estimate > 0 and halfwidth < 0.5 * estimate
            row["resolution_eligible"] = int(eligible)
        eligible_rows = [row for row in group if int(row["resolution_eligible"])]
        # Two or three individually resolved points do not establish a stable slope.
        selected = eligible_rows if len(eligible_rows) >= 4 else []
        for row in group:
            row["fit_included"] = int(row in selected)
        slope = ""
        slope_se = ""
        residual_se = ""
        if len(selected) >= 4:
            h = np.array([float(row["h"]) for row in selected], dtype=np.float64)
            errors = np.array([abs(float(row["estimate"])) for row in selected], dtype=np.float64)
            slope, intercept = fit_loglog_slope(h, errors, np.ones(len(selected), dtype=bool))
            x = np.log(h)
            residuals = np.log(errors) - (intercept + slope * x)
            residual_se = float(np.sqrt(np.dot(residuals, residuals) /
                                        ((len(selected) - 2) * np.sum((x - np.mean(x))**2))))
            if batch_sums is not None and batch_counts is not None:
                keys = [tuple(row[key] for key in group_keys) + (int(row["N"]),)
                        for row in selected]
                paired_sums = np.array([batch_sums[key] for key in keys], dtype=np.float64)
                slope_se = jackknife_slope_se(h, paired_sums, batch_counts, slope)
        for row in group:
            row["fitted_slope"] = slope
            row["fitted_slope_se"] = slope_se
            row["fitted_slope_ci_lower"] = slope - 1.96 * slope_se if slope_se != "" else ""
            row["fitted_slope_ci_upper"] = slope + 1.96 * slope_se if slope_se != "" else ""
            row["fitted_slope_interval_method"] = (
                "normal approximation: slope +/- 1.96 jackknife SE; coverage not validated"
                if slope_se != "" else "")
            row["fitted_slope_se_method"] = "paired delete-one-batch jackknife" if slope_se != "" else ""
            row["fitted_slope_residual_se"] = residual_se
            row["jackknife_batch_count"] = len(batch_counts) if slope_se != "" else ""
            row["fit_point_count"] = len(selected)
            row["eligible_point_count"] = len(eligible_rows)


def run_gbm(cfg: GBMConfig, output: Path, seed: int = SEED) -> dict[str, Path]:
    """One master fine Brownian stream per batch drives all GBM grids."""
    if any(cfg.fine_steps % n for n in cfg.grids) or cfg.distribution_steps not in cfg.grids:
        raise ValueError("GBM grids must nest in fine grid")
    rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(2)[0])
    strong = {(method, n): RunningStats() for method in ("EM", "Milstein") for n in cfg.grids}
    weak = {(method, n): RunningStats() for method in ("EM", "Milstein") for n in cfg.grids}
    strong_batch_sums = {key: [] for key in strong}
    weak_batch_sums = {key: [] for key in weak}
    batch_counts: list[int] = []
    samples: dict[str, list[np.ndarray]] = {"EM": [], "Milstein": []}
    for start in range(0, cfg.paths, cfg.batch_size):
        batch = min(cfg.batch_size, cfg.paths - start)
        batch_counts.append(batch)
        fine = normal_increments(rng, batch, cfg.fine_steps, cfg.t)
        exact = gbm_exact_terminal(cfg.s0, cfg.mu, cfg.sigma, cfg.t, fine)
        for n in cfg.grids:
            h = cfg.t / n
            dw = aggregate_increments(fine, cfg.fine_steps // n)
            values = {"EM": gbm_em_terminal(cfg.s0, cfg.mu, cfg.sigma, h, dw),
                      "Milstein": gbm_milstein_terminal(cfg.s0, cfg.mu, cfg.sigma, h, dw)}
            for method, terminal in values.items():
                signed = terminal - exact
                absolute = np.abs(signed)
                strong[(method, n)].add(absolute)
                weak[(method, n)].add(signed)
                strong_batch_sums[(method, n)].append(float(np.sum(absolute, dtype=np.float64)))
                weak_batch_sums[(method, n)].append(float(np.sum(signed, dtype=np.float64)))
                if n == cfg.distribution_steps:
                    samples[method].append(terminal.copy())

    common = {"seed": seed, "model": "GBM", "s0": cfg.s0, "mu": cfg.mu,
              "sigma": cfg.sigma, "T": cfg.t, "fine_N": cfg.fine_steps}
    strong_rows: list[dict[str, object]] = []
    weak_rows: list[dict[str, object]] = []
    for n in cfg.grids:
        for method in ("EM", "Milstein"):
            base = {**common, "method": method, "N": n, "h": cfg.t / n}
            strong_rows.append({**base, **strong[(method, n)].summary()})
            bias = gbm_analytic_bias(cfg.s0, cfg.mu, cfg.t, n)
            w = {**base, **weak[(method, n)].summary(), "analytic_bias": bias}
            w["analytic_in_ci"] = int(float(w["ci_lower"]) <= bias <= float(w["ci_upper"]))
            w["analytic_z_score"] = (float(w["estimate"]) - bias) / float(w["se"])
            weak_rows.append(w)
    _fit_rows(strong_rows, ("method",), "strong", cfg.strong_fit_min_steps,
              strong_batch_sums, batch_counts)
    _fit_rows(weak_rows, ("method",), "weak", batch_sums=weak_batch_sums,
              batch_counts=batch_counts)

    terminal = {method: np.concatenate(chunks) for method, chunks in samples.items()}
    exact_mean = gbm_exact_mean(cfg.s0, cfg.mu, cfg.t)
    exact_variance = gbm_exact_variance(cfg.s0, cfg.mu, cfg.sigma, cfg.t)
    moment_rows = []
    for method, values in terminal.items():
        m = values.size
        sample_mean = float(np.mean(values))
        sample_variance = float(np.var(values, ddof=1))
        fourth_central = float(np.mean((values - sample_mean)**4))
        se_mean = float(np.sqrt(sample_variance / m))
        # Plug-in large-sample approximation for Var(unbiased sample variance).
        se_variance = float(np.sqrt(max(0.0, (fourth_central -
            (m - 3) / (m - 1) * sample_variance**2) / m)))
        moment_rows.append({**common, "method": method, "N": cfg.distribution_steps,
                            "h": cfg.t / cfg.distribution_steps, "M": m,
                            "sample_mean": sample_mean, "se_mean": se_mean,
                            "ci_lower_mean": sample_mean - 1.96 * se_mean,
                            "ci_upper_mean": sample_mean + 1.96 * se_mean,
                            "sample_variance": sample_variance,
                            "fourth_central_moment": fourth_central,
                            "se_variance": se_variance,
                            "ci_lower_variance": sample_variance - 1.96 * se_variance,
                            "ci_upper_variance": sample_variance + 1.96 * se_variance,
                            "exact_mean": exact_mean, "exact_variance": exact_variance,
                            "se_variance_method": "plug-in fourth central moment"})

    distribution_rows: list[dict[str, object]] = []
    log_mean = np.log(cfg.s0) + (cfg.mu - 0.5 * cfg.sigma**2) * cfg.t
    log_sd = cfg.sigma * np.sqrt(cfg.t)
    dist_base = {**common, "N": cfg.distribution_steps,
                 "h": cfg.t / cfg.distribution_steps, "M": cfg.paths}
    for probability in (0.05, 0.25, 0.5, 0.75, 0.95):
        z = norm.ppf(probability)
        exact_q = float(np.exp(log_mean + log_sd * z))
        exact_density = float(norm.pdf(z) / (exact_q * log_sd))
        se_quantile = float(np.sqrt(probability * (1.0 - probability) / cfg.paths) / exact_density)
        for method, values in terminal.items():
            estimate = float(np.quantile(values, probability))
            distribution_rows.append({**dist_base, "record_type": "quantile", "method": method,
                                      "probability": probability, "estimate": estimate,
                                      "se_quantile": se_quantile,
                                      "ci_lower_quantile": estimate - 1.96 * se_quantile,
                                      "ci_upper_quantile": estimate + 1.96 * se_quantile,
                                      "exact_quantile": exact_q,
                                      "exact_density_at_quantile": exact_density,
                                      "se_quantile_method": "asymptotic exact-lognormal density"})
    # Shared bin edges include every numerical sample; no tail truncation.
    all_min = min(float(np.min(v)) for v in terminal.values())
    all_max = max(float(np.max(v)) for v in terminal.values())
    edges = np.linspace(all_min - 1e-9, all_max + 1e-9, 81)
    for method, values in terminal.items():
        counts, _ = np.histogram(values, bins=edges)
        for i, count in enumerate(counts):
            width = float(edges[i + 1] - edges[i])
            proportion = float(count / cfg.paths)
            height = proportion / width
            se_density = float(np.sqrt(proportion * (1.0 - proportion) / cfg.paths) / width)
            distribution_rows.append({**dist_base, "record_type": "histogram", "method": method,
                                      "bin_left": float(edges[i]), "bin_right": float(edges[i + 1]),
                                      "bin_count": int(count), "density": float(height),
                                      "se_density": se_density,
                                      "ci_lower_density": max(0.0, height - 1.96 * se_density),
                                      "ci_upper_density": height + 1.96 * se_density,
                                      "se_density_method": "binomial fixed-bin approximation"})

    paths = {"gbm_strong": output / "gbm_strong.csv",
             "gbm_weak": output / "gbm_weak.csv",
             "gbm_distribution": output / "gbm_distribution.csv",
             "gbm_moments": output / "gbm_moments.csv"}
    for key, rows in (("gbm_strong", strong_rows), ("gbm_weak", weak_rows),
                      ("gbm_distribution", distribution_rows), ("gbm_moments", moment_rows)):
        write_csv(paths[key], rows)
    return paths


class IncrementMoments:
    """Raw bivariate sums; safe here because increment means are near zero."""
    def __init__(self) -> None:
        self.n = 0
        self.s1 = self.s2 = self.q1 = self.q2 = self.cross = 0.0

    def add(self, w1: np.ndarray, w2: np.ndarray) -> None:
        self.n += w1.size
        self.s1 += float(np.sum(w1, dtype=np.float64))
        self.s2 += float(np.sum(w2, dtype=np.float64))
        self.q1 += float(np.sum(w1 * w1, dtype=np.float64))
        self.q2 += float(np.sum(w2 * w2, dtype=np.float64))
        self.cross += float(np.sum(w1 * w2, dtype=np.float64))

    def summary(self, n_grid: int, t: float, rho: float, seed: int) -> dict[str, object]:
        m1, m2 = self.s1 / self.n, self.s2 / self.n
        v1 = (self.q1 - self.n * m1 * m1) / (self.n - 1)
        v2 = (self.q2 - self.n * m2 * m2) / (self.n - 1)
        covariance = (self.cross - self.n * m1 * m2) / (self.n - 1)
        corr = covariance / np.sqrt(v1 * v2)
        se_mean1, se_mean2 = np.sqrt(v1 / self.n), np.sqrt(v2 / self.n)
        se_var1, se_var2 = v1 * np.sqrt(2 / (self.n - 1)), v2 * np.sqrt(2 / (self.n - 1))
        se_corr = (1.0 - rho * rho) / np.sqrt(self.n - 1)
        return {"seed": seed, "model": "nonaffine", "N": n_grid, "h": t / n_grid,
                "increment_count": self.n, "mean_W1": m1, "mean_W2": m2,
                "se_mean_W1": se_mean1, "se_mean_W2": se_mean2,
                "ci_lower_mean_W1": m1 - 1.96 * se_mean1, "ci_upper_mean_W1": m1 + 1.96 * se_mean1,
                "ci_lower_mean_W2": m2 - 1.96 * se_mean2, "ci_upper_mean_W2": m2 + 1.96 * se_mean2,
                "sample_var_W1": v1, "sample_var_W2": v2,
                "se_var_W1": se_var1, "se_var_W2": se_var2,
                "ci_lower_var_W1": v1 - 1.96 * se_var1, "ci_upper_var_W1": v1 + 1.96 * se_var1,
                "ci_lower_var_W2": v2 - 1.96 * se_var2, "ci_upper_var_W2": v2 + 1.96 * se_var2,
                "sample_correlation": corr, "se_correlation": se_corr,
                "ci_lower_correlation": corr - 1.96 * se_corr,
                "ci_upper_correlation": corr + 1.96 * se_corr,
                "target_mean_W1": 0.0, "target_mean_W2": 0.0,
                "target_var_W1": t / n_grid, "target_var_W2": t / n_grid,
                "target_correlation": rho,
                "se_method": "normal variance and asymptotic correlation"}


def run_nonaffine(cfg: NonAffineConfig, output: Path, seed: int = SEED) -> dict[str, Path]:
    """Both references and every coarse path descend from two 8192-step masters."""
    grids = tuple(dict.fromkeys((*cfg.grids, *cfg.references)))
    if any(cfg.fine_steps % n for n in grids) or cfg.references[-1] != cfg.fine_steps:
        raise ValueError("non-affine grids and references must nest in master grid")
    rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(2)[1])
    strong = {(ref, n): RunningStats() for ref in cfg.references for n in cfg.grids}
    weak = {(ref, n): RunningStats() for ref in cfg.references for n in cfg.grids}
    strong_batch_sums = {key: [] for key in strong}
    weak_batch_sums = {key: [] for key in weak}
    batch_counts: list[int] = []
    ref_strong = RunningStats()
    ref_weak = RunningStats()
    diagnostics = {n: IncrementMoments() for n in grids}
    validity = {n: {"finite_X": 0, "finite_Y": 0, "finite_S": 0,
                    "nonpositive_S": 0, "total_paths": 0} for n in grids}
    for start in range(0, cfg.paths, cfg.batch_size):
        batch = min(cfg.batch_size, cfg.paths - start)
        batch_counts.append(batch)
        b1 = normal_increments(rng, batch, cfg.fine_steps, cfg.t)
        b2 = normal_increments(rng, batch, cfg.fine_steps, cfg.t)
        terminals: dict[int, np.ndarray] = {}
        for n in grids:
            ratio = cfg.fine_steps // n
            a1 = aggregate_increments(b1, ratio)
            a2 = aggregate_increments(b2, ratio)
            w1, w2 = correlate_increments(a1, a2, cfg.rho)
            diagnostics[n].add(w1, w2)
            x, y, s = nonaffine_em_terminal(cfg, cfg.t / n, w1, w2)
            counts = validity[n]
            counts["total_paths"] += batch
            counts["finite_X"] += int(np.count_nonzero(np.isfinite(x)))
            counts["finite_Y"] += int(np.count_nonzero(np.isfinite(y)))
            counts["finite_S"] += int(np.count_nonzero(np.isfinite(s)))
            counts["nonpositive_S"] += int(np.count_nonzero(s <= 0))
            if (counts["finite_X"] != counts["total_paths"] or
                counts["finite_Y"] != counts["total_paths"] or
                counts["finite_S"] != counts["total_paths"] or
                counts["nonpositive_S"]):
                raise FloatingPointError(f"Invalid non-affine path: batch start={start}, N={n}, counts={counts}")
            terminals[n] = s
        for ref in cfg.references:
            ref_s = terminals[ref]
            ref_payoff = np.maximum(ref_s - cfg.strike, 0.0)
            for n in cfg.grids:
                absolute = np.abs(terminals[n] - ref_s)
                signed_payoff = np.maximum(terminals[n] - cfg.strike, 0.0) - ref_payoff
                strong[(ref, n)].add(absolute)
                weak[(ref, n)].add(signed_payoff)
                strong_batch_sums[(ref, n)].append(float(np.sum(absolute, dtype=np.float64)))
                weak_batch_sums[(ref, n)].append(float(np.sum(signed_payoff, dtype=np.float64)))
        r1, r2 = (terminals[n] for n in cfg.references)
        ref_strong.add(np.abs(r1 - r2))
        ref_weak.add(np.maximum(r1 - cfg.strike, 0.0) - np.maximum(r2 - cfg.strike, 0.0))

    common = {"seed": seed, "model": "nonaffine", "s0": cfg.s0, "y0": cfg.y0,
              "mu": cfg.mu, "kappa": cfg.kappa, "theta": cfg.theta, "xi": cfg.xi,
              "rho": cfg.rho, "sigma_min": cfg.sigma_min, "sigma_max": cfg.sigma_max,
              "T": cfg.t, "master_N": cfg.fine_steps, "strike": cfg.strike}
    strong_rows: list[dict[str, object]] = []
    weak_rows: list[dict[str, object]] = []
    for ref in cfg.references:
        for n in cfg.grids:
            base = {**common, "method": "EM_log", "N": n, "h": cfg.t / n,
                    "reference_N": ref, "reference_h": cfg.t / ref}
            strong_rows.append({**base, **strong[(ref, n)].summary()})
            weak_rows.append({**base, **weak[(ref, n)].summary()})
    _fit_rows(strong_rows, ("reference_N",), "strong", batch_sums=strong_batch_sums,
              batch_counts=batch_counts)
    _fit_rows(weak_rows, ("reference_N",), "weak", batch_sums=weak_batch_sums,
              batch_counts=batch_counts)
    diagnostic_rows = [{**common, **diagnostics[n].summary(n, cfg.t, cfg.rho, seed)} for n in grids]
    validity_rows = [{**common, "N": n, "h": cfg.t / n, "M": cfg.paths, **validity[n]}
                     for n in grids]
    finest_coarse = max(cfg.grids)
    coarse_ref, fine_ref = cfg.references
    fine_ref_difference = strong[(fine_ref, finest_coarse)].mean
    observed_ref_sensitivity = abs(fine_ref_difference - strong[(coarse_ref, finest_coarse)].mean)
    slopes = {int(row["reference_N"]): float(row["fitted_slope"])
              for row in strong_rows if row["fitted_slope"] != ""}
    slope_sensitivity = (abs(slopes[coarse_ref] - slopes[fine_ref])
                         if coarse_ref in slopes and fine_ref in slopes else "")
    reference_rows = [
        {**common, "measure": "absolute_price_difference", "coarse_reference_N": cfg.references[0],
         "fine_reference_N": cfg.references[1], **ref_strong.summary(),
         "finest_coarse_N": finest_coarse,
         "finest_coarse_to_fine_reference_mean_abs": fine_ref_difference,
         "reference_gap_to_coarse_discrepancy_ratio": ref_strong.mean / fine_ref_difference,
         "observed_discrepancy_change_between_references": observed_ref_sensitivity,
         "observed_strong_slope_change_between_references": slope_sensitivity},
        {**common, "measure": "signed_payoff_difference", "coarse_reference_N": cfg.references[0],
         "fine_reference_N": cfg.references[1], **ref_weak.summary()},
    ]
    paths = {"nonaffine_strong": output / "nonaffine_strong.csv",
             "nonaffine_weak": output / "nonaffine_weak.csv",
             "nonaffine_increment_diagnostics": output / "nonaffine_increment_diagnostics.csv",
             "nonaffine_path_validation": output / "nonaffine_path_validation.csv",
             "nonaffine_reference_comparison": output / "nonaffine_reference_comparison.csv"}
    for key, rows in (("nonaffine_strong", strong_rows), ("nonaffine_weak", weak_rows),
                      ("nonaffine_increment_diagnostics", diagnostic_rows),
                      ("nonaffine_path_validation", validity_rows),
                      ("nonaffine_reference_comparison", reference_rows)):
        write_csv(paths[key], rows)
    return paths
