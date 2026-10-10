"""Small supplementary experiments added for the final report (Team F).

These experiments reuse the team's solvers unchanged (gbm.py, nonaffine.py,
random_paths.py, statistics.py) and write CSVs to ``results/extensions/``.
They are kept separate from run_all.py so that the main pipeline and its
nine CSVs / five figures are untouched.

E1  Falsification: strong error measured with *uncoupled* Brownian paths
    (numerical path and exact path driven by independent increments) does not
    decrease with h, whereas the coupled error does.
E2  Non-affine model on coarse grids N=2..64 with more paths: (a) the log-EM
    mean oracle E[S_N] = S0 exp(mu T) (exact for every h), (b) paired call-payoff
    differences against two EM references (512 and 1024 steps).
E1b Sensitivity check: a deliberately *misaligned* (strided) aggregation of the
    fine increments, for GBM and for the non-affine model. GBM strong errors
    are insensitive to this bug (they depend on the path only through W_T and
    sum dW^2); the non-affine coupled discrepancy is not.
E3  Indicative wall-clock cost of GBM EM and Milstein (both at N=8 and N=512),
    with column-contiguous increment arrays so that the per-step column access
    of the solvers is equally cache friendly for both. Timings are machine
    dependent and are NOT byte-reproducible.
"""
from __future__ import annotations

import platform
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from config import GBMConfig, NonAffineConfig, SEED
from experiments import _fit_rows, write_csv
from gbm import gbm_em_terminal, gbm_exact_terminal, gbm_milstein_terminal
from nonaffine import nonaffine_em_terminal
from random_paths import aggregate_increments, correlate_increments, normal_increments
from statistics import RunningStats, fit_loglog_slope, jackknife_slope_se

# Children 2 and 3 of the master SeedSequence; children 0 and 1 drive the
# main GBM and non-affine experiments, so these streams are independent of them.
# spawn(n)[k] depends only on k, so adding children 4 and 5 leaves E1/E2 streams unchanged.
_UNCOUPLED_CHILD, _COARSE_CHILD, _TIMING_CHILD, _MISALIGNED_CHILD, _N_CHILDREN = 2, 3, 4, 5, 6


@dataclass(frozen=True)
class ExtensionConfig:
    uncoupled_paths: int = 200_000
    uncoupled_batch: int = 10_000
    uncoupled_fine_steps: int = 512
    uncoupled_grids: tuple[int, ...] = (8, 32, 128, 512)
    coarse_paths: int = 400_000
    coarse_batch: int = 10_000
    coarse_fine_steps: int = 1_024
    coarse_grids: tuple[int, ...] = (2, 4, 8, 16, 32, 64)
    coarse_references: tuple[int, ...] = (512, 1_024)
    timing_paths: int = 10_000
    timing_repeats: int = 7
    misaligned_paths: int = 10_000
    misaligned_batch: int = 1_000
    misaligned_gbm_fine_steps: int = 512
    misaligned_nonaffine_fine_steps: int = 4_096
    misaligned_grids: tuple[int, ...] = (32, 64, 128, 256, 512)


def _rng(child: int, seed: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(seed).spawn(_N_CHILDREN)[child])


def run_uncoupled(cfg: ExtensionConfig, gbm: GBMConfig, output: Path, seed: int = SEED) -> Path:
    """E1: coupled vs uncoupled strong error for GBM EM and Milstein."""
    rng = _rng(_UNCOUPLED_CHILD, seed)
    stats = {(c, m, n): RunningStats() for c in ("coupled", "uncoupled")
             for m in ("EM", "Milstein") for n in cfg.uncoupled_grids}
    for start in range(0, cfg.uncoupled_paths, cfg.uncoupled_batch):
        batch = min(cfg.uncoupled_batch, cfg.uncoupled_paths - start)
        fine = normal_increments(rng, batch, cfg.uncoupled_fine_steps, gbm.t)
        other = normal_increments(rng, batch, cfg.uncoupled_fine_steps, gbm.t)  # independent path
        exact = gbm_exact_terminal(gbm.s0, gbm.mu, gbm.sigma, gbm.t, fine)
        for n in cfg.uncoupled_grids:
            h = gbm.t / n
            for coupling, source in (("coupled", fine), ("uncoupled", other)):
                dw = aggregate_increments(source, cfg.uncoupled_fine_steps // n)
                for method, solver in (("EM", gbm_em_terminal), ("Milstein", gbm_milstein_terminal)):
                    stats[(coupling, method, n)].add(np.abs(solver(gbm.s0, gbm.mu, gbm.sigma, h, dw) - exact))
    rows: list[dict[str, object]] = []
    for coupling in ("coupled", "uncoupled"):
        for method in ("EM", "Milstein"):
            group = []
            for n in cfg.uncoupled_grids:
                row = {"seed": seed, "experiment": "E1_uncoupled_strong", "coupling": coupling,
                       "method": method, "N": n, "h": gbm.t / n,
                       **stats[(coupling, method, n)].summary()}
                group.append(row)
            h = np.array([r["h"] for r in group], dtype=np.float64)
            e = np.array([r["estimate"] for r in group], dtype=np.float64)
            slope, _ = fit_loglog_slope(h, e, np.ones(h.size, dtype=bool))
            for row in group:
                row["ols_slope_all_grids"] = slope
                row["ratio_coarsest_to_finest"] = float(e[0] / e[-1])
            rows.extend(group)
    path = output / "ext_uncoupled_strong.csv"
    write_csv(path, rows)
    return path


def run_coarse_nonaffine(cfg: ExtensionConfig, params: NonAffineConfig, output: Path,
                         seed: int = SEED) -> Path:
    """E2: log-EM mean oracle and coarse-grid paired call-payoff differences."""
    grids = tuple(dict.fromkeys((*cfg.coarse_grids, *cfg.coarse_references)))
    if any(cfg.coarse_fine_steps % n for n in grids):
        raise ValueError("grids must nest in the fine grid")
    rng = _rng(_COARSE_CHILD, seed)
    exact_mean = float(params.s0 * np.exp(params.mu * params.t))
    mean_stats = {n: RunningStats() for n in grids}
    payoff_level = {n: RunningStats() for n in grids}
    paired_mean = {n: RunningStats() for n in grids}  # S_N - S_ref on the same path; exact mean 0
    ref_payoff_gap = RunningStats()  # phi(S_ref_coarse) - phi(S_ref_fine): accuracy of the references
    weak = {(r, n): RunningStats() for r in cfg.coarse_references for n in cfg.coarse_grids}
    weak_sums = {(r, n): [] for r in cfg.coarse_references for n in cfg.coarse_grids}
    batch_counts: list[int] = []
    for start in range(0, cfg.coarse_paths, cfg.coarse_batch):
        batch = min(cfg.coarse_batch, cfg.coarse_paths - start)
        batch_counts.append(batch)
        b1 = normal_increments(rng, batch, cfg.coarse_fine_steps, params.t)
        b2 = normal_increments(rng, batch, cfg.coarse_fine_steps, params.t)
        payoff: dict[int, np.ndarray] = {}
        prices: dict[int, np.ndarray] = {}
        for n in grids:
            ratio = cfg.coarse_fine_steps // n
            w1, w2 = correlate_increments(aggregate_increments(b1, ratio),
                                          aggregate_increments(b2, ratio), params.rho)
            x, y, s = nonaffine_em_terminal(params, params.t / n, w1, w2)
            if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y)) and np.all(s > 0)):
                raise FloatingPointError(f"invalid path at N={n}")
            mean_stats[n].add(s - exact_mean)
            prices[n] = s
            payoff[n] = np.maximum(s - params.strike, 0.0)
            payoff_level[n].add(payoff[n])
        finest = prices[cfg.coarse_references[-1]]
        ref_payoff_gap.add(payoff[cfg.coarse_references[0]] - payoff[cfg.coarse_references[-1]])
        for n in grids:
            paired_mean[n].add(prices[n] - finest)
        for ref in cfg.coarse_references:
            for n in cfg.coarse_grids:
                d = payoff[n] - payoff[ref]
                weak[(ref, n)].add(d)
                weak_sums[(ref, n)].append(float(np.sum(d, dtype=np.float64)))
    common = {"seed": seed, "experiment": "E2_nonaffine_coarse", "fine_N": cfg.coarse_fine_steps}
    rows: list[dict[str, object]] = []
    for ref in cfg.coarse_references:
        for n in cfg.coarse_grids:
            rows.append({**common, "record_type": "payoff_difference", "reference_N": ref,
                         "N": n, "h": params.t / n, **weak[(ref, n)].summary()})
    # Team's own eligibility rule and paired jackknife are reused unchanged.
    _fit_rows(rows, ("reference_N",), "weak", batch_sums=weak_sums, batch_counts=batch_counts)
    for n in grids:
        m = mean_stats[n].summary()
        pm = paired_mean[n]
        pm_lo, pm_hi = pm.mean - 1.96 * pm.se, pm.mean + 1.96 * pm.se
        rows.append({**common, "record_type": "mean_oracle", "reference_N": "", "N": n,
                     "h": params.t / n, **m, "exact_mean": exact_mean,
                     "oracle_in_ci": int(float(m["ci_lower"]) <= 0.0 <= float(m["ci_upper"])),
                     "z_score": float(m["estimate"]) / float(m["se"]),
                     "payoff_mean": payoff_level[n].mean, "payoff_se": payoff_level[n].se,
                     "paired_mean_diff_vs_finest": pm.mean, "paired_mean_diff_se": pm.se,
                     "paired_ci_lower": pm_lo, "paired_ci_upper": pm_hi,
                     "paired_zero_in_ci": int(pm_lo <= 0.0 <= pm_hi) if n != cfg.coarse_references[-1] else ""})
    # Sensitivity check: refit both references on the grids selected for *both* of them, so that
    # comparing the two slopes isolates the reference (the fit rule alone may select different ranges).
    fitted = {ref: {int(r["N"]) for r in rows if r["record_type"] == "payoff_difference"
                    and r["reference_N"] == ref and int(r["fit_included"])} for ref in cfg.coarse_references}
    shared = sorted(set.intersection(*fitted.values()))
    if len(shared) >= 4:
        h_c = np.array([params.t / n for n in shared], dtype=np.float64)
        for ref in cfg.coarse_references:
            est = np.array([abs(weak[(ref, n)].mean) for n in shared], dtype=np.float64)
            slope_c, _ = fit_loglog_slope(h_c, est, np.ones(h_c.size, dtype=bool))
            se_c = jackknife_slope_se(h_c, np.array([weak_sums[(ref, n)] for n in shared]), batch_counts, slope_c)
            rows.append({**common, "record_type": "common_range_fit", "reference_N": ref,
                         "N": "-".join(str(n) for n in shared), "fitted_slope": slope_c, "fitted_slope_se": se_c,
                         "fitted_slope_se_method": "paired delete-one-batch jackknife",
                         "fit_point_count": len(shared), "jackknife_batch_count": len(batch_counts)})
    rows.append({**common, "record_type": "reference_payoff_gap",
                 "reference_N": cfg.coarse_references[-1], "N": cfg.coarse_references[0],
                 "h": params.t / cfg.coarse_references[0], **ref_payoff_gap.summary()})
    path = output / "ext_nonaffine_coarse.csv"
    write_csv(path, rows)
    return path


def run_timing(cfg: ExtensionConfig, gbm: GBMConfig, output: Path, seed: int = SEED) -> Path:
    """E3: median wall-clock of the solver loops, EM and Milstein at N=8 and N=512.

    A second pass repeats the timing with row-major (C-ordered) arrays, the v1 setting, as a controlled
    layout comparison. The solvers read one column dW[:, k] per step. A C-ordered (paths, steps)
    array makes that access strided (stride = steps * 8 bytes), which penalises
    the many-column N=512 array far more than the 8-column N=8 array. The arrays
    are therefore converted to column-contiguous (Fortran) order before timing,
    so both grids get contiguous column access. Increments are generated outside
    the timed region. Machine dependent by construction.
    """
    rng = _rng(_TIMING_CHILD, seed)
    fine = normal_increments(rng, cfg.timing_paths, 512, gbm.t)
    matched = {("EM", 512), ("Milstein", 8)}  # matched-accuracy grids of the main report
    rows = []
    layouts = (("column-contiguous (Fortran order)", np.asfortranarray),
               ("row-major (C order), v1 setting", np.ascontiguousarray))
    for (layout, to_layout), n in ((lay, n) for lay in layouts for n in (8, 512)):
        dw = to_layout(aggregate_increments(fine, 512 // n))
        for method, solver in (("EM", gbm_em_terminal), ("Milstein", gbm_milstein_terminal)):
            solver(gbm.s0, gbm.mu, gbm.sigma, gbm.t / n, dw)  # warm-up, not timed
            times = []
            for _ in range(cfg.timing_repeats):
                t0 = time.perf_counter()
                solver(gbm.s0, gbm.mu, gbm.sigma, gbm.t / n, dw)
                times.append(time.perf_counter() - t0)
            rows.append({"experiment": "E3_wallclock", "method": method, "N": n,
                         "matched_accuracy_grid": int((method, n) in matched),
                         "array_layout": layout,
                         "paths": cfg.timing_paths, "repeats": cfg.timing_repeats,
                         "median_seconds": float(np.median(times)), "min_seconds": float(np.min(times)),
                         "max_seconds": float(np.max(times)),
                         "seconds_per_path_step": float(np.median(times)) / (cfg.timing_paths * n),
                         "python": platform.python_version(), "numpy": np.__version__,
                         "machine": platform.machine(), "note": "machine dependent; not byte-reproducible"})
    path = output / "ext_wallclock.csv"
    write_csv(path, rows)
    return path


def misaligned_aggregate(d_fine: np.ndarray, ratio: int) -> np.ndarray:
    """Deliberately WRONG coupling used only for E1b: coarse step j sums fine steps
    j, j+N, j+2N, ... (N = coarse steps) instead of the consecutive block jR..jR+R-1.
    Each coarse increment is still N(0, h) and the total is still W_T."""
    paths, steps = d_fine.shape
    return np.asarray(d_fine.reshape(paths, ratio, steps // ratio).sum(axis=1), dtype=np.float64)


def run_misaligned(cfg: ExtensionConfig, gbm: GBMConfig, params: NonAffineConfig, output: Path,
                   seed: int = SEED) -> Path:
    """E1b: strong error/discrepancy with correct vs misaligned aggregation."""
    rng = _rng(_MISALIGNED_CHILD, seed)
    keys = [(model, agg, n) for model in ("GBM_EM", "GBM_Milstein", "nonaffine_EM")
            for agg in ("correct", "misaligned") for n in cfg.misaligned_grids]
    stats = {k: RunningStats() for k in keys}
    aggregators = {"correct": aggregate_increments, "misaligned": misaligned_aggregate}
    fg, fn = cfg.misaligned_gbm_fine_steps, cfg.misaligned_nonaffine_fine_steps
    for start in range(0, cfg.misaligned_paths, cfg.misaligned_batch):
        batch = min(cfg.misaligned_batch, cfg.misaligned_paths - start)
        g = normal_increments(rng, batch, fg, gbm.t)
        exact = gbm_exact_terminal(gbm.s0, gbm.mu, gbm.sigma, gbm.t, g)
        b1 = normal_increments(rng, batch, fn, params.t)
        b2 = normal_increments(rng, batch, fn, params.t)
        w1f, w2f = correlate_increments(b1, b2, params.rho)
        _, _, ref = nonaffine_em_terminal(params, params.t / fn, w1f, w2f)  # fine EM reference
        for agg, fun in aggregators.items():
            for n in cfg.misaligned_grids:
                dw = fun(g, fg // n)
                stats[("GBM_EM", agg, n)].add(np.abs(gbm_em_terminal(gbm.s0, gbm.mu, gbm.sigma, gbm.t / n, dw) - exact))
                stats[("GBM_Milstein", agg, n)].add(
                    np.abs(gbm_milstein_terminal(gbm.s0, gbm.mu, gbm.sigma, gbm.t / n, dw) - exact))
                w1, w2 = correlate_increments(fun(b1, fn // n), fun(b2, fn // n), params.rho)
                _, _, s = nonaffine_em_terminal(params, params.t / n, w1, w2)
                stats[("nonaffine_EM", agg, n)].add(np.abs(s - ref))
    rows: list[dict[str, object]] = []
    for model in ("GBM_EM", "GBM_Milstein", "nonaffine_EM"):
        for agg in ("correct", "misaligned"):
            group = [{"seed": seed, "experiment": "E1b_misaligned_aggregation", "model": model,
                      "aggregation": agg, "reference": "exact" if model.startswith("GBM") else f"EM N={fn}",
                      "N": n, "h": 1.0 / n, **stats[(model, agg, n)].summary()} for n in cfg.misaligned_grids]
            h = np.array([r["h"] for r in group]); e = np.array([r["estimate"] for r in group])
            slope, _ = fit_loglog_slope(h, e, np.ones(h.size, dtype=bool))
            for r in group:
                r["ols_slope_all_grids"] = slope
            rows.extend(group)
    path = output / "ext_misaligned.csv"
    write_csv(path, rows)
    return path


def run_extensions(output: Path, cfg: ExtensionConfig | None = None, seed: int = SEED) -> list[Path]:
    cfg = cfg or ExtensionConfig()
    output.mkdir(parents=True, exist_ok=True)
    return [run_uncoupled(cfg, GBMConfig(), output, seed),
            run_coarse_nonaffine(cfg, NonAffineConfig(), output, seed),
            run_timing(cfg, GBMConfig(), output, seed),
            run_misaligned(cfg, GBMConfig(), NonAffineConfig(), output, seed)]
