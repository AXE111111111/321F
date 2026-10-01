import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from gbm import (gbm_analytic_bias, gbm_em_terminal, gbm_exact_mean,
                 gbm_exact_terminal, gbm_exact_variance, gbm_milstein_terminal)
from config import GBMConfig
from random_paths import aggregate_increments, normal_increments
from statistics import fit_loglog_slope, mean_se_ci


class GBMTests(unittest.TestCase):
    def test_coarse_weak_grid_and_analytic_order(self):
        cfg = GBMConfig()
        self.assertEqual(cfg.grids[:4], (2, 4, 8, 16))
        h = np.array([1/n for n in cfg.grids[:4]])
        bias = np.array([abs(gbm_analytic_bias(cfg.s0, cfg.mu, cfg.t, n))
                         for n in cfg.grids[:4]])
        slope, _ = fit_loglog_slope(h, bias, np.ones(4, dtype=bool))
        self.assertTrue(.95 < slope < 1.05)

    def test_exact_repeated_steps_and_path_coupling(self):
        d = normal_increments(np.random.default_rng(8), 100, 64, 1.0)
        exact = gbm_exact_terminal(100, .05, .2, 1, d)
        repeated = np.full(100, 100.0)
        for k in range(64):
            repeated *= np.exp((.05 - .5 * .2**2) / 64 + .2 * d[:, k])
        np.testing.assert_allclose(exact, repeated, rtol=1e-14)
        # Each solver is a deterministic function of the very same supplied array.
        np.testing.assert_array_equal(gbm_em_terminal(100, .05, .2, 1/64, d),
                                      gbm_em_terminal(100, .05, .2, 1/64, d.copy()))
        np.testing.assert_array_equal(gbm_milstein_terminal(100, .05, .2, 1/64, d),
                                      gbm_milstein_terminal(100, .05, .2, 1/64, d.copy()))

    def test_strong_slopes(self):
        fine = normal_increments(np.random.default_rng(32104), 12000, 256, 1.0)
        exact = gbm_exact_terminal(100, .05, .2, 1, fine)
        ns = (16, 32, 64, 128, 256)
        em, mil = [], []
        for n in ns:
            dw = aggregate_increments(fine, 256 // n)
            em.append(float(np.mean(np.abs(gbm_em_terminal(100, .05, .2, 1/n, dw) - exact))))
            mil.append(float(np.mean(np.abs(gbm_milstein_terminal(100, .05, .2, 1/n, dw) - exact))))
        h = np.array([1/n for n in ns])
        s_em, _ = fit_loglog_slope(h, em, np.ones(len(ns), dtype=bool))
        s_mil, _ = fit_loglog_slope(h, mil, np.ones(len(ns), dtype=bool))
        self.assertTrue(.35 < s_em < .70, s_em)
        self.assertTrue(.75 < s_mil < 1.25, s_mil)

    def test_signed_weak_bias(self):
        d = normal_increments(np.random.default_rng(45), 25000, 16, 1.0)
        exact = gbm_exact_terminal(100, .05, .2, 1, d)
        analytic = gbm_analytic_bias(100, .05, 1, 16)
        for solver in (gbm_em_terminal, gbm_milstein_terminal):
            mean, se, _, _ = mean_se_ci(solver(100, .05, .2, 1/16, d) - exact)
            self.assertLess(abs(mean - analytic), 4 * se)

    def test_moments_and_quantiles(self):
        d = normal_increments(np.random.default_rng(12), 20000, 256, 1.0)
        exact_mean = gbm_exact_mean(100, .05, 1)
        exact_var = gbm_exact_variance(100, .05, .2, 1)
        median = np.exp(np.log(100) + .05 - .5 * .2**2)
        for solver in (gbm_em_terminal, gbm_milstein_terminal):
            s = solver(100, .05, .2, 1/256, d)
            self.assertLess(abs(float(np.mean(s)) - exact_mean), 4 * np.sqrt(exact_var / len(s)))
            self.assertLess(abs(float(np.var(s, ddof=1)) - exact_var), .07 * exact_var)
            for p in (.05, .25, .5, .75, .95):
                q = median * np.exp(.2 * norm.ppf(p))
                self.assertLess(abs(float(np.quantile(s, p)) - q), 2.0)


if __name__ == "__main__":
    unittest.main()
