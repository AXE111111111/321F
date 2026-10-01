import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from statistics import RunningStats, fit_loglog_slope, jackknife_slope_se, mean_se_ci, weak_fit_eligible
from experiments import _fit_rows, IncrementMoments


class StatisticsTests(unittest.TestCase):
    def test_batch_combination_and_ddof(self):
        values = np.random.default_rng(2).normal(size=10003)
        stats = RunningStats()
        for chunk in np.array_split(values, 17):
            stats.add(chunk)
        self.assertAlmostEqual(stats.mean, float(np.mean(values)), places=14)
        self.assertAlmostEqual(stats.sd, float(np.std(values, ddof=1)), places=14)
        mean, se, lower, upper = mean_se_ci(values)
        self.assertAlmostEqual(mean, stats.mean, places=14)
        self.assertAlmostEqual(se, stats.se, places=14)
        self.assertAlmostEqual(upper - lower, 3.92 * se)

    def test_weak_fit_gate_and_slope(self):
        self.assertTrue(weak_fit_eligible({"estimate": -2, "ci_lower": -2.4, "ci_upper": -1.6}))
        self.assertFalse(weak_fit_eligible({"estimate": .1, "ci_lower": -.1, "ci_upper": .3}))
        self.assertFalse(weak_fit_eligible({"estimate": 1, "ci_lower": .4, "ci_upper": 1.6}))
        slope, _ = fit_loglog_slope([.25, .125, .0625], [.25, .125, .0625], [True, True, True])
        self.assertAlmostEqual(slope, 1.0)

    def test_fit_requires_four_resolved_points(self):
        rows = [{"method": "EM", "N": n, "h": 1/n, "estimate": -1/n,
                 "ci_lower": -1.1/n, "ci_upper": -.9/n} for n in (2, 4)]
        _fit_rows(rows, ("method",), "weak")
        self.assertEqual(rows[0]["eligible_point_count"], 2)
        self.assertEqual(rows[0]["fit_point_count"], 0)
        self.assertEqual(rows[0]["fitted_slope"], "")
        rows.extend({"method": "EM", "N": n, "h": 1/n, "estimate": -1/n,
                     "ci_lower": -1.1/n, "ci_upper": -.9/n} for n in (8, 16))
        _fit_rows(rows, ("method",), "weak")
        self.assertEqual(rows[0]["fit_point_count"], 4)
        self.assertAlmostEqual(rows[0]["fitted_slope"], 1.0)

    def test_increment_uncertainty_columns(self):
        rng = np.random.default_rng(4)
        w1 = rng.normal(size=(100, 20))
        w2 = -.7 * w1 + np.sqrt(1-.7**2) * rng.normal(size=(100, 20))
        stats = IncrementMoments()
        stats.add(w1, w2)
        row = stats.summary(20, 20, -.7, 4)
        for key in ("se_mean_W1", "se_mean_W2", "se_var_W1", "se_var_W2", "se_correlation"):
            self.assertGreater(row[key], 0)

    def test_paired_batch_jackknife(self):
        h = np.array([.5, .25, .125, .0625])
        counts = np.array([80, 100, 120, 90, 110])
        perturbation = np.array([[.04, -.02, .01, -.03, .02],
                                 [-.01, .03, -.02, .01, -.04],
                                 [.02, -.01, .04, -.02, .01],
                                 [-.03, .02, -.01, .03, -.02]])
        sums = h[:, None] * counts[None, :] * (1.0 + perturbation)
        full_means = sums.sum(axis=1) / counts.sum()
        full_slope, _ = fit_loglog_slope(h, full_means, np.ones(4, dtype=bool))
        se = jackknife_slope_se(h, sums, counts, full_slope)
        self.assertGreater(se, 0)
        self.assertLess(se, .1)
        equal_counts = np.full(5, 100)
        equal_sums = h[:, None] * equal_counts[None, :] * (1.0 + perturbation)
        equal_slope, _ = fit_loglog_slope(h, equal_sums.sum(axis=1) / equal_counts.sum(),
                                         np.ones(4, dtype=bool))
        observed = jackknife_slope_se(h, equal_sums, equal_counts, equal_slope)
        leave_slopes = []
        for j in range(5):
            leave_means = (equal_sums.sum(axis=1) - equal_sums[:, j]) / 400
            leave_slopes.append(fit_loglog_slope(h, leave_means, np.ones(4, dtype=bool))[0])
        expected = np.sqrt(4 / 5 * np.sum((leave_slopes - np.mean(leave_slopes))**2))
        self.assertAlmostEqual(observed, expected)


if __name__ == "__main__":
    unittest.main()
