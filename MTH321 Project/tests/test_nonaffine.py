import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from config import NonAffineConfig
from nonaffine import nonaffine_em_terminal, volatility
from random_paths import aggregate_increments, correlate_increments, normal_increments


class NonAffineTests(unittest.TestCase):
    def test_stable_bounded_volatility(self):
        cfg = NonAffineConfig()
        values = volatility(np.array([-1000.0, -1.0, 0.0, 1.0, 1000.0]), cfg)
        self.assertTrue(np.all(np.isfinite(values)))
        self.assertTrue(np.all(values >= cfg.sigma_min))
        self.assertTrue(np.all(values <= cfg.sigma_max))
        self.assertAlmostEqual(float(values[2]), .3)

    def test_diffusion_derivative_bounds(self):
        cfg = NonAffineConfig()
        y = np.linspace(-15, 15, 1001)
        eps = 1e-5
        dg = (volatility(y + eps, cfg) - volatility(y - eps, cfg)) / (2 * eps)
        factor = lambda z: cfg.xi * np.sqrt(1.0 + z * z)
        df = (factor(y + eps) - factor(y - eps)) / (2 * eps)
        self.assertLessEqual(float(np.max(np.abs(dg))), (cfg.sigma_max - cfg.sigma_min) / 4 + 1e-8)
        self.assertLessEqual(float(np.max(np.abs(df))), cfg.xi + 1e-8)

    def test_finite_positive_and_reference_refinement(self):
        cfg = NonAffineConfig()
        rng = np.random.default_rng(32104)
        a = normal_increments(rng, 128, 8192, 1.0)
        b = normal_increments(rng, 128, 8192, 1.0)
        terminal = {}
        for n in (32, 512, 4096, 8192):
            w1, w2 = correlate_increments(aggregate_increments(a, 8192 // n),
                                          aggregate_increments(b, 8192 // n), cfg.rho)
            x, y, s = nonaffine_em_terminal(cfg, 1/n, w1, w2)
            self.assertTrue(np.all(np.isfinite(x)))
            self.assertTrue(np.all(np.isfinite(y)))
            self.assertTrue(np.all(np.isfinite(s)))
            self.assertTrue(np.all(s > 0))
            terminal[n] = s
        for ref in (4096, 8192):
            self.assertLess(np.mean(np.abs(terminal[512] - terminal[ref])),
                            np.mean(np.abs(terminal[32] - terminal[ref])))
        self.assertLess(np.mean(np.abs(terminal[4096] - terminal[8192])),
                        np.mean(np.abs(terminal[512] - terminal[8192])))


if __name__ == "__main__":
    unittest.main()
