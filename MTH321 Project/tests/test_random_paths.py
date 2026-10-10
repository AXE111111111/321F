import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from config import GBMConfig, NonAffineConfig
from random_paths import (aggregate_increments, coarsen_increments, correlate_increments,
                          normal_increments)


class RandomPathTests(unittest.TestCase):
    def test_reproducibility(self):
        a = normal_increments(np.random.default_rng(32104), 8, 16, 1.0)
        b = normal_increments(np.random.default_rng(32104), 8, 16, 1.0)
        np.testing.assert_array_equal(a, b)

    def test_block_sums_and_nesting(self):
        master = normal_increments(np.random.default_rng(4), 6, 8192, 1.0)
        for ratio in (2, 16, 256):
            coarse = aggregate_increments(master, ratio)
            np.testing.assert_allclose(coarse, master.reshape(6, 8192 // ratio, ratio).sum(axis=2), atol=1e-15)
            np.testing.assert_allclose(coarse.sum(axis=1), master.sum(axis=1), atol=1e-14)

    def test_coarsen_increments_consecutive_blocks_and_master(self):
        master = np.arange(1.0, 17.0).reshape(2, 8)
        original = master.copy()
        np.testing.assert_array_equal(coarsen_increments(master, 2), [[10.0, 26.0], [42.0, 58.0]])
        np.testing.assert_array_equal(coarsen_increments(master, 4),
                                      [[3.0, 7.0, 11.0, 15.0], [19.0, 23.0, 27.0, 31.0]])
        np.testing.assert_array_equal(coarsen_increments(master, 8), original)
        np.testing.assert_array_equal(master, original)

    def test_coarsen_increments_preserves_configured_grid_reductions(self):
        rng = np.random.default_rng(19)
        for cfg in (GBMConfig(), NonAffineConfig()):
            # Cancellation exposes changes to the established direct block-sum order.
            master = np.vstack((rng.normal(size=cfg.fine_steps),
                                np.resize([1e16, 1.0, -1e16, 1.0, 1.0, 1.0, 1.0, 1.0],
                                          cfg.fine_steps)))
            for n in (*cfg.grids, *getattr(cfg, "references", ())):
                with self.subTest(master_steps=cfg.fine_steps, steps=n):
                    expected = aggregate_increments(master, cfg.fine_steps // n)
                    np.testing.assert_array_equal(coarsen_increments(master, n), expected)

    def test_coarsen_increments_rejects_invalid_grids(self):
        master = np.ones((2, 8))
        for steps in (0, -1, 1.5, 3, 16):
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                coarsen_increments(master, steps)
        for invalid in (np.array(1.0), np.ones(8), np.ones((2, 4, 2)), np.empty((2, 0))):
            with self.subTest(shape=invalid.shape), self.assertRaises(ValueError):
                coarsen_increments(invalid, 2)

    def test_correlation_moments(self):
        rng = np.random.default_rng(32104)
        a = normal_increments(rng, 500, 200, 1.0)
        b = normal_increments(rng, 500, 200, 1.0)
        w1, w2 = correlate_increments(a, b, -0.7)
        self.assertLess(abs(float(np.mean(w1))), 5 * np.sqrt(0.005 / w1.size))
        self.assertLess(abs(float(np.mean(w2))), 5 * np.sqrt(0.005 / w2.size))
        self.assertLess(abs(float(np.var(w1, ddof=1)) - 0.005), 5 * 0.005 * np.sqrt(2 / w1.size))
        self.assertLess(abs(float(np.var(w2, ddof=1)) - 0.005), 5 * 0.005 * np.sqrt(2 / w2.size))
        # v3 fix (submission package): the original assertion abs(corr) - 0.7 < 0.015 also passed for
        # corr = 0 or +0.7. The sample correlation must be close to the signed target -0.7.
        corr = float(np.corrcoef(w1.ravel(), w2.ravel())[0, 1])
        self.assertLess(abs(corr - (-0.7)), 0.015)


if __name__ == "__main__":
    unittest.main()
