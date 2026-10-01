import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from random_paths import aggregate_increments, correlate_increments, normal_increments


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

    def test_correlation_moments(self):
        rng = np.random.default_rng(32104)
        a = normal_increments(rng, 500, 200, 1.0)
        b = normal_increments(rng, 500, 200, 1.0)
        w1, w2 = correlate_increments(a, b, -0.7)
        self.assertLess(abs(float(np.mean(w1))), 5 * np.sqrt(0.005 / w1.size))
        self.assertLess(abs(float(np.mean(w2))), 5 * np.sqrt(0.005 / w2.size))
        self.assertLess(abs(float(np.var(w1, ddof=1)) - 0.005), 5 * 0.005 * np.sqrt(2 / w1.size))
        self.assertLess(abs(float(np.var(w2, ddof=1)) - 0.005), 5 * 0.005 * np.sqrt(2 / w2.size))
        self.assertLess(abs(float(np.corrcoef(w1.ravel(), w2.ravel())[0, 1])) - 0.7, 0.015)


if __name__ == "__main__":
    unittest.main()
