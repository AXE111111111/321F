import csv
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from config import GBMConfig, NonAffineConfig
from extensions import (ExtensionConfig, misaligned_aggregate, run_coarse_nonaffine,
                        run_misaligned, run_uncoupled)
from random_paths import aggregate_increments

ROOT = Path(__file__).resolve().parents[1]
SMALL = replace(ExtensionConfig(), uncoupled_paths=4_000, uncoupled_batch=1_000,
                coarse_paths=4_000, coarse_batch=1_000)


def _rows(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class ExtensionTests(unittest.TestCase):
    def test_uncoupled_strong_error_does_not_converge(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = _rows(run_uncoupled(SMALL, GBMConfig(), Path(tmp)))
        ratio = {(r["coupling"], r["method"]): float(r["ratio_coarsest_to_finest"]) for r in rows}
        # Coupled EM error falls by about sqrt(64)=8 from N=8 to N=512; uncoupled stays flat.
        self.assertGreater(ratio[("coupled", "EM")], 4.0)
        self.assertGreater(ratio[("coupled", "Milstein")], 30.0)
        for method in ("EM", "Milstein"):
            self.assertLess(abs(ratio[("uncoupled", method)] - 1.0), 0.2)

    def test_log_em_mean_oracle_and_payoff_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = _rows(run_coarse_nonaffine(SMALL, NonAffineConfig(), Path(tmp)))
        oracle = [r for r in rows if r["record_type"] == "mean_oracle"]
        self.assertEqual(len(oracle), 8)
        for r in oracle:
            self.assertLess(abs(float(r["z_score"])), 4.0)
            if r["paired_zero_in_ci"] != "":
                se = float(r["paired_mean_diff_se"])
                self.assertLess(abs(float(r["paired_mean_diff_vs_finest"])), 4.0 * se)
        payoff = [r for r in rows if r["record_type"] == "payoff_difference"]
        self.assertEqual(len(payoff), 12)
        self.assertEqual(len([r for r in rows if r["record_type"] == "reference_payoff_gap"]), 1)

    def test_correlate_increments_deterministic_cholesky(self):
        # Hand-computed check of dW1 = dB1, dW2 = rho dB1 + sqrt(1 - rho^2) dB2 (sign included).
        from random_paths import correlate_increments
        b1 = np.array([[1.0, -2.0]]); b2 = np.array([[3.0, 0.5]])
        w1, w2 = correlate_increments(b1, b2, -0.7)
        np.testing.assert_allclose(w1, b1)
        np.testing.assert_allclose(w2, -0.7 * b1 + np.sqrt(0.51) * b2, rtol=1e-15)

    def test_misaligned_aggregation_is_strided_and_detected_only_by_nonaffine(self):
        d = np.arange(8.0).reshape(1, 8)
        np.testing.assert_array_equal(aggregate_increments(d, 4), [[6.0, 22.0]])   # (0+1+2+3, 4+5+6+7)
        np.testing.assert_array_equal(misaligned_aggregate(d, 4), [[12.0, 16.0]])  # (0+2+4+6, 1+3+5+7)
        cfg = replace(SMALL, misaligned_paths=1_000, misaligned_batch=500)
        with tempfile.TemporaryDirectory() as tmp:
            rows = _rows(run_misaligned(cfg, GBMConfig(), NonAffineConfig(), Path(tmp)))
        slope = {(r["model"], r["aggregation"]): float(r["ols_slope_all_grids"]) for r in rows}
        # Non-affine coupling bug collapses the refinement slope; GBM is insensitive to it.
        self.assertGreater(slope[("nonaffine_EM", "correct")], 0.35)
        self.assertLess(slope[("nonaffine_EM", "misaligned")], 0.2)
        self.assertLess(abs(slope[("GBM_EM", "correct")] - slope[("GBM_EM", "misaligned")]), 0.1)

    def test_log_em_one_step_martingale_identity(self):
        # E[exp((mu - g^2/2) h + g dW)] = exp(mu h) for any g: checked by Gauss-Hermite quadrature.
        z, w = np.polynomial.hermite_e.hermegauss(60)
        w = w / w.sum()
        mu, h = 0.05, 0.5
        for g in (0.1, 0.28, 0.5):
            value = float(np.dot(w, np.exp((mu - 0.5 * g * g) * h + g * np.sqrt(h) * z)))
            self.assertAlmostEqual(value, np.exp(mu * h), places=12)

    def test_run_extensions_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run([sys.executable, str(ROOT / "code" / "run_extensions.py"), "--smoke",
                            "--output-dir", tmp], check=True, capture_output=True, text=True, timeout=180)
            names = sorted(p.name for p in (Path(tmp) / "results" / "extensions").glob("*.csv"))
            self.assertEqual(names, ["ext_misaligned.csv", "ext_nonaffine_coarse.csv",
                                     "ext_uncoupled_strong.csv", "ext_wallclock.csv"])


if __name__ == "__main__":
    unittest.main()
