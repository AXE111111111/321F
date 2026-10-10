import csv
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from config import GBMConfig
from experiments import run_gbm

ROOT = Path(__file__).resolve().parents[1]


class PipelineTests(unittest.TestCase):
    def test_gbm_csv_reproducibility(self):
        cfg = replace(GBMConfig(), paths=256, batch_size=64)
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            run_gbm(cfg, Path(a))
            run_gbm(cfg, Path(b))
            for name in ("gbm_strong.csv", "gbm_weak.csv", "gbm_distribution.csv", "gbm_moments.csv"):
                self.assertEqual((Path(a) / name).read_bytes(), (Path(b) / name).read_bytes())
            with (Path(a) / "gbm_moments.csv").open(newline="", encoding="utf-8") as f:
                row = next(csv.DictReader(f))
                self.assertGreater(float(row["se_mean"]), 0)
                self.assertGreater(float(row["se_variance"]), 0)
            with (Path(a) / "gbm_distribution.csv").open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
                self.assertTrue(all(float(r["se_quantile"]) > 0 for r in rows if r["record_type"] == "quantile"))

    def test_run_all_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run([sys.executable, str(ROOT / "code" / "run_all.py"),
                            "--smoke", "--output-dir", tmp], check=True,
                           capture_output=True, text=True, timeout=180)
            results = Path(tmp) / "results"
            figures = Path(tmp) / "figures"
            self.assertEqual(len(list(results.glob("*.csv"))), 9)
            self.assertEqual(len(list(figures.glob("*.pdf"))), 5)
            for path in figures.glob("*.pdf"):
                self.assertEqual(path.read_bytes()[:4], b"%PDF")
            with (results / "nonaffine_path_validation.csv").open(newline="", encoding="utf-8") as f:
                self.assertEqual(len(list(csv.DictReader(f))), 7)
            for name in ("gbm_strong", "gbm_weak", "nonaffine_strong", "nonaffine_weak"):
                with (results / f"{name}.csv").open(newline="", encoding="utf-8") as f:
                    for row in csv.DictReader(f):
                        if row["fitted_slope"]:
                            self.assertGreater(float(row["fitted_slope_se"]), 0)
                            self.assertEqual(row["fitted_slope_se_method"], "paired delete-one-batch jackknife")
                            self.assertEqual(row["fitted_slope_interval_method"],
                                             "normal approximation: slope +/- 1.96 jackknife SE; coverage not validated")
                        else:
                            self.assertEqual(row["fitted_slope_se"], "")
                            self.assertEqual(row["fitted_slope_interval_method"], "")


if __name__ == "__main__":
    unittest.main()
