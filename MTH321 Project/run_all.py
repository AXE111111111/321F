"""Run every required Direction 4 experiment and regenerate CSV/PDF outputs."""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import time

from config import GBMConfig, NonAffineConfig, SEED
from experiments import run_gbm, run_nonaffine
from plotting import make_all_figures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1],
                        help="Base directory containing results/ and figures/")
    parser.add_argument("--smoke", action="store_true", help="Small deterministic test run; not for reporting")
    parser.add_argument("--plots-only", action="store_true", help="Rebuild PDFs from existing CSV files")
    args = parser.parse_args()
    gbm_cfg, nonaffine_cfg = GBMConfig(), NonAffineConfig()
    if args.smoke:
        gbm_cfg = replace(gbm_cfg, paths=256, batch_size=64)
        nonaffine_cfg = replace(nonaffine_cfg, paths=64, batch_size=16)
    root = args.output_dir.resolve()
    result_dir, figure_dir = root / "results", root / "figures"
    print(f"Seed={SEED}; output={root}; smoke={args.smoke}", flush=True)
    if args.plots_only:
        figures = make_all_figures(result_dir, figure_dir)
        print(f"Regenerated {len(figures)} PDF figures from saved CSV files", flush=True)
        return
    start = time.perf_counter()
    print(f"GBM: M={gbm_cfg.paths:,}", flush=True)
    run_gbm(gbm_cfg, result_dir)
    print(f"GBM finished in {time.perf_counter()-start:.1f}s", flush=True)
    print(f"Non-affine: M={nonaffine_cfg.paths:,}", flush=True)
    run_nonaffine(nonaffine_cfg, result_dir)
    print(f"Non-affine finished in {time.perf_counter()-start:.1f}s", flush=True)
    figures = make_all_figures(result_dir, figure_dir)
    print(f"Generated {len(list(result_dir.glob('*.csv')))} CSV files and {len(figures)} PDF figures "
          f"in {time.perf_counter()-start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
