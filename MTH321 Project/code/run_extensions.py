"""Run the supplementary experiments E1, E1b, E2, E3 (see extensions.py); writes results/extensions/*.csv."""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import time

from extensions import ExtensionConfig, run_extensions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1],
                        help="Base directory; CSVs go to <output-dir>/results/extensions/")
    parser.add_argument("--smoke", action="store_true", help="Tiny run for testing; not for reporting")
    args = parser.parse_args()
    cfg = ExtensionConfig()
    if args.smoke:
        cfg = replace(cfg, uncoupled_paths=2_000, uncoupled_batch=500, coarse_paths=2_000,
                      coarse_batch=500, timing_paths=500, timing_repeats=3,
                      misaligned_paths=200, misaligned_batch=100)
    out = args.output_dir.resolve() / "results" / "extensions"
    start = time.perf_counter()
    paths = run_extensions(out, cfg)
    print(f"Wrote {len(paths)} extension CSVs to {out} in {time.perf_counter()-start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
