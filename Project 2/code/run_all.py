"""Reproduce all Project 2 Direction 2 numbers, tables and figures.

Usage (from the package root or from code/, with numpy/scipy/matplotlib installed):
    python code/run_all.py

Steps: convergence -> MMS checks (truncation, CG, kappa, forward-error bound) ->
cost vs accuracy -> CG tolerance sweep -> figures -> LaTeX tables ->
incompatible-corner experiment (new files only). Outputs go to ../results/,
../report/figures/ and ../report/tables/ only. The legacy scripts ticket_d2_study.py and
d2_extra.py are not part of this pipeline.
"""
from __future__ import annotations
import os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)


def main():
    t0 = time.perf_counter()
    print("=== 1/7 convergence ===")
    import convergence
    convergence.run()
    print("=== 2/7 MMS checks ===")
    import mms_runner
    mms_runner.run()
    print("=== 3/7 cost vs accuracy ===")
    import cost_accuracy
    cost_accuracy.run()
    print("=== 4/7 CG tolerance sweep ===")
    import run_tolerance_sweep
    run_tolerance_sweep.run()
    print("=== 5/7 figures ===")
    import plot_figures
    plot_figures.main()
    print("=== 6/7 LaTeX tables ===")
    import make_tables
    make_tables.main()
    print("=== 7/7 incompatible-corner experiment ===")
    import run_incompatible
    run_incompatible.run()
    print(f"done in {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
