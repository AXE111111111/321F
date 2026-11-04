"""Reproduce all Project 2 Direction 2 numbers, tables and figures.

Usage (from this directory, with numpy/scipy/matplotlib installed):
    python run_all.py

Steps: convergence → MMS checks (truncation, CG, κ) → cost vs accuracy →
figures (convergence with template fit, truncation, spy, solution surface, cost) → LaTeX tables. Legacy scripts ticket_d2_study.py / d2_extra.py remain
for reference; this driver is the supported path.
"""
from __future__ import annotations
import os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)


def main():
    t0 = time.perf_counter()
    print("=== 1/5 convergence ===")
    import convergence
    convergence.run()
    print("=== 2/5 MMS checks ===")
    import mms_runner
    mms_runner.run()
    print("=== 3/5 cost vs accuracy ===")
    import cost_accuracy
    cost_accuracy.run()
    print("=== 4/5 figures ===")
    import plot_figures
    plot_figures.main()
    print("=== 5/5 LaTeX tables ===")
    import make_tables
    make_tables.main()
    print(f"done in {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
