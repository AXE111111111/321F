# MTH321 Project 2, Direction 2: mixed boundary data for the Poisson equation (Team F)

Five-point finite differences for

    -Δu = f  on (0,1)²,   u = g on x=0, y=0, y=1 (Dirichlet, owns all four corners),
    u_x + β u = r  on x=1 (Neumann β=0, Robin β=2),

comparing two closures of the flux edge: a first-order one-sided derivative row and a
ghost-point elimination. Every number, table and figure in the report is produced by the
scripts in `code/` from a single command.

## Requirements

Python ≥ 3.10 with

    pip install -r code/requirements.txt     # numpy, scipy, matplotlib, pytest

No other packages are used (no scikit-fem / pyamg). The figures use the TeX Gyre Pagella
font when it is installed (TeX Live) and fall back to DejaVu Serif otherwise; numbers do not
depend on the font. The only randomised step is SciPy's onenormest estimate of κ₁(A); the submitted code fixes the NumPy seed before that call.

## Reproduce everything (from the package root, the folder that contains `code/`)

    python -m pytest -q tests        # 71 tests, about 1-2 s
    python code/run_all.py           # all results, figures and LaTeX tables, about 70-130 s
    python validation/bank_vs_solver.py   # course test-problem bank (prints the orders)

`run_all.py` writes to `results/`, `report/figures/` and `report/tables/`, plus
`code/ticket_d2_results.csv` (a backward-compatible copy of `results/convergence.csv`, read only by
the legacy `d2_extra.py`).
Timings (`t_spsolve_s`, `t_min_s`, `t_max_s`, `solve_s`) are machine-dependent. Errors,
residuals and observed orders reproduce; the onenormest κ₁ estimate is seeded, and
diagnostics from unconverged CG are not claimed to match bit for bit.

## What `run_all.py` runs

| Step | Script | Outputs | Report |
|---|---|---|---|
| 1 refinement N=16…256, both closures, both β | `convergence.py` | `results/convergence.csv` | Tables 4–5, Fig. 2 |
| 2 truncation constants, η(A), η(DA), scaled residual ρ, CG check, κ₂(DA), κ₁(A), forward-error bounds | `mms_runner.py` | `results/mms_checks.csv`, `mms_checks_log.txt`, `forward_error_bound.csv` (`d2_extra.csv` is a copy under the old name) | Table 1, Fig. 1, Sec. 5.4 |
| 3 cost vs accuracy at ‖e‖∞ ≤ 5×10⁻³, median/min/max of 5 `spsolve` runs | `cost_accuracy.py` | `results/cost_accuracy.csv`, `cost_accuracy_all.csv`, log | Table 6, Fig. 5 |
| 4 CG tolerance sweep on DA U = DF (N=64, tol 1e-4…1e-12) and fitted orders at tol 1e-8, 1e-10 | `run_tolerance_sweep.py` | `results/tolerance_sweep.csv`, `tolerance_order_check.csv`, `tolerance_order_fit.csv`, log | Tables 2–3 |
| 5 figures | `plot_figures.py` (style in `plotstyle.py`) | `report/figures/{truncation,convergence,spy,solution_surface,cost_accuracy}.pdf` | Figs 1–5 |
| 6 LaTeX tables | `make_tables.py` | `report/tables/{trunc,conv_Neumann,conv_Robin,cost,cost_stats}.tex` | Tables 1, 4–6 |
| 7 incompatible corner data, u* = Im(z Log z), N=16…512 | `run_incompatible.py` | `results/incompatible_*`, `report/figures/incompatible-*.pdf`, `report/tables/{conv,split}_incompatible_*.tex` | Tables 8–11, Figs 6–8 |
| 8 pure-Neumann study (all four edges flux), u* = e^{x+y}, N=16…512: null vector, compatibility defect, projection, bordered mean-zero gauge, pinned gauge, runs without projection, CG on DA | `run_pure_neumann.py` (module `neumann_pure.py`) | `results/pure_neumann_{nullspace,defect,convergence,fit,uncorrected,cg}.csv`, `pure_neumann_log.txt`, `report/tables/pure_neumann.tex`, `pure_neumann_stats.tex` | Sec. 6.3, Table 7 |

`run_tolerance_sweep.py` also writes `report/tables/tol_sweep.tex` and `tol_order.tex`.
Each step can be run on its own, e.g. `python code/run_incompatible.py`.

## Module map

- `code/solver.py` – vectorised COO→CSR assembly for both closures (optional coefficient a(x,y)),
  row symmetrisation D, direct and CG solves, scaled residual, discrete norms.
  Unknowns: i=1..N, j=1..N−1, index k=(j−1)N+(i−1); arrays reshape as (N−1, N), axis 0 = y.
- `code/mms.py` – smooth manufactured solution u* = e^{x+y} and predicted truncation constants.
- `code/mms_incompatible.py` – incompatible-corner solution u* = Im(z Log z), z=(1−x)+iy.
- `code/neumann_pure.py` – pure-Neumann variant (separate from `solver.py`, whose API is unchanged): all (N+1)² nodes
  unknown, index k=j(N+1)+i, ghost and first-order closures, left null vectors, projection F−c·1,
  bordered mean-zero solve, pinned solve, trapezoid-weighted norms.
- `code/convergence_plot_template.py` – course template (order fit, refinement table).
- `code/ticket_d2_study.py`, `code/d2_extra.py` – legacy Week-6 loop-assembly scripts, kept for
  traceability; not used by `run_all.py`; `d2_extra.py` writes only to `results/legacy_d2_extra/`.
- `tests/` – `test_solver.py` (33: course bank, flux signs, index map, loop-vs-vectorised
  regression), `test_mms_incompatible.py` (17), `test_tolerance_sweep.py` (5), `test_pure_neumann.py` (16: A·1=0, left null vector, rank deficiency 1,
  defect orders, projection, bordered λ, gauge uniqueness, observed orders);
  `test_problem_bank.py` is the course bank; `pytest_log.txt` is the latest run.
- `validation/bank_vs_solver.py` – course bank through our solver; output in `bank_vs_solver_log.txt`.

## Conventions

- Corners belong to the Dirichlet edges (mixed problem; in the pure-Neumann study all nodes are unknowns).
- Residuals are compared in the row-scaled form ρ = max_k |F−AU|_k / (|A||U|+|F|)_k, which is
  invariant to the h⁻¹ / h⁻² row scalings of the two closures.
- CG is applied only to the symmetrised SPD system DA U = DF, never to A.
- The first-order boundary scaling D = 1/h in `scale_vector` assumes a constant coefficient a ≡ 1 (as in all experiments); for variable a(x,y) the first-order DA is not symmetric, so such problems (bank P4) are solved with `spsolve`, not CG.
- No number in the report is typed by hand: each comes from a CSV or log written by these scripts.
