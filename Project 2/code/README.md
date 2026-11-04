# Project 2 Direction 2 — code (Team F)

Five-point FDM for −Δu = f on (0,1)² with Dirichlet on three edges and
Neumann (β=0) or Robin (β=2) on x=1. Two closures of the flux edge:
first-order one-sided row vs ghost-point elimination.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate   # from project2/
pip install -r code/requirements.txt
cd code
python run_all.py
```

A ready venv may already exist at `../.venv`.

## What `run_all.py` reproduces

| Step | Script | Output |
|---|---|---|
| Refinement N=16…256 | `convergence.py` | `../results/convergence.csv` (orders ≈1.00 / 2.00) |
| Truncation, η(A), η(DA), scaled residual, CG on DA, κ₂(DA) / κ₁(A) | `mms_runner.py` | `../results/mms_checks.csv`, `mms_checks_log.txt` |
| Cost vs accuracy, target ‖e‖∞≤5×10⁻³ | `cost_accuracy.py` | `../results/cost_accuracy.csv`, `cost_accuracy_all.csv` |
| Figures | `plot_figures.py` | `../report/figures/{convergence,truncation,spy,solution_surface,cost_accuracy}.pdf` (+ copies under `../slides/figures/`) |
| LaTeX tables | `make_tables.py` | `../report/tables/{conv_Neumann,conv_Robin,trunc,cost}.tex` |

## Related coursework note

Week 7 Individual Challenge uses a *different* BC layout (Dirichlet on x=0 and
y=1; Neumann on y=0; Robin on x=1). Those scripts are separate from this Project 2
package and are not required to reproduce the Direction 2 results above.

## Module map

- `solver.py` — assembly, symmetrisation, norms, direct/CG solve
- `mms.py` — manufactured solution u*=e^{x+y} and predicted τ coefficients
- `ticket_d2_study.py`, `d2_extra.py` — legacy Week-6 study scripts (still runnable; `run_all.py` supersedes them)

## Notes

- Corners are owned by Dirichlet (Problem Pack).
- Residuals are compared in the row-scaled form of the report; do not compare raw residuals across closures.
- CG is applied only to the symmetrised system DAU = DF.
- Do not invent numbers: every figure/table cell comes from a CSV written by these scripts.
