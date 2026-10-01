# MTH321 Direction 4: Financial SDE numerical implementation

This directory implements the numerical part of Direction 4. The supplied sources are three PDFs (`ODE_IVP_Project_Brief.pdf`, `problem_pack (5).pdf`, and `Direction_4_Coding_Implementation_Guide.pdf`) plus the theory chapter `mathematical_theory.tex`. This code adds no deterministic ODE experiments. The supplied sources are not generated outputs.

## Environment and reproduction

Use Python 3.10+ with NumPy, SciPy, and Matplotlib. From this directory:

```powershell
python code/run_all.py
python -m unittest discover -s tests -v
```

On the course machine, `..\.venv\Scripts\python.exe` can replace `python`. A quick integration check is `python code/run_all.py --smoke --output-dir <temporary-directory>`; its small samples must not be reported as final results. `python code/run_all.py --plots-only` redraws the five PDF figures from existing CSV files without changing numerical data.

## Report and presentation

The English report is `report/MTH321_Direction4_Report.pdf`; its main source is `report/main.tex`, which includes the supplied `mathematical_theory.tex` unchanged. It contains background, models, mathematical methods, the Direction 4 replacement for deterministic stability/adaptivity, experiments, conclusions, an AI transparency log and five individual contribution forms. The presentation is `slides/MTH321_Direction4_Slides.pdf` with editable Beamer source `slides/main.tex`. `slides/speaker_notes.md` gives a ten-minute rehearsal plan and a three-minute Q&A preparation section.

**Team completion still required:** identities, student IDs, actual contributions and signatures were not provided. The five ICS forms are unsigned, and the AI appendix explicitly identifies the manual checks and any additional tools the members must confirm. Fill `report/team_details.tex` and the contribution/AI appendix sources from the team's actual records before submitting. The course's separate report template and detailed ICS rubric were not supplied; use any mandatory declaration wording from those materials.

Rebuild the documents after regenerating the numerical data (a LaTeX installation with Beamer is required):

```powershell
python report/build_evidence.py
cd report
pdflatex -interaction=nonstopmode -halt-on-error -jobname=MTH321_Direction4_Report main.tex
pdflatex -interaction=nonstopmode -halt-on-error -jobname=MTH321_Direction4_Report main.tex
cd ../slides
pdflatex -interaction=nonstopmode -halt-on-error -jobname=MTH321_Direction4_Slides main.tex
pdflatex -interaction=nonstopmode -halt-on-error -jobname=MTH321_Direction4_Slides main.tex
```

`report/build_evidence.py` reads the nine saved CSVs, derives the report's tables and numeric macros, and records source hashes in `report/data/provenance.json`. It does not simulate or edit numerical results. The matched-accuracy cost table counts path updates M*N from saved GBM rows, not wall-clock time. The report and slides use compiled LaTeX mathematics and the existing five figure PDFs.

`FINAL_SELF_CHECK.md` records the third-round checks and remaining team-only information. Optional read-only document/evidence checks are available through `python report/check_delivery.py` (requires pypdf). After all team edits and document recompilation, `python report/build_submission.py` rebuilds the named submission ZIP from an explicit allowlist and verifies every entry against the current file before replacing the archive.

The default full run overwrites only generated files under `results/` and `figures/`. The master seed is 32104. Separate `SeedSequence` children drive GBM and the non-affine model; solvers receive increments explicitly and never draw random numbers. CSV results under `results/` are byte-reproducible with the same software, path counts and batch sizes. Figure PDFs may differ in bytes because Matplotlib writes creation timestamps. Changing the batch size changes which random numbers are assigned to paths but not the sampling law.

## Methods and interpretation

- GBM uses $S_0=100,\ \mu=0.05,\ \sigma=0.2,\ T=1$ and 2,000,000 paths in batches of 10,000. Each batch has one 512-step Brownian master path per trajectory. Every grid $N\in\{2,4,8,16,32,64,128,256,512\}$ is a block-sum descendant. The exact terminal price, Euler--Maruyama (EM) and Milstein all use the same path. Strong errors are pathwise absolute differences; weak biases are **signed paired** differences. The numerical EM and Milstein terminal distributions at $N=256$ are compared with the exact lognormal law.
- The non-affine model uses the parameters in the supplied problem, 50,000 paths in batches of 1,000, and two independent 8,192-step master streams. Independent streams are aggregated before correlation is imposed. All coarse grids and the 4,096/8,192-step numerical references are coupled descendants. EM advances $(X,Y)$ and reconstructs $S=e^X$. Any non-finite state or non-positive terminal price triggers an explicit error; paths are never clipped or dropped.
- Cross-batch statistics use a stable count/mean/centred-second-moment combination. Sample standard deviations use one degree-of-freedom correction. Reported mean intervals are approximately $\widehat m\pm1.96\,\mathrm{SE}$. The GBM strong fit range is fixed in advance at $N\ge64$; weak points must first pass the signed confidence-interval and relative-half-width criteria. A slope is reported only when at least four individually resolved points are available. Non-affine slopes are **empirical refinement slopes**, not exact-error orders.

Additional uncertainty columns make the supporting checks interpretable. The sample-variance standard error is a fourth-central-moment plug-in estimate; quantile standard errors use the asymptotic formula with the exact lognormal density at the exact quantile; histogram density errors use a fixed-bin binomial approximation. Increment-variance and correlation standard errors use normal/asymptotic formulas. These are approximations, not simultaneous confidence bands, and the histogram bin edges themselves are data-dependent.

Fitted-slope uncertainty uses a delete-one-batch jackknife over the same paired batches at every grid (200 GBM batches, 50 non-affine batches). The fit-point mask is held fixed for each deletion. Thus `fitted_slope_se` captures Monte Carlo variation and cross-grid dependence conditional on the chosen mask; it does not quantify time-discretisation bias, reference error or uncertainty from selecting the fit range. The existing `fitted_slope_ci_lower` and `fitted_slope_ci_upper` columns give the normal approximation slope +/- 1.96 jackknife SE. Their companion `fitted_slope_interval_method` explicitly records that coverage has not been validated; these bounds are not calibrated confidence intervals for the true asymptotic order. `fitted_slope_residual_se` is an additional ordinary regression diagnostic that ignores cross-grid dependence; it is **not** guaranteed to be a lower bound.

The optional GBM positivity experiment was **not** performed. The old unused `track_negative` hook was removed because it counted negative time steps rather than distinct paths. Under the benchmark parameters, the EM one-step negative-factor probability at the coarsest grid $h=1/2$ is approximately $2.12\times10^{-13}$, so even 2,000,000 two-step paths have fewer than $10^{-6}$ expected negative-factor events. Milstein's update factor is non-negative for every $h>0$ here because $\sigma^2-2\mu=-0.06<0$. Zero observed events would therefore not measure a useful rate. A separate high-volatility parameter set would be needed for a meaningful empirical positivity study.

## Generated data and figures

For a submission bundle, include the code, tests, generated results and figures, this README, and whichever supplied sources the course requires. Temporary PDF text extracts and working notes should be excluded; they have not been deleted from this working directory.

| CSV under `results/` | Purpose |
|---|---|
| `gbm_strong.csv` | Exact-path strong errors, intervals and fitted points |
| `gbm_weak.csv` | Signed paired mean biases, analytic biases, intervals and fitted points |
| `gbm_distribution.csv` | Numerical quantiles and density-normalised histogram bins with uncertainty |
| `gbm_moments.csv` | Numerical and exact mean/variance with numerical standard errors |
| `nonaffine_strong.csv` | Coupled price differences against both numerical references |
| `nonaffine_weak.csv` | Signed paired call-payoff differences against both references |
| `nonaffine_increment_diagnostics.csv` | Increment means, variances, correlations, targets and uncertainty |
| `nonaffine_path_validation.csv` | Finite-state and positive-price counts on every grid |
| `nonaffine_reference_comparison.csv` | Direct comparison of the 4,096- and 8,192-step references |

The five PDFs under `figures/` are `gbm_strong_convergence.pdf`, `gbm_weak_convergence.pdf`, `gbm_distribution.pdf`, `nonaffine_strong_convergence.pdf` and `nonaffine_weak_convergence.pdf`. Plots read the saved CSVs. Filled convergence markers enter a fit; open markers do not. Weak plots also show signed estimates with 95% intervals, so a visually declining absolute estimate cannot hide a confidence interval crossing zero.

## Verified full-run findings

The figures and numbers in this section are regenerated from the full run, not the smoke test.

With 2,000,000 GBM paths, the predeclared fine-grid strong fits give slopes 0.4991 $\pm$ 0.00036 (EM) and 0.9979 $\pm$ 0.00016 (Milstein). The signed weak-bias fits give 1.0100 $\pm$ 0.04405 for EM on six resolved grids ($N=2$ through $64$) and 0.9981 $\pm$ 0.00043 for Milstein on all nine grids. Every $\pm$ value in this paragraph is one paired-batch jackknife standard error, not a 95% interval. The finer EM points $N=128,256,512$ remain below the chosen Monte Carlo signal threshold and were not fitted. All 18 simulated signed-bias intervals contain the corresponding analytic bias. At $N=256$, both numerical means and variances, and all ten numerical quantiles, have approximate 95% intervals containing their exact GBM counterparts. These individual intervals are not a joint 95% guarantee.

The weak EM slope standard error is much larger than Milstein's because its finest fitted point, $N=64$, has an absolute estimated bias only about seven times its own standard error. On the log scale the derivative $1/|\hat\delta|$ amplifies noise at that small-bias point and propagates it into the slope. This explains the larger Monte Carlo uncertainty and does not change the 18/18 analytic-bias interval checks.

The EM weak fit includes the coarse step $h=1/2$. Fitting the **analytic** bias on the same six grids gives slope 0.9956 (or 0.9982 using only $N=8,16,32,64$), reflecting the $O(h^2)$ curvature of a first-order bias. The simulated 1.0100 is not a rate better than the theory; its difference from the analytic fit must be read with the Monte Carlo slope uncertainty.

Against the 4,096- and 8,192-step non-affine references, the empirical strong refinement slopes are 0.5205 $\pm$ 0.00187 and 0.5090 $\pm$ 0.00184 (one paired-batch jackknife standard error each). All ten signed call-payoff difference intervals include zero, so no weak-payoff slope is claimed. All 50,000 terminal states are finite and positive at each of the seven simulated grids. The mean absolute difference between the two numerical references is 0.04022, while the $N=512$ discrepancy against the 8,192-step reference is 0.15495; their ratio is 25.96%. By the triangle inequality, 0.04022 bounds how much the **reported mean absolute discrepancy** can change when switching between these two references. The observed change at $N=512$ is only 0.00552, and the observed fitted-slope change is 0.01153. Neither number is a bound on error relative to an unknown exact solution or on the true convergence order.
