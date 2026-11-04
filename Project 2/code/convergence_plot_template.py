"""Student template: convergence plots and observed-order fits (Week 6).

Reusable for every refinement study in this course: give it mesh sizes and
errors, get a labelled log--log plot, a fitted slope, and a warning line to
annotate when the finest mesh leaves the asymptotic regime.
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless: write PNG files, no window
import matplotlib.pyplot as plt


def fit_observed_order(hs, errs, fit_slice=None):
    """Least-squares slope of log(err) vs log(h).  Returns (slope, residuals).

    Report the slope AND the fit quality: a decorative straight line through
    non-asymptotic points proves nothing. ``fit_slice`` explicitly selects the
    asymptotic window (for example, slice(1, 5)); the default uses every point
    only for initial diagnosis and must not be the headline fit when known
    contaminated levels are present.
    """
    hs = np.asarray(hs, dtype=float)
    errs = np.asarray(errs, dtype=float)
    if len(hs) != len(errs) or len(hs) < 2:
        raise ValueError("need at least two (h, error) pairs")
    if not (np.all(np.isfinite(hs)) and np.all(np.isfinite(errs))):
        raise ValueError("hs and errs must be finite")
    if np.any(errs <= 0) or np.any(hs <= 0):
        raise ValueError("nonpositive errors: cannot take log")
    selected = np.arange(len(hs))[fit_slice] if fit_slice is not None else np.arange(len(hs))
    if selected.size < 2:
        raise ValueError("fit window must contain at least two levels")
    fit_hs, fit_errs = hs[selected], errs[selected]
    if np.unique(fit_hs).size < 2:
        # repeated h makes log(h) degenerate: the design matrix is singular
        # and polyfit silently hands back a plausible-looking slope (e.g.
        # [0.1, 0.1] with equal errors "fits" order 1 with zero deviation).
        raise ValueError("fit window must contain at least two DISTINCT mesh "
                         "sizes; log-log slopes need h to vary")
    coeffs, res, *_ = np.polyfit(np.log(fit_hs), np.log(fit_errs), 1, full=True)
    slope = coeffs[0]
    max_dev = np.max(np.abs(np.log(fit_errs) - np.polyval(coeffs, np.log(fit_hs))))
    return slope, max_dev


def plot_convergence(hs, errs, expected, title, outfile, annotation=None,
                     fit_slice=None):
    """Log--log plot with a reference line of the expected order.

    hs, errs: mesh sizes and errors; expected: theoretical order for the
    reference line; annotation: optional string explaining any deviation
    (round-off floor, boundary treatment, singularity).
    """
    hs = np.asarray(hs, dtype=float)
    errs = np.asarray(errs, dtype=float)
    slope, max_dev = fit_observed_order(hs, errs, fit_slice=fit_slice)

    fig, ax = plt.subplots(figsize=(5.2, 3.9))
    ax.loglog(hs, errs, "o-", label="measured error")
    ref = errs[0] * (hs / hs[0]) ** expected
    ax.loglog(hs, ref, "--", label=f"reference slope {expected}")
    ax.set_xlabel("mesh size $h$")
    ax.set_ylabel("error norm")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    window = "all levels" if fit_slice is None else f"selected window {fit_slice}"
    text = (f"fitted order = {slope:.2f} ({window}; "
            f"max log-deviation {max_dev:.2f})")
    if annotation:
        text += f"\n{annotation}"
    ax.text(0.04, 0.04, text, transform=ax.transAxes, fontsize=8,
            va="bottom", bbox=dict(facecolor="white", alpha=0.8))
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    return slope


def refinement_table(hs, errs, expected):
    """Print a refinement table with per-step observed orders."""
    print(f"{'h':>10} {'error':>12} {'observed order':>15}")
    print(f"{hs[0]:>10.5f} {errs[0]:>12.3e} {'---':>15}")
    for k in range(1, len(hs)):
        p = np.log(errs[k - 1] / errs[k]) / np.log(hs[k - 1] / hs[k])
        flag = "" if abs(p - expected) < 0.25 else "   <-- deviation"
        print(f"{hs[k]:>10.5f} {errs[k]:>12.3e} {p:>15.2f}{flag}")


if __name__ == "__main__":
    # Demonstration with synthetic O(h^2) data; the finest mesh is limited
    # by the algebraic-solver tolerance rather than discretisation.  Replace
    # with your own (hs, errors) once the solver runs.  Note how the printed
    # per-step order drops at the contaminated point --- that drop is data,
    # not noise: explain it in the report (tighten the tolerance and it
    # moves or disappears).
    hs = np.array([1 / 8, 1 / 16, 1 / 32, 1 / 64, 1 / 128, 1 / 256, 1 / 512])
    clean = 1e-1 * hs**2
    # Demo-only synthetic floor (not Project 2 measured data); plot_figures.py uses CSVs.
    errs = np.sqrt(clean**2 + (5e-7)**2)  # solver-tolerance floor
    refinement_table(hs, errs, expected=2.0)
    # The last two levels are deliberately contaminated, so select the clean
    # asymptotic window explicitly rather than fitting every point.
    fit_slice = slice(0, 5)
    slope = plot_convergence(hs, errs, expected=2.0,
                             title="demo: second order with a tolerance floor",
                             outfile="convergence_demo.png",
                             annotation="finest mesh limited by the "
                             "algebraic-solver tolerance",
                             fit_slice=fit_slice)
    all_slope, _ = fit_observed_order(hs, errs)
    print(f"selected-window order {fit_slice}: {slope:.2f}")
    print(f"all-level diagnostic fit (not the headline result): {all_slope:.2f}")
    print("wrote convergence_demo.png")
