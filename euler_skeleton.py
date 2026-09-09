"""
euler_skeleton.py - Explicit Euler scaffolding for the ODE IVP project.

This is a FILL-IN-THE-BLANK template. The structure, plotting, and test harness
are provided. Your job is to complete the four marked sections:

    [TASK 1]  implement the vectorized RHS of the logistic equation
    [TASK 2]  implement the explicit Euler step
    [TASK 3]  implement the convergence study (error vs h)
    [TASK 4]  implement the stability-threshold experiment

Run with:
    python euler_skeleton.py

When all four tasks are complete, the script prints PASS/FAIL diagnostics and
produces two figures: `euler_logistic.png` and `euler_convergence.png`.

Author: <your names>
Course: Numerical Analysis of ODEs and PDEs
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Model: logistic growth  y' = r y (1 - y/K)
#
# Note the RHS is written for a VECTOR y of shape (n_states,) - for a scalar
# problem n_states = 1. Keep the vector signature: it costs nothing and makes
# the scaffold work for the coupled systems in the problem pack.
# ---------------------------------------------------------------------------

def logistic_rhs(t, y):
    """Right-hand side of the logistic equation.

    Parameters
    ----------
    t : float
        Current time (unused by an autonomous ODE, kept for API compatibility).
    y : ndarray, shape (1,)
        Current state.

    Returns
    -------
    f : ndarray, shape (1,)
        Derivative dy/dt at (t, y).
    """
    r = 1.0    # growth rate
    K = 10.0   # carrying capacity
    # [TASK 1]  return r * y * (1 - y / K)  as a 1-D numpy array of shape (1,)
    # (hint: y is already a length-1 array; multiply element-wise)
    raise NotImplementedError("TODO: implement logistic_rhs")


# ---------------------------------------------------------------------------
# Explicit (forward) Euler:  y_{n+1} = y_n + h f(t_n, y_n)
# ---------------------------------------------------------------------------

def euler_step(f, t, y, h):
    """One explicit Euler step.

    Parameters
    ----------
    f : callable
        RHS function f(t, y) -> ndarray.
    t : float
        Current time.
    y : ndarray
        Current state.
    h : float
        Step size.

    Returns
    -------
    y_next : ndarray
        Approximate state at t + h.
    """
    # [TASK 2]  one line:  y_next = y + h * f(t, y)
    raise NotImplementedError("TODO: implement euler_step")


def solve_euler(f, y0, t0, t1, h):
    """Integrate f from t0 to t1 with fixed step h using explicit Euler.

    Returns (t, Y): the time grid and the solution trajectory. The requested
    h is a nominal maximum step; the final step is shortened so t[-1] == t1.
    For an order study, prefer h=(t1-t0)/N with integer N, so every run uses a
    genuinely uniform grid and the x-axis is unambiguous.
    """
    n_steps = int(np.ceil((t1 - t0) / h))
    t = np.empty(n_steps + 1)
    t[0] = t0
    y = np.zeros((n_steps + 1,) + np.shape(y0))
    y[0] = y0
    for n in range(n_steps):
        step = min(h, t1 - t[n])
        t[n + 1] = t[n] + step
        y[n + 1] = euler_step(f, t[n], y[n], step)
    return t, y


# ---------------------------------------------------------------------------
# Reference solution: the logistic equation has a closed-form solution
# ---------------------------------------------------------------------------

def logistic_exact(t, y0, r=1.0, K=10.0):
    """Closed-form solution of y' = r y (1 - y/K)."""
    return K / (1 + (K / y0 - 1) * np.exp(-r * t))


# ---------------------------------------------------------------------------
# Convergence study: global error at fixed T vs step size h
# ---------------------------------------------------------------------------

def convergence_study(f, y0, t0, t1, h_values, exact, plot_path=None):
    """Estimate the observed order of convergence.

    Choose h values that divide t1-t0 exactly. For each h, integrate to t1 and
    measure the L2 norm of the full-state error
    ||y_approx(t1) - y_exact(t1)||_2, so the same code works for scalar
    problems and systems. Then fit log(error) ~ order * log(h).

    `exact` is a function exact(t) -> float for a scalar state, or an ndarray
    of the same shape as the state for a system.

    Returns (errors, order).
    """
    h_values = np.asarray(h_values, dtype=float)
    duration = t1 - t0
    ratios = duration / h_values
    if not np.allclose(ratios, np.round(ratios), rtol=0.0, atol=1e-12):
        raise ValueError("For an order study, choose h=(t1-t0)/N with integer N")
    errors = []
    for h in h_values:
        t, y = solve_euler(f, y0, t0, t1, h)
        # [TASK 3a]  compute the full-state error at the final time
        #   err = np.linalg.norm(y[-1] - exact(t[-1]))
        # The L2 norm covers all components, so the same line works for a
        # scalar state (shape (1,)) and for a system (shape (d,)).
        err = 0.0  # TODO
        errors.append(err)

    errors = np.asarray(errors)
    # [TASK 3b]  fit a straight line in log-log space using np.polyfit
    #   slope, intercept = np.polyfit(np.log(h_values), np.log(errors), 1)
    # and use the slope as the observed order.
    order = 0.0  # TODO

    if plot_path is not None:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.loglog(h_values, errors, "o-", label="explicit Euler")

        # theoretical slope: overlay the order-1 reference line
        # (visualization guide, Rule 1 --- no benchmark, no claim)
        h_lin = np.logspace(np.log10(h_values.min()), np.log10(h_values.max()), 50)
        c_ref = errors[0] / h_values[0]
        ax.loglog(h_lin, c_ref * h_lin**1.0, "k--", label="order-1 reference")

        # confidence band on the fitted slope
        # (visualization guide, Rule 1 --- polyfit with cov=True)
        if order > 0:
            coef, cov = np.polyfit(np.log(h_values), np.log(errors), 1, cov=True)
            slope_err = np.sqrt(cov[0, 0])
            lx = np.linspace(np.log(h_values.min()), np.log(h_values.max()), 50)
            line = coef[0] * lx + coef[1]
            se = np.sqrt(cov[1, 1] + 2 * lx * cov[0, 1] + lx**2 * cov[0, 0])
            ax.loglog(np.exp(lx), np.exp(line), "r--",
                      label=f"fit {order:.2f} \u00b1 {slope_err:.2f}")
            ax.fill_between(np.exp(lx), np.exp(line - 1.96 * se),
                            np.exp(line + 1.96 * se),
                            color="r", alpha=0.15, label="95% band")

        # annotate the round-off floor: error stops falling at small h
        floor_idx = int(np.argmin(errors))
        if floor_idx < len(h_values) - 1:
            ax.annotate("round-off floor",
                        xy=(h_values[floor_idx], errors[floor_idx]),
                        xytext=(h_values[floor_idx] * 4, errors[floor_idx] * 4),
                        arrowprops=dict(arrowstyle="->"))

        ax.set_xlabel("step size h")
        ax.set_ylabel("global error at T")
        ax.set_title(f"observed order about {order:.2f} (theory: 1.00)")
        ax.grid(True, which="both", ls=":", alpha=0.6)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)

    return errors, order


# ---------------------------------------------------------------------------
# Stability experiment: linear model  y' = lambda y,  |1 + h lambda| <= 1
# ---------------------------------------------------------------------------

def stability_threshold():
    """Find the largest h for which explicit Euler stays bounded on y' = -y.

    Theory: stability requires |1 - h| <= 1  ->  h <= 2. Above h = 2 the
    solution should oscillate and grow. Returns (h_inside, h_outside).

    NB: just above the boundary the growth per step is only ~1.1x, so a long
    horizon is needed for the growth to become visible. T = 200 gives
    ~95 steps at h = 2.1 -> 1.1^95 ~ 10^4, clearly unstable.
    """
    lam = -1.0
    y0 = np.array([1.0])
    t0, t1 = 0.0, 200.0
    f_linear = lambda t, y: lam * y     # y' = -y

    def test_stable(h):
        t, y = solve_euler(f_linear, y0, t0, t1, h)
        # [TASK 4]  return True if |y[-1]| stays bounded (e.g. <= 10), False otherwise.
        raise NotImplementedError("TODO: implement test_stable")

    # the probes are given: just inside (h = 1.9) and just outside (h = 2.1)
    # the stability boundary  |1 - h| <= 1   (i.e. h <= 2)
    h_inside = 1.9
    h_outside = 2.1
    assert test_stable(h_inside), f"expected stability at h={h_inside}"
    assert not test_stable(h_outside), f"expected instability at h={h_outside}"
    return h_inside, h_outside


# ---------------------------------------------------------------------------
# Main: run everything and print PASS/FAIL
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("euler_skeleton.py - explicit Euler scaffolding")
    print("=" * 60)

    # figures are written next to this script (not the CWD), so the output
    # works regardless of where the interpreter is launched from
    output_dir = Path(__file__).resolve().parent

    y0 = np.array([0.5])
    t0, t1 = 0.0, 10.0
    h = 0.05

    # 1. sanity run
    t, y = solve_euler(logistic_rhs, y0, t0, t1, h)
    final = y[-1, 0]
    exact_final = logistic_exact(t[-1], y0[0])
    ok1 = abs(final - exact_final) < 0.2
    print(f"[1] logistic t=10:  euler={final:.4f}  exact={exact_final:.4f}  "
          f"|err|={abs(final - exact_final):.4f}  -> {'PASS' if ok1 else 'FAIL'}")
    # note: order-1 method with h=0.05 over t=10 gives error ~ 0.05*scale, so 0.2 is a loose bound

    # 2. convergence
    h_values = [0.4, 0.2, 0.1, 0.05, 0.025]
    errors, order = convergence_study(
        logistic_rhs, y0, t0, t1, h_values,
        lambda tt: logistic_exact(tt, y0[0]),   # scalar-valued exact solution
        plot_path=output_dir / "euler_convergence.png",
    )
    ok2 = 0.7 <= order <= 1.3
    print(f"[2] observed order = {order:.2f}  (theory 1.00) -> {'PASS' if ok2 else 'FAIL'}")

    # your own error table --- these are YOUR numbers from TASK 3a; the
    # instructor compares them against the reference (no expected values here)
    print("  Your error table (global error at t=10, each h):")
    print(f"  {'h':>8} {'error':>14}")
    for hh, e in zip(h_values, errors):
        print(f"  {hh:>8.4f} {e:>14.6e}")

    # 3. stability threshold
    h_in, h_out = stability_threshold()
    ok3 = True
    print(f"[3] stable at h={h_in}, unstable at h={h_out} (theory: boundary h=2) -> "
          f"{'PASS' if ok3 else 'FAIL'}")

    # 4. trajectory figure
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t, y[:, 0], label="explicit Euler (h=0.05)")
    ax.plot(t, logistic_exact(t, y0[0]), "k--", label="exact")
    ax.set_xlabel("t"); ax.set_ylabel("y")
    ax.set_title("Logistic growth")
    ax.legend(); ax.grid(True, ls=":", alpha=0.6)
    fig.tight_layout(); fig.savefig(output_dir / "euler_logistic.png", dpi=150); plt.close(fig)

    print("=" * 60)
    all_ok = ok1 and ok2 and ok3
    print("ALL TESTS PASS" if all_ok else "SOME TESTS FAIL - see above")
    print("=" * 60)


if __name__ == "__main__":
    main()
