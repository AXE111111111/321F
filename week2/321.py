"""
implicit_euler_skeleton.py - Implicit Euler with Newton's iteration.

Scaffolding for Week 2. The structure, plotting, and test harness are provided.
Your job is to complete the marked sections:

    [TASK 1]  implement the RHS of the Robertson problem (stiff benchmark)
    [TASK 2]  implement the Jacobian (analytic) of the Robertson RHS
    [TASK 3]  implement one Newton step for the implicit Euler equation
    [TASK 4]  build your own scipy oracle and cross-check y(40) component-wise

Provided (no code to write, but read the comments):
    - the solver harness (implicit Euler via Newton, mass check, figures)
    - the damping switch (0 < damping <= 1) wired into TASK 3's Newton step;
      Tutorial 2's individual challenge lives in rc_diode_challenge.py
      (full vs adaptive-damped Newton on an RC-diode circuit)
    - the explicit-Euler blow-up demo and the frozen-Jacobian estimate
    - the deep-work hook (Tutorial 2): implement the NEW solver
      adaptive_implicit_euler --- a step-doubling controller (one step of
      size h (y_coarse) vs two of h/2 (y_fine));
      ||y_fine - y_coarse|| is the local-error estimate that drives h
      (halve if e > tol, double if e < tol/10).  Until it is implemented,
      main()'s [4] demo prints a "skipped" notice and moves on, so the
      Lecture-2 lab run (which uses this same file) does not crash

Run with:
    python implicit_euler_skeleton.py

Outputs: `robertson_implicit.png` (solution) and, once the deep-work
solver is implemented, `robertson_adaptive_h.png` (accepted step vs t).

Author: <李昆泽>
Course: Numerical Analysis of ODEs and PDEs
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# ---------------------------------------------------------------------------
# Model: Robertson chemical kinetics (stiff: largest Jacobian eigenvalue
# ~ 3.4e3, NOT the rate constant 3e7, because y2 stays ~1e-5; stiffness ratio
# ~ 5e3 during the transient, growing to ~1.6e5 at t=40; mass y1+y2+y3 = 1 is
# a linear invariant)
#
#   y1' = -0.04 y1 + 1e4 y2 y3
#   y2' =  0.04 y1 - 1e4 y2 y3 - 3e7 y2^2
#   y3' =  3e7 y2^2
#   y(0) = [1, 0, 0]
#
# Verify against your own oracle: scipy.integrate.solve_ivp with a very tight
# tolerance (method='BDF' or 'Radau', rtol=1e-12). There is no handed-in
# reference value in this course - see project brief Section 2, "How to verify without a given answer".
# Mass conservation: y1 + y2 + y3 = 1 (linear invariant)
# ---------------------------------------------------------------------------

def robertson_rhs(t, y):
    """Robertson RHS. y has shape (3,)."""
    y1, y2, y3 = y
    # [TASK 1]  return the 3-component derivative as a 1-D array
    dydt = np.array([
        -0.04 * y1 + 1e4 * y2 * y3,
         0.04 * y1 - 1e4 * y2 * y3 - 3e7 * y2 * y2,
         3e7 * y2 * y2
    ])
    return dydt


def robertson_jacobian(t, y):
    """Analytic 3x3 Jacobian of the Robertson RHS.

    J[i, j] = d f_i / d y_j
    """
    y1, y2, y3 = y
    # [TASK 2]  build the 3x3 matrix J.
    #
    #   df1/dy1 = -0.04        df1/dy2 = 1e4*y3      df1/dy3 = 1e4*y2
    #   df2/dy1 =  0.04        df2/dy2 = -1e4*y3 - 6e7*y2   df2/dy3 = -1e4*y2
    #   df3/dy1 =  0           df3/dy2 = 6e7*y2      df3/dy3 = 0
    J = np.array([
        [-0.04,  1e4 * y3,               1e4 * y2],
        [ 0.04, -1e4 * y3 - 6e7 * y2,   -1e4 * y2],
        [ 0.0,   6e7 * y2,               0.0     ]
    ])
    return J


# ---------------------------------------------------------------------------
# Newton's iteration for the implicit Euler step
#
# Implicit Euler:  y_{n+1} = y_n + h f(t_{n+1}, y_{n+1})
# Rearranged:      F(w) = w - y_n - h f(t_{n+1}, w) = 0
# Newton:          w <- w - J_F(w)^{-1} F(w),  J_F(w) = I - h J_f(w)
# ---------------------------------------------------------------------------

# trace=True prints the residual ||F||_inf per iteration (lightning-talk
# diagnostic); line_search=True adds the backtracking (adaptive) damping
# you implement as [TASK B] of the Tutorial-2 individual challenge
# (rc_diode_challenge.py), replayed here on Robertson (implemented in the
# reference); the scaffold itself leaves both False.
def newtonsolve_implicit_euler(f, jac, t_next, y_n, h, tol=1e-12, max_iter=30,
                              damping=1.0, trace=False, line_search=False):
    """Solve the implicit Euler equation for y_{n+1}.

    Returns (y_next, converged).
    """
    w = y_n.copy()          # initial guess: value from previous time step
    for _ in range(max_iter):
        # residual F(w) = w - y_n - h f(t_next, w)
        F = w - y_n - h * f(t_next, w)
        residual_norm = np.linalg.norm(F, np.inf)
        # trace=True (passed from main()) prints the residual here - the
        # lightning-talk demo on the first step needs no extra code
        if trace:
            print(f"        ||F||_inf = {residual_norm:.3e}")
        if residual_norm < tol:
            return w, True
            
        # [TASK 3]  Newton step:
        I = np.eye(len(w))
        JF = I - h * jac(t_next, w)
        delta = np.linalg.solve(JF, F)
        w = w - damping * delta

    return w, False


def implicit_euler_step(f, jac, t, y, h):
    """One implicit Euler step using Newton's iteration."""
    y_next, converged = newtonsolve_implicit_euler(f, jac, t + h, y, h)
    if not converged:
        raise RuntimeError(f"Newton failed to converge at t={t:.3f}")
    return y_next


def solve_implicit_euler(f, jac, y0, t0, t1, h):
    """Integrate with nominal step h, shortened at the end to land on t1."""
    n_steps = int(np.ceil((t1 - t0) / h))
    t = np.empty(n_steps + 1)
    t[0] = t0
    y = np.zeros((n_steps + 1,) + np.shape(y0))
    y[0] = y0
    for n in range(n_steps):
        step = min(h, t1 - t[n])
        t[n + 1] = t[n] + step
        y[n + 1] = implicit_euler_step(f, jac, t[n], y[n], step)
    return t, y


# ---------------------------------------------------------------------------
# Deep-work (Tutorial 2): adaptive step-size controller (step doubling)
#
# A NEW solver, parallel to solve_implicit_euler above.  The step h is
# driven by the method's own local-error signal instead of staying fixed:
#     y_coarse = one  implicit Euler step of size h
#     y_fine   = two  implicit Euler steps of size h/2
#     e = ||y_fine - y_coarse||   (an O(h^2) local-error estimate)
# Controller: accept if e <= tol; halve h if e > tol; double h (up to h0)
# if e < tol/10.  Accept y_fine (the more accurate of the two candidates).
# ---------------------------------------------------------------------------

def adaptive_implicit_euler(f, jac, y0, t0, t1, h0, tol):
    """Step-doubling adaptive implicit Euler.

    Returns (t, y, h_used): the accepted grid, the solution on it, and the
    step actually taken at each accepted step (len(h_used) + 1 == len(t)).
    """
    t = [t0]
    y = [np.array(y0, dtype=float)]
    h_used = []
    h = h0
    
    # Deep-work controller loop implementation
    while t[-1] < t1:
        h = min(h, t1 - t[-1])          # never overshoot the end
        y_coarse = implicit_euler_step(f, jac, t[-1], y[-1], h)
        y_half   = implicit_euler_step(f, jac, t[-1], y[-1], h/2)
        y_fine   = implicit_euler_step(f, jac, t[-1] + h/2, y_half, h/2)
        
        e = np.linalg.norm(y_fine - y_coarse, np.inf)
        
        if e <= tol:
            # Accept step
            t.append(t[-1] + h)
            y.append(y_fine)
            h_used.append(h)
            
            # Double h if error is very small (up to h0)
            if e < tol / 10:
                h = min(2 * h, h0)
        else:
            # Reject step and halve h
            h = h / 2
            
    return np.array(t), np.array(y), np.array(h_used)


# ---------------------------------------------------------------------------
# Explicit Euler (imported from the Week-1 scaffold or re-implemented here)
# ---------------------------------------------------------------------------

def euler_step(f, t, y, h):
    return y + h * f(t, y)


def solve_euler(f, y0, t0, t1, h):
    """Use nominal step h; shorten the final step to land exactly at t1."""
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
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("implicit_euler_skeleton.py - Newton for stiff problems")
    print("=" * 60)

    y0 = np.array([1.0, 0.0, 0.0])
    t0, t1 = 0.0, 40.0
    h = 0.1

    # 1. cross-check against YOUR OWN tight-tolerance oracle.
    #    [TASK 4]  Build it with scipy.integrate.solve_ivp and compare ALL
    #    three components at t=40:
    #      from scipy.integrate import solve_ivp
    #      sol = solve_ivp(robertson_rhs, [t0, t1], y0,
    #                      method='BDF', rtol=1e-12, atol=1e-14)
    #      y_oracle = sol.y[:, -1]        # tight reference, shape (3,)
    #    There is no handed-out value here - learning to construct and trust a
    #    reference is the exercise. (Sanity: every relative error < 1%.)
    # 1b. lightning-talk demo: Newton residual on the FIRST step (t = 0.1).
    #     trace=True makes newtonsolve_implicit_euler print ||F||_inf per
    #     Newton iteration.  Expect it to RISE on the first full step (the
    #     initial guess y2 = 0 is a degenerate point where the 3e7*y2^2 term
    #     is dormant) and then decay quadratically - show this live.
    newtonsolve_implicit_euler(robertson_rhs, robertson_jacobian, t0 + h, y0, h,
                               trace=True)
    t, y = solve_implicit_euler(robertson_rhs, robertson_jacobian, y0, t0, t1, h)
    y_end = y[-1]
    
    # [TASK 4] your scipy oracle's y(40), shape (3,)
    sol = solve_ivp(robertson_rhs, [t0, t1], y0, 
                    method='BDF', rtol=1e-12, atol=1e-14)
    y_oracle = sol.y[:, -1]
    
    if y_oracle is None:
        print(f"[1] implicit Euler h={h}:  y(40) = {y_end}  (build your scipy "
              f"oracle in [TASK 4] to see the comparison)")
    else:
        print(f"[1] implicit Euler h={h}:  y(40) vs oracle (rel. err per component):")
        for i, name in enumerate(["y1", "y2", "y3"]):
            rel = abs(y_end[i] - y_oracle[i]) / abs(y_oracle[i])
            print(f"    {name}(40) = {y_end[i]:.6e}  vs oracle {y_oracle[i]:.6e}  "
                  f"(rel. err = {rel:.3%})")

    # 2. mass conservation
    mass = y.sum(axis=1)
    ok2 = np.max(np.abs(mass - 1.0)) < 1e-10
    print(f"[2] mass conservation: max|sum-1| = {np.max(np.abs(mass-1.0)):.2e} "
          f"-> {'PASS' if ok2 else 'FAIL'}")
    # For the lightning talk: plot (mass - 1) vs. t to show the drift curve,
    # e.g. ax.plot(t, mass - 1.0) on a log scale.  A clean line at ~0 means
    # the RHS/Newton solve preserves the linear invariant.

    # 3. explicit Euler is impractical: it blows up even with thousands of steps
    #    Theory: the largest Jacobian eigenvalue is ~3.4e3 (NOT the rate
    #    constant 3e7, since y2 ~ 1e-5), so Forward Euler stability needs
    #    h < 2/3.4e3 ~ 5.9e-4, i.e. ~7e4 steps for T=40. Running h=0.01 (only
    #    4000 steps) already fails.
    #    (The overflow warnings from the blow-up are EXPECTED - that is the point.)
    t_full = 40.0
    with np.errstate(over="ignore", invalid="ignore"):
        tE, yE = solve_euler(robertson_rhs, y0, 0.0, t_full, 0.01)
    finite = np.all(np.isfinite(yE))
    h_max = 2.0 / 3393.0
    n_steps_theory = int(np.ceil(t_full / h_max))
    print(f"[3] explicit Euler h=0.01 over T=40 ({len(tE)} steps): "
          f"{'finite' if finite else 'blow-up (inf/nan)'}")
    print(f"    frozen-Jacobian estimate near the most restrictive sampled state: "
          f"h < {h_max:.2e} "
          f"-> ~{n_steps_theory:,} steps for T=40 (impractical)")

    # 4. deep-work demo (Tutorial 2): fixed vs adaptive step size.
    #    The controller should shrink h through the stiff transient and
    #    grow back toward h0 once the solution relaxes.  tol=2e-6 sits
    #    inside the local-error range at h=0.1 (e ~ 3.6e-5 at t=0 decaying
    #    to ~1.4e-7 at t=40), so the transient misses the budget (halve)
    #    and the relaxed tail beats it comfortably (double).  Try other
    #    tolerances and watch what happens to the h(t) curve.
    #    Lecture-2 note: adaptive_implicit_euler is still a stub until the
    #    Tutorial-2 deep work, so this demo is guarded below -- it prints a
    #    notice and moves on instead of crashing the Week-2 lab run.
    #    Re-run main() after the deep work to get the comparison and the
    #    h(t) figure.
    tol = 2e-6
    try:
        tA, yA, hA = adaptive_implicit_euler(robertson_rhs, robertson_jacobian,
                                             y0, t0, t1, h0=h, tol=tol)
    except NotImplementedError:
        print("[4] deep-work demo skipped: adaptive_implicit_euler is not "
              "implemented yet (Tutorial-2 deep work).")
        tA = yA = hA = None
    if tA is not None:
        print(f"[4] deep-work demo: fixed h={h} vs adaptive (start h0={h}, "
              f"tol={tol:g}):")
        print(f"    accepted steps:  fixed {len(t) - 1}   vs adaptive {len(hA)}")
        print(f"    smallest h reached: {hA.min():.3e}   (final h: {hA[-1]:.3e})")
        if y_oracle is not None:
            print(f"    final error vs oracle (inf-norm):")
            print(f"      fixed    y(40) = {y_end}   e_inf = "
                  f"{np.max(np.abs(y_end - y_oracle)):.3e}")
            print(f"      adaptive y(40) = {yA[-1]}   e_inf = "
                  f"{np.max(np.abs(yA[-1] - y_oracle)):.3e}")
        else:
            print("    (build your scipy oracle in [TASK 4] to get the error comparison)")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(tA[:-1], hA, marker=".", ls="-")
        ax.axhline(h, color="k", ls=":", lw=1, label=f"fixed h = {h}")
        ax.set_xlabel("t"); ax.set_ylabel("accepted step h")
        ax.set_title("Robertson problem - adaptive implicit Euler (step doubling)")
        ax.set_yscale("log"); ax.legend(); ax.grid(True, ls=":", alpha=0.6)
        fig.tight_layout(); fig.savefig("robertson_adaptive_h.png", dpi=150); plt.close(fig)

    # 5. figures
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t, y[:, 0], label="y1 (A)")
    ax.plot(t, y[:, 1] * 1e4, label="y2 (B) x 1e4")
    ax.plot(t, y[:, 2], label="y3 (C)")
    ax.set_xlabel("t"); ax.set_ylabel("concentration")
    ax.set_title("Robertson problem - implicit Euler, h=0.1")
    ax.legend(); ax.grid(True, ls=":", alpha=0.6)
    ax.set_xscale("log")
    fig.tight_layout(); fig.savefig("robertson_implicit.png", dpi=150); plt.close(fig)

    print("=" * 60)
    print("ALL TESTS PASS" if ok2 else "SOME TESTS FAIL - see above")
    print("=" * 60)


if __name__ == "__main__":
    main()