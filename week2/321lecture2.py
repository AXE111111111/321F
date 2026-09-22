"""
rc_diode_challenge.py - Full vs adaptive-damped Newton for one implicit
Euler step in a two-node RC circuit with a diode.

Individual challenge, Tutorial 2.  The physics is given; your job is the
Newton iteration and the experiment.

Circuit (SI units): R1 = R2 = 100 ohm, C1 = C2 = 10 uF, Vs = 5 V,
diode  I_D(v2) = Is (exp(v2/VT) - 1),  Is = 1e-12 A,  VT = 25.85 mV.
The physical time scale is RC = 1 ms, so h = 1 ms is a natural step, not
an artificially huge one.

The model (Kirchhoff's current law) is the ODE system
    C1 v1' = (Vs - v1)/R1 - (v1 - v2)/R2          v1(0) = 0
    C2 v2' = (v1 - v2)/R2 - Is*(exp(v2/VT) - 1)   v2(0) = 0
The stiff unknown is the diode exponential in the v2 equation.

One implicit Euler step from v_n to v solves F(v) = 0 with
    F1 = C1 (v1-v1n)/h - (Vs-v1)/R1 + (v1-v2)/R2
    F2 = C2 (v2-v2n)/h - (v1-v2)/R2 + Is*(exp(v2/VT) - 1)
and J = dF/dv is given below.

Why this problem?  At the FIRST step from v_n = (0,0), h = 1 ms:
  * full Newton jumps to v2 ~ 1 V (the diode current there is ~6e4 A -- a
    linearization artifact), then retreats by one thermal voltage
    VT ~ 26 mV per iteration: 21 iterations to reach tol = 1e-12.
  * backtracking damping rejects the overshooting step and converges in 6.
The same protection matters later, every time v2 crosses the diode turn-on
region (v2 ~ 0.4-0.6 V) with a step that is too large.

Your tasks:
    [TASK A]  one FULL Newton step:        w <- w + delta
    [TASK B]  ADAPTIVE damping: backtracking on the SAME delta -- halve
              the step length until ||F|| strictly decreases; rejected
              trial steps do NOT count as iterations
    [TASK C]  the experiment: for each h on the doubling grid, integrate
              over [0, 20 ms] and record the MAX number of Newton
              iterations over ALL time steps; find the largest h for which
              every Newton solve converges in <= 10 iterations, for full
              and for adaptive-damped Newton; print the table.

Deliverable (report + Individual-Challenge DB):
    the before/after pair  h_max(full) -> h_max(damped)  and the
    iteration table that justifies it.

The h = 1 ms first-step reference values are given on the Tutorial 2 slide
(full Newton: spike then ~V_T retreat, 21 iterations; adaptive damped:
monotone, 6 iterations) -- your traces must reproduce them.  F and J below
follow from the model.

Run with:
    python rc_diode_challenge.py

Author: <李昆泽>
Course: Numerical Analysis of ODEs and PDEs
"""

import numpy as np

# ---------------------------------------------------------------------------
# Circuit parameters and the implicit-Euler equations (given)
# ---------------------------------------------------------------------------

R1 = R2 = 100.0
C1 = C2 = 10e-6
Vs = 5.0
Is = 1e-12
VT = 0.02585
T = 20e-3          # simulation horizon: 20 RC time constants (RC = 1 ms)
GRID_MS = [0.1, 0.2, 0.4, 0.8, 1.0, 2.0, 4.0, 8.0, 20.0]  # doubling grid, ms
MAX_NEWTON_ITER = 10   # the challenge criterion


def F(v, v_n, h):
    """Residual of the implicit-Euler equations, shape (2,).  F(v) = 0 at
    the step solution."""
    v1, v2 = v
    v1n, v2n = v_n
    return np.array([
        C1*(v1 - v1n)/h - (Vs - v1)/R1 + (v1 - v2)/R2,
        C2*(v2 - v2n)/h - (v1 - v2)/R2 + Is*(np.exp(v2/VT) - 1),
    ])


def J(v, h):
    """2x2 Jacobian dF/dv."""
    v1, v2 = v
    return np.array([
        [C1/h + 1/R1 + 1/R2,          -1/R2],
        [-1/R2,       C2/h + 1/R2 + (Is/VT)*np.exp(v2/VT)],
    ])


# ---------------------------------------------------------------------------
# Newton's iteration for one implicit Euler step
# ---------------------------------------------------------------------------

def newton_step(v_n, h, mode, tol=1e-12, max_iter=300):
    """Solve F(v) = 0 for one implicit-Euler step, starting from the warm
    start v_n (the previous time step's value).

    mode='full'   -> [TASK A] full Newton
    mode='damped' -> [TASK B] adaptive damping (backtracking line search)

    Returns (v_next, n_iter, converged).  n_iter counts ACCEPTED Newton
    steps only; backtracking trial steps do not count.
    """
    w = v_n.copy()
    n_iter = 0
    for _ in range(max_iter):
        f = F(w, v_n, h)
        r = np.linalg.norm(f)
        if r < tol:
            return w, n_iter, True
        delta = np.linalg.solve(J(w, h), -f)   # Newton direction (both modes)
        
        if mode == "full":
            # [TASK A] full Newton step
            w = w + delta
            n_iter += 1
        else:
            # [TASK B] adaptive damping (backtracking on the SAME delta)
            lam = 1.0
            while True:
                w_try = w + lam * delta
                f_try = F(w_try, v_n, h)
                r_try = np.linalg.norm(f_try)
                if r_try < r:
                    break  # strict decrease, accept the step
                lam /= 2.0
                if lam < 1e-10: # safeguard to prevent infinite loops
                    break 
            w = w_try
            n_iter += 1
            
    return w, n_iter, False


# ---------------------------------------------------------------------------
# Simulation harness (given)
# ---------------------------------------------------------------------------

def simulate(h, mode):
    """Integrate over [0, T] with nominal step h.  Returns
    (v_end, max_iter_over_all_steps, all_converged)."""
    n_steps = max(1, int(round(T / h)))
    v = np.zeros(2)
    max_iter = 0
    for _ in range(n_steps):
        v, n_iter, ok = newton_step(v, h, mode)
        max_iter = max(max_iter, n_iter)
        if not ok:
            return v, max_iter, False
    return v, max_iter, True


def main():
    print("=" * 68)
    print("rc_diode_challenge.py - full vs adaptive-damped Newton")
    print("criterion: every Newton solve over [0, 20 ms] converges")
    print(f"           in <= {MAX_NEWTON_ITER} iterations")
    print("=" * 68)

    # [TASK C] the experiment
    print(f"{'h[ms]':>7} | {'full maxit':>10} | {'damped maxit':>12} | {'full <=10?':>10} | {'damped <=10?':>12}")
    print("-" * 68)
    
    max_h_full = None
    max_h_damped = None

    for h_ms in GRID_MS:
        h = h_ms * 1e-3
        
        # Run simulations
        _, max_iter_full, _ = simulate(h, 'full')
        _, max_iter_damped, _ = simulate(h, 'damped')
        
        # Check criteria
        pass_full = max_iter_full <= MAX_NEWTON_ITER
        pass_damped = max_iter_damped <= MAX_NEWTON_ITER
        
        # Update max valid h
        if pass_full:
            max_h_full = h_ms
        if pass_damped:
            max_h_damped = h_ms
            
        print(f"{h_ms:>7.1f} | {max_iter_full:>10d} | {max_iter_damped:>12d} | "
              f"{str(pass_full):>10} | {str(pass_damped):>12}")

    print("-" * 68)
    print("largest h with every step <= 10 iterations:")
    print(f"  full   = {max_h_full} ms")
    print(f"  damped = {max_h_damped} ms")


if __name__ == "__main__":
    main()