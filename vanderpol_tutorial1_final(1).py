
"""
Van der Pol submission script for Tutorial 1.

This script:
1. Implements Explicit Euler.
2. Solves the Van der Pol oscillator with mu = 1.
3. Builds a high-accuracy scipy solve_ivp oracle using DOP853.
4. Computes the observed Euler convergence order for
   h = 0.1, 0.05, 0.025, 0.0125.
5. Prints oracle y1(10).
6. Saves the required phase-plane trajectory (y2 vs y1).
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def euler_step(f, t, y, h):
    """One explicit Euler step."""
    return y + h * f(t, y)


def solve_euler(f, y0, t0, t1, h):
    """Integrate an ODE using fixed-step Explicit Euler."""
    n_steps = int(round((t1 - t0) / h))
    t = np.linspace(t0, t1, n_steps + 1)
    y = np.empty((n_steps + 1,) + np.shape(y0), dtype=float)
    y[0] = y0

    for n in range(n_steps):
        y[n + 1] = euler_step(f, t[n], y[n], h)

    return t, y


def vanderpol_rhs(t, y, mu=1.0):
    """Van der Pol system: y1' = y2, y2' = mu(1-y1^2)y2-y1."""
    return np.array([
        y[1],
        mu * (1.0 - y[0] ** 2) * y[1] - y[0]
    ])


def build_oracle(y0, t0, t1):
    """High-accuracy scipy reference solution."""
    sol = solve_ivp(
        lambda t, y: vanderpol_rhs(t, y, mu=1.0),
        [t0, t1],
        y0,
        method="DOP853",
        rtol=1e-12,
        atol=1e-14,
        dense_output=True,
    )

    if not sol.success:
        raise RuntimeError(sol.message)

    return sol


def convergence_study(f, y0, t0, t1, h_values, exact):
    """Return final-time errors and observed order."""
    errors = []

    for h in h_values:
        t, y = solve_euler(f, y0, t0, t1, h)
        err = np.linalg.norm(y[-1] - exact(t1))
        errors.append(err)

    errors = np.asarray(errors, dtype=float)
    h_values = np.asarray(h_values, dtype=float)

    slope, intercept = np.polyfit(np.log(h_values), np.log(errors), 1)
    return errors, slope


def main():
    output_dir = Path(__file__).resolve().parent

    mu = 1.0
    y0 = np.array([0.5, 0.0])
    t0 = 0.0
    T = 10.0
    h_values = np.array([0.1, 0.05, 0.025, 0.0125])

    # Build scipy oracle.
    oracle = build_oracle(y0, t0, T)
    exact = lambda t: oracle.sol(t)

    # Required number: y1(T) from the oracle.
    oracle_at_T = exact(T)
    y1_T = oracle_at_T[0]

    # Required number: observed Explicit Euler order.
    errors, order = convergence_study(
        lambda t, y: vanderpol_rhs(t, y, mu),
        y0, t0, T, h_values, exact
    )

    print("=" * 64)
    print("Van der Pol Tutorial 1 results")
    print("=" * 64)
    print(f"Observed order (2 digits): {order:.2f}")
    print(f"y1(10) from scipy oracle (3 digits): {y1_T:.3f}")
    print()
    print("Error table")
    print(f"{'h':>10} {'L2 error at T':>18}")
    for h, err in zip(h_values, errors):
        print(f"{h:10.4f} {err:18.8e}")

    # Required phase-plane trajectory: y2 vs y1, from the oracle.
    tt = np.linspace(t0, T, 3000)
    yy = exact(tt)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(yy[0], yy[1])
    ax.set_xlabel("y1")
    ax.set_ylabel("y2")
    ax.set_title("Van der Pol phase-plane trajectory (mu = 1)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    phase_path = output_dir / "vanderpol_phase_plane.png"
    fig.savefig(phase_path, dpi=180)
    plt.close(fig)

    print()
    print(f"Saved required trajectory figure: {phase_path.name}")
    print("=" * 64)


if __name__ == "__main__":
    main()
