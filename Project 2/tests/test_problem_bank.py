"""Analytic test-problem bank for Project 2 (student-facing).

Every entry states the PDE, an exact solution, the forcing, the boundary
data, and the asymptotic rate a correct five-point FDM should approach in
the discrete L^2 norm (h * ||.||_2) with the indicated boundary treatment.

The bank is verification infrastructure, not a topic list: manufacture a
problem, prove the code, then trust it on the team's real problem.

The examples used in class are imported from here by the scaffolding:
P1 (zero Dirichlet, RHS = f) is the Lecture 3 coding-lab example; P5
(non-zero Dirichlet, boundary values moved across) is the Tutorial 5
individual-challenge example. Switch PROBLEM in poisson_fd_skeleton.py
to pick which one the scaffold solves.
"""
import numpy as np


def _one(x, y):
    return 1.0 + 0.0 * x * y


def _one_plus_x(x, y):
    return 1.0 + x


class Problem:
    """One manufactured test problem on the unit square."""

    def __init__(self, name, exact, forcing, boundary, expected_rate, note,
                 coefficient=_one):
        self.name = name
        self.exact = exact
        self.forcing = forcing
        self.boundary = boundary
        self.expected_rate = expected_rate
        self.note = note
        self.coefficient = coefficient


def _p1(x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y)


def _f1(x, y):
    return 2.0 * np.pi**2 * _p1(x, y)


def _p2(x, y):
    return x * x - y * y


def _f2(x, y):
    return 0.0 * x * y


def _p3(x, y):
    return np.sin(np.pi * x) * np.exp(y)


def _f3(x, y):
    return (np.pi**2 - 1.0) * np.sin(np.pi * x) * np.exp(y)


def _p4(x, y):
    return _p1(x, y)  # reused under a variable coefficient


def _f4(x, y):
    return 2.0 * np.pi**2 * (1.0 + x) * _p1(x, y) - np.pi * np.cos(np.pi * x) * np.sin(np.pi * y)


def _p5(x, y):
    return np.exp(x) * np.sin(np.pi * y)


def _f5(x, y):
    return (np.pi**2 - 1.0) * _p5(x, y)


P1 = Problem(
    name="P1 trigonometric mode (Dirichlet)",
    exact=_p1,
    forcing=_f1,
    boundary="u = 0 on the whole boundary",
    expected_rate=2.0,
    note="The workhorse, and the Lecture 3 coding-lab example. Homogeneous "
    "Dirichlet data keep the boundary contribution to the right-hand side "
    "zero, so b = f and the lab tests the interior stencil alone.",
)
P2 = Problem(
    name="P2 harmonic polynomial (Dirichlet)",
    exact=_p2,
    forcing=_f2,
    boundary="u = x^2 - y^2 on the boundary (non-homogeneous)",
    expected_rate=2.0,
    note="Laplace equation with nonzero data: tests that boundary values "
    "enter the right-hand side with the correct sign and scaling. The "
    "five-point stencil reproduces this quadratic exactly, so the error is "
    "round-off --- a useful sign check, not a convergence study.",
)
P3 = Problem(
    name="P3 exponential-trig mode (mixed Dirichlet/Neumann)",
    exact=_p3,
    forcing=_f3,
    boundary="Dirichlet on x in {0,1}, y = 0, and the edge y = 1 uses "
    "Neumann data du/dy = sin(pi x) e",
    expected_rate="2 with a second-order (ghost-point) closure; 1 with a "
    "one-sided closure",
    note="The boundary treatment is part of the method: the observed rate "
    "should expose which closure the code actually implements. Expected "
    "forcing: f = (pi^2 - 1) sin(pi x) e^y because -d_xx u - d_yy u = "
    "(pi^2 - 1) u.",
)
P4 = Problem(
    name="P4 variable diffusion (Dirichlet)",
    exact=_p4,
    forcing=_f4,
    boundary="u = 0 on the whole boundary",
    expected_rate=2.0,
    note="Solves -div(a grad u) = f with a(x, y) = 1 + x > 0. The "
    "discrete operator stays symmetric if fluxes at half-points are "
    "averaged consistently; state whether yours does and why. Core "
    "direction 3 of the problem pack.",
    coefficient=_one_plus_x,
)
P5 = Problem(
    name="P5 exponential-trig mode (non-zero Dirichlet)",
    exact=_p5,
    forcing=_f5,
    boundary="u = e^x sin(pi y) on the whole boundary (non-zero on x = 0, 1)",
    expected_rate=2.0,
    note="The Tutorial 5 individual-challenge example. Non-zero Dirichlet "
    "data mean the boundary values must be moved across into the right-hand "
    "side: b[k] += u(x_edge, y_edge)/h^2. Where P1 tests the interior "
    "stencil alone, P5 tests the boundary rows on top of it. Centre "
    "benchmark: u_h(0.5,0.5) = 1.65200 at n = 15 (h = 1/16), 1.64954 at "
    "n = 31.",
)

BANK = [P1, P2, P3, P4, P5]


def verify_bank(n=400, tol=2e-3):
    """Guard against typos: check forcing = -div(a grad u) by finite
    differences, using the conservative flux form on half-points."""
    h = 1.0 / (n - 1)
    axis = np.linspace(0.0, 1.0, n)
    X, Y = np.meshgrid(axis, axis, indexing="ij")
    worst = 0.0
    for prob in BANK:
        u = prob.exact(X, Y)
        a = prob.coefficient
        fx = np.zeros_like(u)
        fy = np.zeros_like(u)
        axp = a(X + 0.5 * h, Y)
        axm = a(X - 0.5 * h, Y)
        ayp = a(X, Y + 0.5 * h)
        aym = a(X, Y - 0.5 * h)
        fx[1:-1, :] = (axp[1:-1, :] * (u[2:, :] - u[1:-1, :])
                       - axm[1:-1, :] * (u[1:-1, :] - u[:-2, :])) / h**2
        fy[:, 1:-1] = (ayp[:, 1:-1] * (u[:, 2:] - u[:, 1:-1])
                       - aym[:, 1:-1] * (u[:, 1:-1] - u[:, :-2])) / h**2
        rhs = -(fx + fy)
        fvals = prob.forcing(X, Y)[2:-2, 2:-2]
        scale = max(np.abs(fvals).max(), np.abs(u).max())
        rel = np.abs(rhs[2:-2, 2:-2] - fvals).max() / scale if scale > 0 else 0.0
        worst = max(worst, rel)
        assert rel < tol, f"{prob.name}: forcing check failed (rel={rel:.2e})"
    return worst


if __name__ == "__main__":
    worst = verify_bank()
    print(f"forcing consistency check passed (worst relative error {worst:.1e})")
    for prob in BANK:
        print(f"\n{prob.name}\n  BCs: {prob.boundary}\n  expected rate: "
              f"{prob.expected_rate}\n  {prob.note}")
