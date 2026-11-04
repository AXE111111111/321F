"""Direction 2 five-point FDM on (0,1)^2 with mixed BC.

PDE:  -Δu = f
Dirichlet on Γ_D = {x=0} ∪ {y=0} ∪ {y=1} (owns all four corners).
Flux law on Γ_B = {x=1}×(0,1):  u_x + β u = r.

Closures of the right-edge row:
  "first" — replace PDE row by one-sided BC  (1/h+β)U_N - (1/h)U_{N-1} = r
  "ghost" — centred BC via ghost, eliminate into five-point row

Unknowns: i=1..N, j=1..N-1  (M = N(N-1)), lex order k=(j-1)*N+(i-1).
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def index(i: int, j: int, N: int) -> int:
    return (j - 1) * N + (i - 1)


def assemble(N: int, beta: float, closure: str, f, g, r):
    """Assemble A U = F for one (N, beta, closure).

    f(x,y), g(x,y) Dirichlet data, r(y) flux-edge data.
    Returns A (CSR), F, h, I, J (integer mesh indices of unknowns).
    """
    if closure not in ("first", "ghost"):
        raise ValueError(f"unknown closure {closure!r}")
    h = 1.0 / N
    M = N * (N - 1)
    rows, cols, vals = [], [], []
    F = np.zeros(M)

    def add(k, c, v):
        rows.append(k)
        cols.append(c)
        vals.append(v)

    for j in range(1, N):
        y = j * h
        for i in range(1, N + 1):
            k = index(i, j, N)
            x = i * h
            if i == N and closure == "first":
                add(k, k, 1.0 / h + beta)
                add(k, index(N - 1, j, N), -1.0 / h)
                F[k] = r(y)
                continue
            # five-point (ghost on right edge if i==N)
            diag = 4.0 / h**2
            F[k] = f(x, y)
            if i == N:
                diag += 2.0 * beta / h
                F[k] += 2.0 * r(y) / h
                neighbours = [(i - 1, j, 2.0), (i, j - 1, 1.0), (i, j + 1, 1.0)]
            else:
                neighbours = [(i - 1, j, 1.0), (i + 1, j, 1.0), (i, j - 1, 1.0), (i, j + 1, 1.0)]
            add(k, k, diag)
            for p, q, w in neighbours:
                if p == 0 or q == 0 or q == N:
                    F[k] += w * g(p * h, q * h) / h**2
                else:
                    add(k, index(p, q, N), -w / h**2)

    A = sp.csr_matrix((vals, (rows, cols)), shape=(M, M))
    I, J = np.meshgrid(np.arange(1, N + 1), np.arange(1, N), indexing="xy")
    return A, F, h, I.ravel(), J.ravel()


def scale_vector(I: np.ndarray, h: float, closure: str) -> np.ndarray:
    """Diagonal weights that make DA symmetric (Prop. in the report)."""
    if closure == "ghost":
        return np.where(I == I.max(), 0.5, 1.0)
    return np.where(I == I.max(), 1.0 / h, 1.0)


def symmetrise(A, F, I, h, closure: str):
    D = scale_vector(I, h, closure)
    return (sp.diags(D) @ A).tocsr(), D * F, D


def solve_direct(A, F):
    return spla.spsolve(A.tocsc(), F)


def solve_cg(A, F, tol=1e-12, maxiter=20000):
    it = [0]

    def cb(_):
        it[0] += 1

    x, info = spla.cg(A, F, rtol=tol, atol=0.0, maxiter=maxiter, callback=cb)
    relres = np.linalg.norm(F - A @ x) / np.linalg.norm(F)
    return x, it[0], relres, info


def asymmetry(A) -> float:
    return float(spla.norm(A - A.T) / spla.norm(A))


def scaled_residual(A, U, F) -> float:
    den = abs(A) @ np.abs(U) + np.abs(F)
    return float(np.max(np.abs(F - A @ U) / den))


def discrete_norms(e: np.ndarray, I: np.ndarray, h: float):
    """Nodal L∞ and trapezoidal L2_h (half-weight on right-edge nodes)."""
    w = np.where(I == I.max(), 0.5, 1.0)
    L2 = float(np.sqrt(h**2 * np.sum(w * e**2)))
    Linf = float(np.abs(e).max())
    b = I == I.max()
    return L2, Linf, float(np.abs(e[b]).max()), float(np.abs(e[~b]).max())
