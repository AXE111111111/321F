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


def _vec(fun, *args):
    """Evaluate user data on arrays; falls back to np.vectorize for scalar-only callables."""
    try:
        v = np.asarray(fun(*args), dtype=float)
        if v.shape != np.broadcast(*args).shape:
            v = np.broadcast_to(v, np.broadcast(*args).shape).astype(float)
        return v
    except (TypeError, ValueError):
        return np.vectorize(fun, otypes=[float])(*args)


def assemble(N: int, beta: float, closure: str, f, g, r, a=None):
    """Assemble A U = F for one (N, beta, closure), vectorised (no loop over grid points).

    PDE  -div(a grad u) = f ;  Dirichlet u=g on x=0, y=0, y=1 ;  u_x + beta u = r on x=1.
    f(x,y), g(x,y), r(y), a(x,y) are swappable callables (array-capable preferred).
    a=None means a == 1 (Laplacian) and reproduces the original assembly bit-for-bit.
    Half-point coefficients a_{i+-1/2,j} = a(x_i +- h/2, y_j) (midpoint sampling).
    Ghost closure with variable a: x-flux row 2 a_{N-1/2}(U_N-U_{N-1})/h^2 + 2 a_N (beta U_N - r)/h.
    First closure: boundary row is the BC itself (independent of a).
    Returns A (CSR), F, h, I, J (integer mesh indices of unknowns, lex order k=(j-1)N+(i-1)).
    """
    if closure not in ("first", "ghost"):
        raise ValueError(f"unknown closure {closure!r}")
    h = 1.0 / N
    M = N * (N - 1)
    Jg, Ig = np.meshgrid(np.arange(1, N), np.arange(1, N + 1), indexing="ij")
    I = Ig.ravel(); J = Jg.ravel()
    k = (J - 1) * N + (I - 1)
    x = I * h; y = J * h
    if a is None:
        aE = aW = aN = aS = aC = np.ones(M)
    else:
        aE = _vec(a, x + h / 2, y); aW = _vec(a, x - h / 2, y)
        aN = _vec(a, x, y + h / 2); aS = _vec(a, x, y - h / 2)
        aC = _vec(a, x, y)
    edge = I == N
    pde = ~edge if closure == "first" else np.ones(M, bool)
    F = np.zeros(M)
    rows, cols, vals = [], [], []

    def put(mask, c, v):
        rows.append(k[mask]); cols.append(c[mask] if np.ndim(c) else np.full(mask.sum(), c)); vals.append(v[mask])

    # weights: east/west (ghost edge: west doubled, east removed), south, north
    wW = np.where(edge, 2.0 * aW, aW)
    wE = np.where(edge, 0.0, aE)
    diag = (wW + wE + aS + aN) / h**2 + np.where(edge, 2.0 * beta * aC / h, 0.0)
    F[pde] = _vec(f, x, y)[pde]
    rr = _vec(r, y)
    F[pde & edge] += (2.0 * aC * rr / h)[pde & edge]
    put(pde, k, diag)
    for (p, q, w) in ((I - 1, J, wW), (I + 1, J, wE), (I, J - 1, aS), (I, J + 1, aN)):
        nz = pde & (w != 0)
        bnd = nz & ((p == 0) | (q == 0) | (q == N))
        inn = nz & ~bnd
        if bnd.any():
            F[bnd] += w[bnd] * _vec(g, p[bnd] * h, q[bnd] * h) / h**2
        put(inn, (q - 1) * N + (p - 1), -w / h**2)
    if closure == "first":
        put(edge, k, np.full(M, 1.0 / h + beta))
        put(edge, k - 1, np.full(M, -1.0 / h))
        F[edge] = rr[edge]
    A = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(M, M)).tocsr()
    return A, F, h, I, J


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
