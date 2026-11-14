"""Pure-Neumann variant (optional depth, Problem Pack): -Δu = f on (0,1)^2, ∂_n u = q on all of ∂Ω.

All (N+1)^2 nodes are unknowns, i,j = 0..N, index k = j(N+1)+i (arrays reshape as (N+1, N+1),
axis 0 = y).  q is the OUTWARD normal derivative on each edge:
    left  x=0: q = -u_x,  right x=1: q = u_x,  bottom y=0: q = -u_y,  top y=1: q = u_y.

Closures (rows written in the unscaled form A U = F):
  "ghost": the five-point PDE row at every node; a missing neighbour across an edge is a ghost
           eliminated by the centred condition (U_ghost - U_inner)/(2h) = q, so the inner neighbour
           gets weight 2 and F gains 2q/h.  At a corner this happens in both directions:
               (4U_c - 2U_x - 2U_y)/h^2 = f_c + 2(q^x + q^y)/h .
  "first": interior PDE rows unchanged; every edge node (corners excluded) gets the one-sided row
               (U_b - U_in)/h = q ;
           a corner gets the average of its two one-sided rows,
               (2U_c - U_x - U_y)/(2h) = (q^x + q^y)/2 .
           No other row references a corner value, so the corner rows only post-process U.

Both matrices have zero row sums (A 1 = 0).  The left null vectors are
  ghost: w = trapezoid weights (1, 1/2 on edges, 1/4 at corners) = D 1, since D A is symmetric;
  first: w = 1 at interior nodes, 1/h at edge nodes, 0 at corners (D = diag(w) symmetrises the
         non-corner block, and corner columns contain only their own diagonal).
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

CLOSURES = ("first", "ghost")


def grid(N):
    J, I = np.meshgrid(np.arange(N + 1), np.arange(N + 1), indexing="ij")
    return I.ravel(), J.ravel()


def trap_weights(N):
    """Trapezoid weights 1, 1/2 (edges), 1/4 (corners) on the (N+1)^2 grid."""
    I, J = grid(N)
    wx = np.where((I == 0) | (I == N), 0.5, 1.0)
    wy = np.where((J == 0) | (J == N), 0.5, 1.0)
    return wx * wy


def left_null(N, closure):
    I, J = grid(N)
    h = 1.0 / N
    if closure == "ghost":
        return trap_weights(N)
    bx = (I == 0) | (I == N); by = (J == 0) | (J == N)
    return np.where(bx & by, 0.0, np.where(bx | by, 1.0 / h, 1.0))


def assemble(N, closure, f, qL, qR, qB, qT):
    """Return A (CSR, singular), F, h, I, J.  qL,qR take y; qB,qT take x (outward ∂_n u)."""
    if closure not in CLOSURES:
        raise ValueError(closure)
    h = 1.0 / N
    I, J = grid(N)
    x, y = I * h, J * h
    k = J * (N + 1) + I
    n = (N + 1) ** 2
    bx = (I == 0) | (I == N); by = (J == 0) | (J == N)
    qx = np.where(I == 0, qL(y), np.where(I == N, qR(y), 0.0))
    qy = np.where(J == 0, qB(x), np.where(J == N, qT(x), 0.0))
    rows, cols, vals = [], [], []
    F = np.zeros(n)

    def put(mask, c, v):
        rows.append(k[mask]); cols.append(c[mask]); vals.append(np.broadcast_to(v, k.shape)[mask])

    if closure == "ghost":
        allm = np.ones(n, bool)
        put(allm, k, 4.0 / h**2)
        # x-direction neighbours
        wW = np.where(I == N, 2.0, np.where(I == 0, 0.0, 1.0))
        wE = np.where(I == 0, 2.0, np.where(I == N, 0.0, 1.0))
        wS = np.where(J == N, 2.0, np.where(J == 0, 0.0, 1.0))
        wN = np.where(J == 0, 2.0, np.where(J == N, 0.0, 1.0))
        for w, c in ((wW, k - 1), (wE, k + 1), (wS, k - (N + 1)), (wN, k + (N + 1))):
            m = w > 0
            put(m, c, -w / h**2)
        F[:] = f(x, y) + 2.0 * (qx + qy) / h
    else:
        inner = ~(bx | by)
        put(inner, k, 4.0 / h**2)
        for c in (k - 1, k + 1, k - (N + 1), k + (N + 1)):
            put(inner, c, -1.0 / h**2)
        F[inner] = f(x, y)[inner]
        edge = (bx ^ by)
        nb = np.where(I == 0, k + 1, np.where(I == N, k - 1,
                      np.where(J == 0, k + (N + 1), k - (N + 1))))
        put(edge, k, 1.0 / h)
        put(edge, nb, -1.0 / h)
        F[edge] = (qx + qy)[edge]
        cor = bx & by
        nbx = np.where(I == 0, k + 1, k - 1)
        nby = np.where(J == 0, k + (N + 1), k - (N + 1))
        put(cor, k, 1.0 / h)
        put(cor, nbx, -0.5 / h)
        put(cor, nby, -0.5 / h)
        F[cor] = 0.5 * (qx + qy)[cor]
    A = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(n, n)).tocsr()
    return A, F, h, I, J


def defect(A_w, F):
    """Relative compatibility defect w^T F / (|w|^T |F|)."""
    return float(A_w @ F / (np.abs(A_w) @ np.abs(F)))


def project(F, w):
    """Make F compatible by subtracting one constant from every entry,
    F~ = F - c 1 with c = w^T F / w^T 1, so that w^T F~ = 0 (range(A) = w^⊥).
    For the ghost scheme w^T 1 = 1/h^2, so c = h^2 w^T F is the defect itself."""
    return F - (w @ F) / w.sum()


def wmean(v, N):
    om = trap_weights(N)
    return float(om @ v / om.sum())


def solve_mean(A, F, w, N):
    """Bordered system [[A, 1], [ω^T, 0]] [U; λ] = [F; 0] (ω = trapezoid weights): the trapezoid
    mean of U is 0 and λ = w^T F / w^T 1, so λ = 0 iff F is compatible (λ = h^2 w^T F for ghost)."""
    om = trap_weights(N)
    n = A.shape[0]
    B = sp.bmat([[A, sp.csr_matrix(np.ones((n, 1)))],
                 [sp.csr_matrix(om.reshape(1, -1)), None]], format="csc")
    z = spla.spsolve(B, np.concatenate([F, [0.0]]))
    return z[:n], float(z[n])


def solve_pin(A, F, p):
    """Pin U_p = 0: delete row and column p (row p must carry w_p != 0) and solve."""
    n = A.shape[0]
    keep = np.r_[0:p, p + 1:n]
    Ar = A[keep][:, keep].tocsc()
    U = np.zeros(n)
    U[keep] = spla.spsolve(Ar, F[keep])
    return U


def norms(e, N):
    """Trapezoid L2_h over all nodes, nodal L∞, and its split boundary / interior."""
    h = 1.0 / N
    I, J = grid(N)
    om = trap_weights(N)
    b = (I == 0) | (I == N) | (J == 0) | (J == N)
    return (float(np.sqrt(h**2 * om @ e**2)), float(np.abs(e).max()),
            float(np.abs(e[b]).max()), float(np.abs(e[~b]).max()))


# manufactured solution u* = e^{x+y}, f = -2 e^{x+y}; outward normal derivatives
def u_star(x, y):
    return np.exp(x + y)

def f_star(x, y):
    return -2.0 * np.exp(x + y)

Q_STAR = dict(qL=lambda y: -np.exp(y), qR=lambda y: np.exp(1.0 + y),
              qB=lambda x: -np.exp(x), qT=lambda x: np.exp(1.0 + x))
