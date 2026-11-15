"""Five-gate audit (AI-solver audit worksheet, Project 2) of the AI-assisted solver code in
MTH321_P2_code_submission.zip: code/solver.py (vectorised assemble, scale_vector, solve_cg)
and the pure-Neumann code added 14 Nov (code/neumann_pure.py).
Reference formulas are typed from the report (eqs. fivept, first, ghost, pncorner, pnborder).
Run:  python five_gate_audit.py <path to unpacked MTH321_Project2_TeamF_Code>
"""
import sys, os, datetime, platform
import numpy as np
ROOT = sys.argv[1]
sys.path.insert(0, os.path.join(ROOT, "code"))
import scipy, scipy.sparse as sp, scipy.sparse.linalg as spla
import solver as S
import neumann_pure as P
import mms

out = []
def log(s=""):
    print(s); out.append(s)

log(f"five_gate_audit.py run {datetime.datetime.now().isoformat(timespec='seconds')} (Asia/Shanghai)")
log(f"python {platform.python_version()} numpy {np.__version__} scipy {scipy.__version__}")
log(f"code root: {ROOT}")
RES = {}

# ---- test data (deliberately generic, not the MMS) ------------------------------------
fT = lambda x, y: np.sin(3 * x) + x * y**2 + 0.3
gT = lambda x, y: np.cos(x + 2 * y) + x
rT = lambda y: 1.0 + y**3

def ref_rows_mixed(N, beta, closure, f, g, r):
    """Rows typed from report eqs. (fivept), (first), (ghost): dict (i,j) -> ({(i',j'):coef}, rhs).
    Dirichlet neighbours (i'=0, j'=0, j'=N) are moved to the right-hand side."""
    h = 1.0 / N
    rows = {}
    for j in range(1, N):
        for i in range(1, N + 1):
            x, y = i * h, j * h
            if i < N:                      # (fivept)
                c = {(i, j): 4 / h**2}; nb = [(i - 1, j, 1.), (i + 1, j, 1.), (i, j - 1, 1.), (i, j + 1, 1.)]
                rhs = f(x, y)
            elif closure == "first":        # (first)
                c = {(i, j): 1 / h + beta, (i - 1, j): -1 / h}; nb = []; rhs = r(y)
            else:                           # (ghost)
                c = {(i, j): 4 / h**2 + 2 * beta / h}
                nb = [(i - 1, j, 2.), (i, j - 1, 1.), (i, j + 1, 1.)]
                rhs = f(x, y) + 2 * r(y) / h
            for (p, q, w) in nb:
                if p == 0 or q == 0 or q == N:
                    rhs += w * g(p * h, q * h) / h**2
                else:
                    c[(p, q)] = c.get((p, q), 0.0) - w / h**2
            rows[(i, j)] = (c, rhs)
    return rows

def dense_from_rows(rows, order):
    pos = {node: n for n, node in enumerate(order)}
    A = np.zeros((len(order), len(order))); F = np.zeros(len(order))
    for node, (c, rhs) in rows.items():
        for nb, v in c.items():
            A[pos[node], pos[nb]] += v
        F[pos[node]] = rhs
    return A, F

# =========================================================================================
log("\n=== Gate 1: transcription (code rows vs report equations, term by term) ===")
g1 = 0.0
for N in (2, 3, 6, 9):
    for beta in (0.0, 2.0, 0.7):
        for cl in ("first", "ghost"):
            A, F, h, I, J = S.assemble(N, beta, cl, fT, gT, rT)
            order = [(int(i), int(j)) for i, j in zip(I, J)]       # solver's own order, only to align
            Ar, Fr = dense_from_rows(ref_rows_mixed(N, beta, cl, fT, gT, rT), order)
            d = max(np.abs(A.toarray() - Ar).max() / np.abs(Ar).max(), np.abs(F - Fr).max() / np.abs(Fr).max())
            g1 = max(g1, d)
log(f"mixed problem, N in {{2,3,6,9}}, beta in {{0,0.7,2}}, both closures: max rel |A-A_ref|, |F-F_ref| = {g1:.2e}")
RES["g1_mixed"] = g1
# MMS data signs against the problem statement, checked by centred differences
x0, y0, e = 0.37, 0.61, 1e-4
lap = (mms.u_star(x0 + e, y0) + mms.u_star(x0 - e, y0) + mms.u_star(x0, y0 + e) + mms.u_star(x0, y0 - e) - 4 * mms.u_star(x0, y0)) / e**2
ux = lambda y: (mms.u_star(1 + e, y) - mms.u_star(1 - e, y)) / (2 * e)
log(f"MMS: -Lap u* - f* at (0.37,0.61) = {-lap - mms.f_star(x0, y0):.1e}; u_x+beta u - r at (1,0.61), beta=2: {ux(y0) + 2 * mms.u_star(1, y0) - mms.r_star(y0, 2.0):.1e}")
# pure-Neumann outward data
qs = P.Q_STAR
du = lambda X, Y, dx, dy: (P.u_star(X + dx * e, Y + dy * e) - P.u_star(X - dx * e, Y - dy * e)) / (2 * e)
qerr = max(abs(qs["qL"](y0) + du(0, y0, 1, 0)), abs(qs["qR"](y0) - du(1, y0, 1, 0)),
           abs(qs["qB"](x0) + du(x0, 0, 0, 1)), abs(qs["qT"](x0) - du(x0, 1, 0, 1)))
log(f"pure-Neumann outward q vs centred differences of u*: max diff {qerr:.1e}; -Lap u* - f* = {-lap - P.f_star(x0, y0):.1e}")

def ref_rows_pure(N, closure, f, qL, qR, qB, qT):
    """Rows typed from report Sec. 6.3 (ghost edge/corner rows, eq. pncorner; first-order rows)."""
    h = 1.0 / N; rows = {}
    for j in range(N + 1):
        for i in range(N + 1):
            x, y = i * h, j * h
            qx = qL(y) if i == 0 else (qR(y) if i == N else 0.0)
            qy = qB(x) if j == 0 else (qT(x) if j == N else 0.0)
            onx, ony = i in (0, N), j in (0, N)
            if closure == "ghost":
                c = {(i, j): 4 / h**2}
                for (p, q) in ((i - 1, j), (i + 1, j)):
                    if 0 <= p <= N:
                        c[(p, q)] = c.get((p, q), 0) - (2 if onx else 1) / h**2
                for (p, q) in ((i, j - 1), (i, j + 1)):
                    if 0 <= q <= N:
                        c[(p, q)] = c.get((p, q), 0) - (2 if ony else 1) / h**2
                rows[(i, j)] = (c, f(x, y) + 2 * (qx + qy) / h)
            else:
                if not onx and not ony:
                    c = {(i, j): 4 / h**2, (i - 1, j): -1 / h**2, (i + 1, j): -1 / h**2, (i, j - 1): -1 / h**2, (i, j + 1): -1 / h**2}
                    rows[(i, j)] = (c, f(x, y))
                elif onx and ony:
                    ix = i + 1 if i == 0 else i - 1; jy = j + 1 if j == 0 else j - 1
                    rows[(i, j)] = ({(i, j): 2 / (2 * h), (ix, j): -1 / (2 * h), (i, jy): -1 / (2 * h)}, (qx + qy) / 2)
                else:
                    inn = (i + 1, j) if i == 0 else (i - 1, j) if i == N else (i, j + 1) if j == 0 else (i, j - 1)
                    rows[(i, j)] = ({(i, j): 1 / h, inn: -1 / h}, qx + qy)
    return rows

qT_ = dict(qL=lambda y: np.cos(y), qR=lambda y: 1 + y, qB=lambda x: x**2, qT=lambda x: -np.sin(x))
g1p = 0.0
for N in (2, 3, 6, 9):
    for cl in P.CLOSURES:
        A, F, h, I, J = P.assemble(N, cl, fT, **qT_)
        order = [(int(i), int(j)) for i, j in zip(I, J)]
        Ar, Fr = dense_from_rows(ref_rows_pure(N, cl, fT, **qT_), order)
        g1p = max(g1p, np.abs(A.toarray() - Ar).max() / np.abs(Ar).max(), np.abs(F - Fr).max() / np.abs(Fr).max())
log(f"pure-Neumann rows (edge, corner, interior), N in {{2,3,6,9}}, both closures: max rel diff = {g1p:.2e}")
RES["g1_pure"] = g1p

# =========================================================================================
log("\n=== Gate 2: structure and indexing ===")
ok_idx = True
for N in (4, 7, 16):
    A, F, h, I, J = S.assemble(N, 2.0, "ghost", fT, gT, rT)
    M = N * (N - 1)
    for (i, j) in ((1, 1), (N, 1), ((N + 1) // 2, N // 2), (1, N - 1), (N, N - 1)):
        k = (j - 1) * N + (i - 1)
        ok_idx &= (I[k] == i and J[k] == j and S.index(i, j, N) == k)
    U = np.arange(M, dtype=float)
    ok_idx &= U.reshape(N - 1, N)[N // 2 - 1, N - 1] == S.index(N, N // 2, N)
log(f"mixed k=(j-1)N+(i-1): ends (1,1),(N,1),(1,N-1),(N,N-1) and centre, N in {{4,7,16}}, reshape (N-1,N) axis0=y: {'all match' if ok_idx else 'MISMATCH'}")
ok_p = True
for N in (4, 7):
    I, J = P.grid(N)
    for (i, j) in ((0, 0), (N, 0), (N // 2, N // 2), (0, N), (N, N)):
        k = j * (N + 1) + i; ok_p &= (I[k] == i and J[k] == j)
log(f"pure-Neumann k=j(N+1)+i: corners and centre, N in {{4,7}}: {'all match' if ok_p else 'MISMATCH'}")
RES["g2_idx"] = ok_idx and ok_p
nnz_info = []
for N in (16, 256):
    for cl in ("first", "ghost"):
        A, *_ = S.assemble(N, 2.0, cl, mms.f_star, mms.g_star, lambda y: mms.r_star(y, 2.0))
        Ar_nnz = sum(len(c) for c, _ in ref_rows_mixed(N, 2.0, cl, mms.f_star, mms.g_star, lambda y: mms.r_star(y, 2.0)).values()) if N == 16 else None
        nnz_info.append((N, cl, type(A).__name__, A.nnz, A.shape[0], Ar_nnz))
for t in nnz_info:
    log(f"N={t[0]:3d} {t[1]:5s}: {t[2]}, nnz={t[3]} ({t[3]/t[4]:.2f} per row), M={t[4]}" + (f", reference row count nnz={t[5]}" if t[5] else ""))
RES["nnz256"] = [t for t in nnz_info if t[0] == 256]
# non-standard rows written out at N=4 (ghost, beta=2) for the log
A, F, h, I, J = S.assemble(4, 2.0, "ghost", fT, gT, rT)
for (i, j) in ((4, 1), (4, 2), (3, 2)):
    k = S.index(i, j, 4); row = A.getrow(k)
    log(f"  N=4 ghost beta=2 row (i,j)=({i},{j}) h^2*coeffs: " + ", ".join(f"({I[c]},{J[c]}):{v*h*h:+.3g}" for c, v in zip(row.indices, row.data)))
# symmetrisation weights (Prop. sym), constant a
asym = []
for N in (8, 64):
    for cl in ("first", "ghost"):
        for beta in (0.0, 2.0):
            A, F, h, I, J = S.assemble(N, beta, cl, fT, gT, rT)
            DA, DF, D = S.symmetrise(A, F, I, h, cl)
            asym.append(abs(DA - DA.T).max())
log(f"max|DA-DA^T| (a=1, N in {{8,64}}, both closures, beta in {{0,2}}) = {max(asym):.1e}")
RES["g2_sym"] = max(asym)

# =========================================================================================
log("\n=== Gate 3: linearisation (worksheet item is for nonlinear problems) ===")
log("Problem is linear: residual R(U)=AU-F, Jacobian J=A exactly. Directional-derivative test anyway:")
rng = np.random.default_rng(321)
A, F, h, I, J = S.assemble(32, 2.0, "ghost", mms.f_star, mms.g_star, lambda y: mms.r_star(y, 2.0))
U0 = rng.standard_normal(A.shape[0])
R = lambda U: A @ U - F
worst = {}
for eps in (1e-2, 1e-4, 1e-6, 1e-8):
    m = 0.0
    for _ in range(5):
        v = rng.standard_normal(A.shape[0])
        fd = (R(U0 + eps * v) - R(U0 - eps * v)) / (2 * eps)
        m = max(m, np.linalg.norm(fd - A @ v) / np.linalg.norm(A @ v))
    worst[eps] = m
log("  N=32 ghost beta=2, 5 random v each: rel|FD - Av| = " + ", ".join(f"eps={k:.0e}: {v:.1e}" for k, v in worst.items()))
log("  (no truncation term to shrink: the discrepancy is round-off amplified by 1/eps, as expected for a linear map)")
RES["g3_jac"] = worst
# stopping criterion of solve_cg
A, F, h, I, J = S.assemble(64, 2.0, "ghost", mms.f_star, mms.g_star, lambda y: mms.r_star(y, 2.0))
DA, DF, D = S.symmetrise(A, F, I, h, "ghost")
Ud = S.solve_direct(A, F)
x, it, rel, info = S.solve_cg(DA, DF, tol=1e-10)
x5, it5, rel5, info5 = S.solve_cg(DA, DF, tol=1e-10, maxiter=5)
log(f"  solve_cg stops on rtol (relative residual, atol=0) with maxiter and returns info, iterations and the true relres.")
log(f"  N=64 ghost beta=2, tol=1e-10: info={info}, iters={it}, relres={rel:.2e}, max|U_cg-U_spsolve|={np.abs(x-Ud).max():.1e}")
log(f"  same with maxiter=5: info={info5} (nonzero = cap reached, reported, not silent), iters={it5}, relres={rel5:.2e}")
RES["g3_cg"] = (info, it, rel, np.abs(x - Ud).max(), info5, rel5)
Ap, Fp, hp, Ip, Jp = P.assemble(64, "ghost", P.f_star, **P.Q_STAR)
wp = P.left_null(64, "ghost"); Fpp = P.project(Fp, wp)
Dp = sp.diags(wp); it_ = [0]
xp, infop = spla.cg((Dp @ Ap).tocsr(), wp * Fpp, rtol=1e-10, atol=0.0, maxiter=5000, callback=lambda _: it_.__setitem__(0, it_[0] + 1))
log(f"  pure-Neumann CG (as run_pure_neumann.py: DA, projected F, tol 1e-10, maxiter 5000): info={infop}, iters={it_[0]}")
RES["g3_pncg"] = (infop, it_[0])

# =========================================================================================
log("\n=== Gate 4: independent reproduction ===")
def dense_mixed_solve(N, beta, cl, f, g, r):
    # own ordering (i-major, column by column) and dense LAPACK solve, independent of solver.py
    order = [(i, j) for i in range(1, N + 1) for j in range(1, N)]
    Ar, Fr = dense_from_rows(ref_rows_mixed(N, beta, cl, f, g, r), order)
    Ur = np.linalg.solve(Ar, Fr)
    return {node: Ur[n] for n, node in enumerate(order)}
d4 = 0.0
for N in (4, 8, 16, 32):
    for beta in (0.0, 2.0):
        for cl in ("first", "ghost"):
            A, F, h, I, J = S.assemble(N, beta, cl, fT, gT, rT)
            U = S.solve_direct(A, F)
            Ur = dense_mixed_solve(N, beta, cl, fT, gT, rT)
            d4 = max(d4, max(abs(U[k] - Ur[(int(I[k]), int(J[k]))]) for k in range(len(U))) / np.abs(U).max())
log(f"dense numpy re-implementation (own ordering, np.linalg.solve) vs solver spsolve, N in {{4,8,16,32}}, beta in {{0,2}}, both closures: max rel diff {d4:.1e}")
RES["g4_dense"] = d4
# MMS order with the independent dense code
for beta in (0.0, 2.0):
    for cl in ("first", "ghost"):
        errs = []
        Ns = (8, 16, 32, 64)
        for N in Ns:
            Ur = dense_mixed_solve(N, beta, cl, mms.f_star, mms.g_star, lambda y: mms.r_star(y, beta))
            errs.append(max(abs(v - mms.u_star(i / N, j / N)) for (i, j), v in Ur.items()))
        p = [np.log2(errs[n] / errs[n + 1]) for n in range(len(errs) - 1)]
        log(f"  dense code, u*=e^(x+y), beta={beta:g} {cl:5s}: Linf N=8..64 = " + ", ".join(f"{e:.3e}" for e in errs) + "; orders " + ", ".join(f"{q:.2f}" for q in p))
        RES[f"g4_ord_{cl}_{beta:g}"] = p[-1]
# exact-solution limit cases (hand calculation): quadratic u is reproduced exactly by the ghost scheme
uq = lambda x, y: x**2 + x * y + y**2 + 1.0
fq = lambda x, y: -4.0 + 0 * x
for beta in (0.0, 2.0):
    rq = lambda y, b=beta: (2 + y) + b * uq(1.0, y)
    out_ = []
    for cl in ("ghost", "first"):
        A, F, h, I, J = S.assemble(16, beta, cl, fq, uq, rq)
        U = S.solve_direct(A, F); out_.append(np.abs(U - uq(I * h, J * h)).max())
    log(f"  quadratic u=x^2+xy+y^2+1, N=16, beta={beta:g}: ghost max err {out_[0]:.1e} (exact in exact arithmetic), first-order {out_[1]:.1e} (O(h) expected)")
    RES[f"g4_quad_{beta:g}"] = out_
# pure-Neumann: dense, null vectors from SVD (not from left_null), lstsq, trapezoid mean removed
d4p = 0.0; dw = 0.0
for N in (4, 8, 16):
    for cl in P.CLOSURES:
        A, F, h, I, J = P.assemble(N, cl, P.f_star, **P.Q_STAR)
        Ad = A.toarray()
        Us, s, Vt = np.linalg.svd(Ad)
        wl = Us[:, -1]; wl = wl / wl[np.argmax(np.abs(wl))]
        wf = P.left_null(N, cl); wf = wf / wf[np.argmax(np.abs(wf))]
        dw = max(dw, np.abs(wl - wf).max())
        Ft = F - (wl @ F) / wl.sum()
        Ul = np.linalg.lstsq(Ad, Ft, rcond=None)[0]
        om = P.trap_weights(N); Ul -= om @ Ul / om.sum()
        Um, lam = P.solve_mean(A, P.project(F, P.left_null(N, cl)), P.left_null(N, cl), N)
        d4p = max(d4p, np.abs(Ul - Um).max() / np.abs(Um).max())
log(f"pure-Neumann: SVD left null vector vs left_null() formula, N in {{4,8,16}}: max diff {dw:.1e}")
log(f"pure-Neumann: dense lstsq on SVD-projected data, trapezoid mean removed, vs solve_mean: max rel diff {d4p:.1e}")
RES["g4_pn"] = (dw, d4p)
for cl in P.CLOSURES:
    errs = []
    for N in (8, 16, 32, 64):
        A, F, h, I, J = P.assemble(N, cl, P.f_star, **P.Q_STAR)
        w = P.left_null(N, cl)
        U, lam = P.solve_mean(A, P.project(F, w), w, N)
        ue = P.u_star(I * h, J * h)
        errs.append(np.abs((U - P.wmean(U, N)) - (ue - P.wmean(ue, N))).max())
    p = [np.log2(errs[n] / errs[n + 1]) for n in range(3)]
    log(f"  pure-Neumann {cl:5s}: Linf (mod const) N=8..64 = " + ", ".join(f"{e:.3e}" for e in errs) + "; orders " + ", ".join(f"{q:.2f}" for q in p))
    RES[f"g4_pnord_{cl}"] = p[-1]

# =========================================================================================
log("\n=== Gate 5: interrogation (hypotheses that convergence would not reveal) ===")
log("Hypotheses were proposed by AI (Codex r17 review; this audit) and tested here, not accepted on trust.")
# H1 corner treatment, mixed problem: corner Dirichlet values
for cl in ("first", "ghost"):
    N = 16
    A, F, h, I, J = S.assemble(N, 2.0, cl, fT, gT, rT)
    bump = lambda x, y: gT(x, y) + 1.0 * ((np.isclose(x, 1) | np.isclose(x, 0)) & (np.isclose(y, 0) | np.isclose(y, 1)))
    A2, F2, *_ = S.assemble(N, 2.0, cl, fT, bump, rT)
    dU = np.abs(S.solve_direct(A2, F2) - S.solve_direct(A, F)).max()
    log(f"H1 corners (N=16, beta=2, {cl}): unknowns at corners = {int(((I==N)&((J==0)|(J==N))).sum())}; changing g by 1 at the 4 corners changes F in {int((F2!=F).sum())} rows, max|dU|={dU:.2e}")
    RES[f"g5_h1_{cl}"] = (int((F2 != F).sum()), dU)
log("   -> first-order rows never read corner data; ghost rows (N,1),(N,N-1) read g(1,0), g(1,1) as their Dirichlet neighbour, as eq. (ghost) states.")
# H2 pure-Neumann compatibility
for cl in P.CLOSURES:
    N = 32
    A, F, h, I, J = P.assemble(N, cl, P.f_star, **P.Q_STAR)
    w = P.left_null(N, cl)
    r1 = np.abs(A @ np.ones(A.shape[0])).max() / abs(A).max()
    rw = np.abs(A.T @ w).max() / (abs(A).max() * np.abs(w).max())
    Ft = P.project(F, w)
    Um, lam = P.solve_mean(A, F, w, N)
    c = (w @ F) / w.sum()
    log(f"H2 pure-Neumann {cl:5s} N=32: |A1|/|A|={r1:.1e}, |A^T w|={rw:.1e}, defect h^2 w^T F={h*h*(w@F):.3e}, w^T F~/|w||F|={abs(w@Ft)/(np.abs(w)@np.abs(F)):.1e}, bordered lambda (raw F)={lam:.4e} vs c={c:.4e}")
    RES[f"g5_h2_{cl}"] = (r1, rw, lam, c)
# H3 variable coefficient symmetrisation
def a1x(x, y): return 1.0 + x
A, F, h, I, J = S.assemble(2, 0.0, "first", fT, gT, rT, a=a1x)
DA, DF, D = S.symmetrise(A, F, I, h, "first")
h3 = abs(DA - DA.T).max()
log(f"H3 a=1+x, N=2, beta=0, first: A={A.toarray().tolist()}, D={D.tolist()}, max|DA-DA^T|={h3:g}, eta(DA)={S.asymmetry(DA):.4f}")
h3b = []
for N in (8, 32):
    for cl in ("first", "ghost"):
        A, F, h, I, J = S.assemble(N, 2.0, cl, fT, gT, rT, a=a1x)
        DA, DF, D = S.symmetrise(A, F, I, h, cl)
        h3b.append((N, cl, S.asymmetry(DA)))
log("   a=1+x, beta=2: eta(DA) " + ", ".join(f"N={n} {c}: {v:.1e}" for n, c, v in h3b))
A0, F0, *_ = S.assemble(16, 2.0, "first", fT, gT, rT)
A1_, F1_, *_ = S.assemble(16, 2.0, "first", fT, gT, rT, a=lambda x, y: 1.0 + 0 * x)
log(f"   a=None vs a(x,y)=1 callable, N=16: max|dA|={abs(A0-A1_).max():.1e}, max|dF|={np.abs(F0-F1_).max():.1e}")
log("   -> first-order D=diag(1/h) symmetrises DA only for a==1 (all report experiments); ghost DA stays symmetric for a=1+x.")
log("   -> P4 (a=1+x) is solved with spsolve, not CG, so no reported number depends on the asymmetric case.")
RES["g5_h3"] = (h3, h3b)
# H4 boundary flux sign (a sign slip still converges with smooth data only if r has the wrong sign)
A, F, h, I, J = S.assemble(64, 2.0, "ghost", mms.f_star, mms.g_star, lambda y: mms.r_star(y, 2.0))
U = S.solve_direct(A, F)
A2, F2, *_ = S.assemble(64, 2.0, "ghost", mms.f_star, mms.g_star, lambda y: -mms.r_star(y, 2.0))
U2 = S.solve_direct(A2, F2)
ue = mms.u_star(I * h, J * h)
log(f"H4 flux sign, N=64 ghost beta=2: Linf error with r={np.abs(U-ue).max():.2e}, with -r={np.abs(U2-ue).max():.2e} (a sign slip would be O(1), not hidden)")
RES["g5_h4"] = (np.abs(U - ue).max(), np.abs(U2 - ue).max())

open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "five_gate_audit_log.txt"), "w").write("\n".join(out) + "\n")
