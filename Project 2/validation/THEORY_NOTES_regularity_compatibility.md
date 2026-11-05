# Theory notes for Math Theory owner (draft; NOT yet in report/main.tex)
Prepared 28 Oct 2026 (Asia/Shanghai). Paste/adapt into report §2 Remark (Regularity) and §2.2 if team agrees.

## 1. Pure-Neumann compatibility (role card: "compatibility condition if pure-Neumann")
Our baseline has Γ_D ≠ ∅ → unique solution, no compatibility needed (report already says this, main.tex ~l.161).
If the team attempts the ⋆⋆⋆ pure-Neumann item (−Δu=f in Ω, ∂_n u=r on ∂Ω):
- Continuous: integrate + divergence theorem ⇒ ∫_Ω f dx + ∫_∂Ω r ds = 0 is necessary; solution unique only up to a constant.
- Discrete (ghost closure on all edges, symmetrised with trapezoidal weights w_k = 1, ½ edges, ¼ corners):
  DA is symmetric singular with kernel span{1}; solvable iff Σ_k w_k F_k = 0 (discrete compatibility). Exact data
  satisfy it only up to O(h²) → project F onto range: F ← F − (Σ w F / Σ w)·1 before solving.
- Fix the constant: either pin one node, add mean-zero Lagrange multiplier (bordered system), or solve with
  minres/cg on the projected RHS and subtract the weighted mean. Compare against MMS u* with zero weighted mean.
- Report: state the compatibility residual Σ w F before projection (should be O(h²)) as a verification number.

## 2. Regularity vs observed rate (role card: "what rate does theory promise on your domain?")
- Domain (0,1)², all corners π/2. Dirichlet–Dirichlet corners at x=0: no singularity (u∈H² ... smooth for compatible data).
- Dirichlet–Neumann corners at (1,0),(1,1), interior angle ω=π/2: corner exponents λ_k=(k+½)π/ω = 1,3,5,…
  These are odd integers, so the singular functions r^λ sin/cos(λθ) are harmonic polynomials (e.g. r sinθ = y):
  NO genuine singularity for smooth data that satisfy the corner compatibility conditions; if data are
  incompatible at the corner, r^λ log r terms appear and u_xxx/u_xxxx blow up there, invalidating the C^4
  assumption used in the max-norm (M-matrix) proof → expect reduced max-norm rate near (1,0),(1,1).
  (Math owner: verify against Grisvard and cite exact theorem/page before putting in report.)
- Our MMS solutions are entire functions and satisfy corner compatibility, so the full rates 1 (first) / 2 (ghost)
  are expected and observed. A falsification experiment (Dimension D top band): choose data incompatible at (1,1)
  (e.g. g≠0 corner mismatch with r) and show the rate degrading → strengthens Dimension B/D.

## 3. Bank run evidence (Testing & Validation, Week-5 duty)
See bank_vs_solver.py / bank_vs_solver_log.txt in this folder: P1,P2,P3(rotated),P5 pass with expected orders;
P2 ghost = round-off (stencil exact on quadratics; good sign check); first-order closure gives order ≈1 except P1 where
u_xx(1,y)=0 kills the leading boundary truncation term (explains observed 2.04 — worth one sentence in report).
P4 (variable diffusion) uses a=1+x via the solver `a=` argument; log shows first≈2.05 and ghost≈2.00 (see bank_vs_solver_log.txt).
