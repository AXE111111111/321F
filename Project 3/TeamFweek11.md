### TeamF

### 1. Implementation and measured acceptance result

The problem is u_tt = u_xx on 0 < x < 1, with u(0,t) = u(1,t) = 0, u(x,0) = sin(pi x), and u_t(x,0) = 0. The continuous exact solution is sin(pi x) cos(pi t). TASKS 1–2 are completed in the supplied hyperbolic scaffold; the optional advection TASKS 3–4 are not part of this wave result.

The solver uses n = 63 interior nodes, h = 1/64, dt = 0.006, T = 0.05, and CFL = 0.384. Zero boundary values enter both end stencils; the returned arrays contain only the interior nodes. The unchanged official acceptance driver prints PASS. The results below are from actual Python executions, not the example output in the tutorial.

| Start-up        | Computed u(0.5,0.05) | Full-field maximum error vs continuous exact | Discrete L2 error vs continuous exact |
| --------------- | -------------------- | -------------------------------------------- | ------------------------------------- |
| Taylor (chosen) | 0.987690493990       | 2.1533944e-6                                 | 1.5226798e-6                          |
| Copy            | 0.989164643302       | 1.4763027e-3                                 | 1.0439037e-3                          |

The exact midpoint is 0.987688340595. Thus Taylor reproduces 0.9877 at four decimal places (also 0.988 to three significant figures); copy gives 0.9892. Both runs return t = 0.05 exactly. The L2 norm is sqrt(h sum_j |error_j|^2).

### 2. Start-up and the shortened final step

Let L_h U_j = (U_{j-1} - 2U_j + U_{j+1})/h^2, including zero endpoint values. The uniform-step update is

U^{m+1} = 2U^m - U^{m-1} + dt^2 L_h U^m.

Taylor expansion using u_t(0) = 0 and u_tt = u_xx gives U^1 = U^0 + (k^2/2)L_h U^0, where k = min(dt,T) is the first executed step. Copy instead sets U^1 = U^0 and omits the acceleration term. For this zero-velocity semidiscrete wave, Taylor's first-level error is O(k^4), whereas copy's is O(k^2); these are first-level errors, not global orders. At fixed h, the latter becomes a global O(dt) error: for L_h phi = -mu phi, theta = 2 asin(sqrt(mu) dt/2), Taylor's amplitude is cos(m theta), while copy adds tan(theta/2) sin(m theta), an O(dt) term at fixed physical time. Taylor's global temporal error is O(dt^2).

Here eight full steps reach 0.048, leaving s = 0.002. The final update is

U(T) = U^m + (s/dt)(U^m-U^{m-1}) + s(dt+s)L_h U^m/2.

Indeed, (U^m-U^{m-1})/dt = u_t(t_m) - dt L_h U^m/2 + O(dt^2). Substitute the reconstructed velocity into a Taylor step of length s. The endpoint defect is O(s dt^2 + s^3), so a single shortened endpoint retains second-order global accuracy. Merely changing the Courant number in the uniform recurrence is wrong: the modal test detects a 1.7312662e-3 difference from the consistent endpoint amplitude.

### 3. Separated temporal refinement

To avoid a spatial-error plateau, keep n = 63 and T = 0.05 fixed and compare against the same-grid semidiscrete solution

u_h(x_j,t) = sin(pi x_j) cos(sqrt(mu)t), mu = 4 sin^2(pi h/2)/h^2.

Use E = max_j |U_j-u_h(x_j,T)| and p = log2(E(dt)/E(dt/2)). These time steps divide T exactly, isolating the uniform-step temporal order; the acceptance run above separately tests the unequal endpoint.

| dt          | CFL   | Taylor E     | Taylor p | Copy E       | Copy p  |
| ----------- | ----- | ------------ | -------- | ------------ | ------- |
| 0.0125      | 0.8   | 1.5786141e-6 | —        | 3.0701785e-3 | —       |
| 0.00625     | 0.4   | 3.9459274e-7 | 2.00022  | 1.5351885e-3 | 0.99991 |
| 0.003125    | 0.2   | 9.8644388e-8 | 2.00006  | 7.6765600e-4 | 0.99988 |
| 0.0015625   | 0.1   | 2.4660858e-8 | 2.00001  | 3.8384805e-4 | 0.99992 |
| 0.00078125  | 0.05  | 6.1651994e-9 | 2.00000  | 1.9192961e-4 | 0.99996 |
| 0.000390625 | 0.025 | 1.5413351e-9 | 1.99997  | 9.5966275e-5 | 0.99998 |

This confirms second-order Taylor versus first-order copy, not just two different midpoint values. Against the continuous wave, Taylor errors instead approach the analytically computed fixed-grid spatial floor, max|u_h-u| = 2.4668695e-6 for these parameters. A separate dt = 0.006/2^k series, with shortened final steps, also retains the expected temporal orders (data supplied).

### 4. Checks and conclusion

Full-field/grid checks supplement the midpoint; a hand-computable three-node stencil checks zero boundaries, and an independently evaluated sine-mode recurrence checks the noninteger endpoint for both start-ups. Additional checks cover zero/small final times, one shortened first step, invalid arguments and CFL rejection. Uniform-step discrete-energy conservation is tested separately from shortened-endpoint runs; no conservation claim is made across that unequal step. The reproducibility package includes code, raw CSV data, logs, a figure and a claim audit.

Conclusion: use the PDE-based Taylor start-up. Copy can look plausible and remain bounded but is only first-order in time for this zero-initial-velocity problem. Stability alone does not establish the correct start-up or temporal accuracy.

Sources: Tutorial 11, pp. 5 and 7; hyperbolic_skeleton.py, TASKS 1–2 and executable acceptance driver; cn_wave_worksheet.pdf, wave start-up and energy sections.

AI disclosure: AI assistance was used for code, derivations, tests and drafting. All numerical values are from actual Python runs; independent analytical, hand-stencil and separated-refinement checks are documented in the supplied audit.