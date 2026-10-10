"""Read the saved CSVs and write the LaTeX tables/macros used by the report.

Read-only with respect to results: it never simulates. Usage:
    python code/report_tables.py --results results --out <dir>
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def f(x: float, d: int = 5) -> str:
    return f"{x:.{d}f}"


def sci(x: float, d: int = 2) -> str:
    m, e = f"{x:.{d}e}".split("e")
    return rf"{m}\times10^{{{int(e)}}}"


def pm(est: float, se: float) -> str:
    """Estimate +/- SE with the SE to two significant figures."""
    d = max(0, -math.floor(math.log10(se)) + 1) if se > 0 else 4
    return rf"${est:.{d}f}\pm{se:.{d}f}$"


def by_se(est: float, se: float) -> str:
    """Estimate rounded to the decimal place of the second significant figure of its SE."""
    d = max(0, -math.floor(math.log10(se)) + 1) if se > 0 else 4
    return f"{est:.{d}f}"


def order_cols(errs: list[float]) -> list[tuple[str, str]]:
    out = [("---", "---")]
    for a, b in zip(errs, errs[1:]):  # successive grids differ by a factor 2 in h
        r = a / b
        out.append((f(r, 3), f(math.log2(r), 3)))
    return out


def table(caption: str, label: str, header: str, body: list[str], spec: str, size: str = r"\small") -> str:
    return "\n".join([r"\begin{table}[H]", r"\centering", size,
                      rf"\caption{{{caption}}}", rf"\label{{{label}}}",
                      rf"\begin{{tabular}}{{{spec}}}", r"\toprule", header + r" \\", r"\midrule",
                      *[b + r" \\" for b in body], r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=Path(__file__).resolve().parents[1] / "results")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    R, out = a.results, a.out
    out.mkdir(parents=True, exist_ok=True)
    macros: dict[str, str] = {}

    def mac(name: str, value: str) -> None:
        macros[name] = value

    # ---- GBM strong: h / error / ratio / order -------------------------------------------
    gs = rows(R / "gbm_strong.csv")
    em = [r for r in gs if r["method"] == "EM"]
    mi = [r for r in gs if r["method"] == "Milstein"]
    oe, om = order_cols([float(r["estimate"]) for r in em]), order_cols([float(r["estimate"]) for r in mi])
    body = []
    for r1, r2, (ra, oa), (rb, ob) in zip(em, mi, oe, om):
        star1 = r"$^\ast$" if r1["fit_included"] == "1" else ""
        body.append(f"{r1['N']} & $1/{r1['N']}$ & {by_se(float(r1['estimate']), float(r1['se']))}{star1} & {ra} & {oa} & "
                    f"{by_se(float(r2['estimate']), float(r2['se']))}{star1} & {rb} & {ob}")
    (out / "tab_gbm_strong_order.tex").write_text(table(
        r"GBM strong error $\widehat e_s(h)=\overline{|S_N-S_T|}$ against the exact coupled path "
        r"($M=2{,}000{,}000$). Ratio $=\widehat e_s(2h)/\widehat e_s(h)$; local order $=\log_2(\text{Ratio})$. "
        r"$^\ast$: fit range $N\ge64$ (team rule). Each error is rounded to the second significant figure of its own Monte Carlo SE "
        r"(largest SE " + f(max(float(r["se"]) for r in gs), 4) + r", at EM $N=2$); ratios and orders use unrounded values. Source: \texttt{results/gbm\_strong.csv}.",
        "tab:gbm-strong-order",
        r"$N$ & $h$ & EM error & Ratio & Order & Milstein error & Ratio & Order",
        body, "@{}rrrrrrrr@{}"))
    mac("emLocalOrderFine", oe[-1][1]); mac("milLocalOrderFine", om[-1][1])
    mac("emLocalOrderCoarse", oe[1][1]); mac("milLocalOrderCoarse", om[1][1])

    # ---- GBM weak --------------------------------------------------------------------------
    gw = rows(R / "gbm_weak.csv")
    wem = [r for r in gw if r["method"] == "EM"]
    wmi = [r for r in gw if r["method"] == "Milstein"]
    an = [abs(float(r["analytic_bias"])) for r in wem]
    oan = order_cols(an)
    body = []
    for r1, r2, (_, o) in zip(wem, wmi, oan):
        m1 = "" if r1["fit_included"] == "1" else r"$^\circ$"
        body.append(f"{r1['N']} & {pm(float(r1['estimate']), float(r1['se']))}{m1} & "
                    f"{pm(float(r2['estimate']), float(r2['se']))} & ${f(float(r1['analytic_bias']),6)}$ & {o}")
    (out / "tab_gbm_weak_order.tex").write_text(table(
        r"GBM signed mean bias $\widehat\delta(h)\pm1$ SE (paired with the exact path) and the analytic bias "
        r"$\delta(h)$ of~\eqref{eq:analytic-bias}; the last column is the local order $\log_2(\delta(2h)/\delta(h))$ of the analytic bias. "
        r"$^\circ$: EM grid failing the team's resolution rule (excluded from the fit; its interval may still exclude zero). Source: \texttt{results/gbm\_weak.csv}.",
        "tab:gbm-weak-order",
        r"$N$ & EM $\widehat\delta$ & Milstein $\widehat\delta$ & Analytic $\delta$ & Order of $\delta$",
        body, "@{}rrrrr@{}"))
    mac("analyticBiasHalf", f(float(wem[0]["analytic_bias"]), 4))
    mac("analyticInCI", str(sum(int(r["analytic_in_ci"]) for r in gw)))

    # ---- Non-affine strong -------------------------------------------------------------------
    ns = rows(R / "nonaffine_strong.csv")
    by = {}
    for r in ns:
        by.setdefault(r["reference_N"], []).append(r)
    refs = sorted(by, key=int)
    o1 = order_cols([float(r["estimate"]) for r in by[refs[0]]])
    o2 = order_cols([float(r["estimate"]) for r in by[refs[1]]])
    body = []
    for r1, r2, (ra, oa), (rb, ob) in zip(by[refs[0]], by[refs[1]], o1, o2):
        body.append(f"{r1['N']} & {f(float(r1['estimate']))} & {ra} & {oa} & {f(float(r2['estimate']))} & {rb} & {ob}")
    (out / "tab_nonaffine_strong_order.tex").write_text(table(
        r"Non-affine coupled price discrepancy $\overline{|S^{(h)}_T-S^{(h_{\rm ref})}_T|}$ ($M=50{,}000$) against the two EM references, "
        r"with successive ratios and local empirical orders. Source: \texttt{results/nonaffine\_strong.csv}.",
        "tab:nonaff-strong-order",
        rf"$N$ & vs.\ {refs[0]} & Ratio & Order & vs.\ {refs[1]} & Ratio & Order",
        body, "@{}rrrrrrr@{}"))

    # ---- Slope summary ----------------------------------------------------------------------
    def slope_of(rs):
        r = rs[0]
        fitted = [x["N"] for x in rs if x["fit_included"] == "1"]
        rng = f"{fitted[0]}--{fitted[-1]}" if fitted else "---"
        if not r["fitted_slope"]:
            return r["fit_point_count"], "not reported", rng
        return r["fit_point_count"], rf"${f(float(r['fitted_slope']),4)}\pm{f(float(r['fitted_slope_se']),4)}$", rng

    nw = rows(R / "nonaffine_weak.csv")
    ext = rows(R / "extensions" / "ext_nonaffine_coarse.csv")
    pay = [r for r in ext if r["record_type"] == "payoff_difference"]
    lines = []
    for study, meth, theory, rs in (
            ("GBM strong", "EM", r"$1/2$", em), ("GBM strong", "Milstein", "$1$", mi),
            ("GBM weak ($\\phi(s)=s$)", "EM", "$1$", wem), ("GBM weak ($\\phi(s)=s$)", "Milstein", "$1$", wmi),
            ("Non-aff.\\ strong", f"EM, ref.\\ {refs[0]}", r"($1/2$)", by[refs[0]]),
            ("Non-aff.\\ strong", f"EM, ref.\\ {refs[1]}", r"($1/2$)", by[refs[1]]),
            ("Non-aff.\\ weak, main", "EM, ref.\\ 4096", r"($1$)", [r for r in nw if r["reference_N"] == "4096"]),
            ("Non-aff.\\ weak, main", "EM, ref.\\ 8192", r"($1$)", [r for r in nw if r["reference_N"] == "8192"]),
            ("Non-aff.\\ weak, E2", "EM, ref.\\ 512", r"($1$)", [r for r in pay if r["reference_N"] == "512"]),
            ("Non-aff.\\ weak, E2", "EM, ref.\\ 1024", r"($1$)", [r for r in pay if r["reference_N"] == "1024"])):
        n, s, rg = slope_of(rs)
        lines.append(f"{study} & {meth} & {theory} & {n} & {s} & {rg}")
    (out / "tab_slopes.tex").write_text(table(
        r"Summary of fitted log--log slopes. $\pm$ is one paired delete-one-batch jackknife SE conditional on the fitted grids "
        r"(not a calibrated interval for the true order). Theory in parentheses is expected/empirical rather than proved for the price statistic. "
        r"Sources: \texttt{results/*.csv} and \texttt{results/extensions/ext\_nonaffine\_coarse.csv}.",
        "tab:slopes", r"Study & Method & Theory & Points & Slope $\pm$ SE & Fitted $N$", lines, "@{}lllrll@{}"))
    for r in pay:
        if r["reference_N"] == "512":
            mac("eTwoSlopeA", f(float(r["fitted_slope"]), 3)); mac("eTwoSlopeSEA", f(float(r["fitted_slope_se"]), 3))
        else:
            mac("eTwoSlopeB", f(float(r["fitted_slope"]), 3)); mac("eTwoSlopeSEB", f(float(r["fitted_slope_se"]), 3))

    # ---- E1 uncoupled -----------------------------------------------------------------------
    e1 = rows(R / "extensions" / "ext_uncoupled_strong.csv")
    key = {(r["coupling"], r["method"], r["N"]): r for r in e1}
    Ns = sorted({r["N"] for r in e1}, key=int)
    body = []
    for n in Ns:
        c = [key[(cp, m, n)] for cp in ("coupled", "uncoupled") for m in ("EM", "Milstein")]
        body.append(f"{n} & " + " & ".join(pm(float(x['estimate']), float(x['se'])) for x in c))
    sl = [f(float(key[(cp, m, Ns[0])]["ols_slope_all_grids"]), 4) for cp in ("coupled", "uncoupled") for m in ("EM", "Milstein")]
    body.append(r"\midrule OLS slope & " + " & ".join(sl))
    (out / "tab_e1_uncoupled.tex").write_text(table(
        r"Falsification experiment E1 ($M=200{,}000$, independent seed stream): strong error $\pm1$ SE when the numerical path is "
        r"coupled to the exact path (same increments) and when it is driven by an \emph{independent} Brownian path. "
        r"Source: \texttt{results/extensions/ext\_uncoupled\_strong.csv}.",
        "tab:e1", r"$N$ & Coupled EM & Coupled Milstein & Uncoupled EM & Uncoupled Milstein", body, "@{}rllll@{}"))
    mac("eOneUncoupled", f(float(key[("uncoupled", "EM", Ns[-1])]["estimate"]), 2))
    mac("eOneUncSlopeEM", sl[2]); mac("eOneUncSlopeMil", sl[3])
    mac("eOneCoupledSlopeEM", sl[0]); mac("eOneCoupledSlopeMil", sl[1])

    # ---- E2 coarse non-affine -----------------------------------------------------------------
    orc = {r["N"]: r for r in ext if r["record_type"] == "mean_oracle"}
    pk = {(r["reference_N"], r["N"]): r for r in pay}
    gap = [r for r in ext if r["record_type"] == "reference_payoff_gap"][0]
    body = []
    for n in sorted({r["N"] for r in pay}, key=int) + ["512"]:
        o = orc[n]
        if n == "512":  # no payoff difference against itself; only the paired mean difference vs 1024
            body.append(r"\midrule 512 & --- & " + pm(float(gap["estimate"]), float(gap["se"])) + r"$^\dagger$ & "
                        + pm(float(o['paired_mean_diff_vs_finest']), float(o['paired_mean_diff_se']))
                        + " & " + pm(float(o['estimate']), float(o['se'])))
            continue
        cells = []
        for ref in ("512", "1024"):
            r = pk[(ref, n)]
            mark = "" if r["fit_included"] == "1" else r"$^\circ$"
            cells.append(pm(float(r['estimate']), float(r['se'])) + mark)
        body.append(f"{n} & " + " & ".join(cells) +
                    " & " + pm(float(o['paired_mean_diff_vs_finest']), float(o['paired_mean_diff_se']))
                    + " & " + pm(float(o['estimate']), float(o['se'])))
    (out / "tab_e2_coarse.tex").write_text(table(
        r"Experiment E2 (non-affine, $M=400{,}000$, 40 batches, coarse grids $N=2$--$64$, EM references 512 and 1024). "
        r"Columns 2--3: paired call-payoff differences $\overline{\phi(S^{(h)}_T)-\phi(S^{\rm ref}_T)}\pm1$ SE; "
        r"$^\circ$ = not resolved by the team's rule. Column 4: paired \emph{mean} difference $\overline{S^{(h)}_T-S^{(1024)}_T}$, "
        r"whose exact expectation is $0$ by~\eqref{eq:logem-mean}. $^\dagger$: reference check $\overline{\phi(S^{(512)}_T)-\phi(S^{(1024)}_T)}$. Column 5: $\overline{S^{(h)}_T}-S_0e^{\mu T}$. "
        r"Source: \texttt{results/extensions/ext\_nonaffine\_coarse.csv}.",
        "tab:e2", r"$N$ & Payoff diff.\ (ref.\ 512) & Payoff diff.\ (ref.\ 1024) & Mean diff.\ (ref.\ 1024) & $\bar S-S_0e^{\mu T}$",
        body, "@{}rllll@{}", r"\footnotesize"))
    mac("eTwoPayoffTwo", f(float(pk[("1024", "2")]["estimate"]), 4)); mac("eTwoPayoffTwoSE", f(float(pk[("1024", "2")]["se"]), 4))
    mac("eTwoMeanTwo", f(float(orc["2"]["paired_mean_diff_vs_finest"]), 4)); mac("eTwoMeanTwoSE", f(float(orc["2"]["paired_mean_diff_se"]), 4))
    mac("eTwoOracleAll", str(sum(int(r["oracle_in_ci"]) for r in orc.values())))
    mac("eTwoOracleN", str(len(orc)))
    mac("eTwoPairedAll", str(sum(1 for r in orc.values() if r["paired_zero_in_ci"] == "1")))
    mac("eTwoPairedN", str(sum(1 for r in orc.values() if r["paired_zero_in_ci"] != "")))
    cr = {r["reference_N"]: r for r in ext if r["record_type"] == "common_range_fit"}
    mac("eTwoCommonRange", cr["512"]["N"].replace("-", ","))
    for ref, tag in (("512", "A"), ("1024", "B")):
        mac(f"eTwoCommonSlope{tag}", f(float(cr[ref]["fitted_slope"]), 3))
        mac(f"eTwoCommonSlopeSE{tag}", f(float(cr[ref]["fitted_slope_se"]), 3))
    n32 = pk[("1024", "32")]
    mac("eTwoGateThirtyTwo", f(1.96 * float(n32["se"]) / abs(float(n32["estimate"])), 3))
    mac("eTwoRefGap", f(float(gap["estimate"]), 5)); mac("eTwoRefGapSE", f(float(gap["se"]), 5))
    o5 = orc["512"]
    mac("eTwoMeanFiveTwelve", f(float(o5["paired_mean_diff_vs_finest"]), 5))
    mac("eTwoMeanFiveTwelveSE", f(float(o5["paired_mean_diff_se"]), 5))
    mac("eTwoMinPairedSE", f(min(float(r["paired_mean_diff_se"]) for r in orc.values() if r["paired_mean_diff_se"] and float(r["paired_mean_diff_se"]) > 0), 5))
    fit1024 = [r for r in pay if r["reference_N"] == "1024" and r["fit_included"] == "1"]
    smallest = min(fit1024, key=lambda r: abs(float(r["estimate"])))
    mac("eTwoSmallestFit", f(float(smallest["estimate"]), 4)); mac("eTwoSmallestFitN", smallest["N"])
    mac("eTwoCallRef", f(float(orc["1024"]["payoff_mean"]), 3)); mac("eTwoCallRefSE", f(float(orc["1024"]["payoff_se"]), 3))


    # ---- GBM moments and quantiles -------------------------------------------------------------
    mo = {r["method"]: r for r in rows(R / "gbm_moments.csv")}
    body = [f"{m} & {pm(float(mo[m]['sample_mean']), float(mo[m]['se_mean']))} & "
            f"{pm(float(mo[m]['sample_variance']), float(mo[m]['se_variance']))} & "
            f"{'yes' if float(mo[m]['ci_lower_mean']) <= float(mo[m]['exact_mean']) <= float(mo[m]['ci_upper_mean']) else 'no'} / "
            f"{'yes' if float(mo[m]['ci_lower_variance']) <= float(mo[m]['exact_variance']) <= float(mo[m]['ci_upper_variance']) else 'no'}"
            for m in ("EM", "Milstein")]
    # Analytic moments of the *discrete* schemes (one-step factor has mean 1+mu h and second moment
    # (1+mu h)^2 + sigma^2 h [EM] or + sigma^2 h + sigma^4 h^2 / 2 [Milstein]); computed here, not simulated.
    r0 = mo["EM"]; s0_, mu_, sg_, n_ = float(r0["s0"]), float(r0["mu"]), float(r0["sigma"]), int(r0["N"])
    h_ = float(r0["T"]) / n_
    a_mean = s0_ * (1 + mu_ * h_) ** n_
    a_var = {"EM": s0_**2 * (((1 + mu_*h_)**2 + sg_**2*h_)**n_ - (1 + mu_*h_)**(2*n_)),
             "Milstein": s0_**2 * (((1 + mu_*h_)**2 + sg_**2*h_ + 0.5*sg_**4*h_**2)**n_ - (1 + mu_*h_)**(2*n_))}
    for m in ("EM", "Milstein"):
        z = (float(mo[m]["sample_variance"]) - a_var[m]) / float(mo[m]["se_variance"])
        body.append(f"{m}, analytic (scheme) & {f(a_mean, 3)} & {f(a_var[m], 3)} & $z_{{\\rm var}}={f(z, 2)}$")
        mac(f"anaVar{m}", f(a_var[m], 3))
    mac("anaMean", f(a_mean, 3))
    body.append(f"Exact (SDE) & {f(float(mo['EM']['exact_mean']), 3)} & {f(float(mo['EM']['exact_variance']), 3)} & ---")
    (out / "tab_moments.tex").write_text(table(
        r"Terminal moments at $N=256$, $M=" + f"{int(mo['EM']['M']):,}".replace(",", "{,}") +
        r"$ ($\pm1$ SE; variance SE from the sample fourth central moment). Last column: does the approximate 95\% interval contain the exact mean / variance? "
        r"Rows `analytic (scheme)': exact mean and variance of the discrete scheme $S^{(h)}_N$ at $N=256$, computed in closed form by \texttt{report\_tables.py}; "
        r"$z_{\rm var}$ = (sample $-$ analytic)/SE. Source: \texttt{results/gbm\_moments.csv}.", "tab:moments",
        r"Method & Mean & Variance & Exact in CI", body, "@{}llll@{}"))
    mac("emMean", f(float(mo["EM"]["sample_mean"]))); mac("exactMean", f(float(mo["EM"]["exact_mean"])))
    mac("exactVar", f(float(mo["EM"]["exact_variance"])))
    dq = [r for r in rows(R / "gbm_distribution.csv") if r["record_type"] == "quantile"]
    probs = sorted({r["probability"] for r in dq}, key=float)
    qk = {(r["method"], r["probability"]): r for r in dq}
    body, inside = [], 0
    for pr in probs:
        e, m_ = qk[("EM", pr)], qk[("Milstein", pr)]
        for r in (e, m_):
            inside += int(float(r["ci_lower_quantile"]) <= float(r["exact_quantile"]) <= float(r["ci_upper_quantile"]))
        body.append(f"{float(pr):.2f} & {f(float(e['exact_quantile']),4)} & ${f(float(e['estimate']),4)}\\pm{f(float(e['se_quantile']),4)}$ & "
                    f"${f(float(m_['estimate']),4)}\\pm{f(float(m_['se_quantile']),4)}$")
    (out / "tab_quantiles.tex").write_text(table(
        r"Numerical terminal quantiles at $N=256$ against~\eqref{eq:quantile}; $\pm1$ SE is an exact-law-based approximation (asymptotic quantile SE "
        r"with the exact lognormal density at the exact quantile in place of the unknown density of the numerical law), so the intervals are approximate, not calibrated. "
        rf"{inside} of {2*len(probs)} such approximate 95\% intervals contain the exact quantile. Source: \texttt{{results/gbm\_distribution.csv}}.",
        "tab:quantiles", r"$p$ & Exact & EM & Milstein", body, "@{}cccc@{}"))
    mac("quantInside", str(inside)); mac("quantTotal", str(2 * len(probs)))

    # ---- Main non-affine payoff differences ------------------------------------------------------
    body = []
    zero_in = 0
    for r in sorted(nw, key=lambda r: (int(r["reference_N"]), int(r["N"]))):
        lo, hi = float(r["ci_lower"]), float(r["ci_upper"])
        zero_in += int(lo <= 0 <= hi)
        body.append(f"{r['reference_N']} & {r['N']} & {f(float(r['estimate']),6)} & ${f(lo,6)}$ & {f(hi,6)}")
    (out / "tab_payoff_main.tex").write_text(table(
        rf"Main run ($M=50{{,}}000$): paired call-payoff differences and approximate 95\% intervals; {zero_in} of {len(nw)} intervals contain zero. "
        r"Source: \texttt{results/nonaffine\_weak.csv}.", "tab:payoff",
        r"Reference $N$ & Coarse $N$ & Signed difference & Lower & Upper", body, "@{}rrrrr@{}", r"\footnotesize"))
    mac("payoffZeroIn", str(zero_in)); mac("payoffTotal", str(len(nw)))

    # ---- Increment diagnostics and path validation ----------------------------------------------
    inc = rows(R / "nonaffine_increment_diagnostics.csv")
    body = [f"{r['N']} & {f(float(r['sample_var_W1'])/float(r['target_var_W1']),6)} & "
            f"{f(float(r['sample_var_W2'])/float(r['target_var_W2']),6)} & "
            f"${f(float(r['sample_correlation']),6)}\\pm{f(float(r['se_correlation']),6)}$" for r in inc]
    (out / "tab_increments.tex").write_text(table(
        r"Increment variances normalised by their target $h$ (targets $1,1$) and sample correlation (target $\rho=-0.7$, $\pm1$ asymptotic SE). "
        r"Source: \texttt{results/nonaffine\_increment\_diagnostics.csv}.", "tab:increments",
        r"$N$ & $s^2_{W^{(1)}}/h$ & $s^2_{W^{(2)}}/h$ & Correlation", body, "@{}rrrr@{}"))
    pv = rows(R / "nonaffine_path_validation.csv")
    body = [f"{r['N']} & {r['finite_X']} & {r['finite_Y']} & {r['finite_S']} & {r['nonpositive_S']}" for r in pv]
    (out / "tab_pathval.tex").write_text(table(
        r"Terminal-state audit on every grid (each with $M=50{,}000$ paths), including both references. Source: \texttt{results/nonaffine\_path\_validation.csv}.",
        "tab:pathval", r"$N$ & Finite $X$ & Finite $Y$ & Finite $S$ & $S\le0$", body, "@{}rrrrr@{}"))

    # ---- Reference comparison ------------------------------------------------------------------
    rc = [r for r in rows(R / "nonaffine_reference_comparison.csv") if r["measure"] == "absolute_price_difference"][0]
    mac("refGap", f(float(rc["estimate"]), 5)); mac("refGapSE", f(float(rc["se"]), 5))
    mac("refFinest", f(float(rc["finest_coarse_to_fine_reference_mean_abs"]), 5))
    mac("refRatio", f(100 * float(rc["reference_gap_to_coarse_discrepancy_ratio"]), 2))
    mac("refChange", f(float(rc["observed_discrepancy_change_between_references"]), 5))
    mac("refSlopeChange", f(float(rc["observed_strong_slope_change_between_references"]), 5))

    # ---- Cost at matched accuracy (target: CI upper endpoint <= 0.11) ----------------------------
    target = 0.11
    body = []
    for name, rs in (("EM", em), ("Milstein", mi)):
        r = min((x for x in rs if float(x["ci_upper"]) <= target), key=lambda x: int(x["N"]))
        cost = int(r["M"]) * int(r["N"])
        body.append(f"{name} & {r['N']} & {f(float(r['estimate']))} & {f(float(r['ci_upper']))} & "
                    + f"{cost:,}".replace(",", "{,}") + " & " + f"{int(r['N']):,}")
        mac(f"cost{name}N", r["N"])
    (out / "tab_cost.tex").write_text(table(
        r"Matched accuracy on the tested grids: smallest $N$ whose strong-error 95\% upper endpoint is $\le 0.11$. "
        r"Work $C=MN$ counts scalar trajectory-step updates for $M=2{,}000{,}000$; last column is the per-path step count. "
        r"Excluded: random-number generation, aggregation, exact evaluation, statistics, plotting, per-step arithmetic differences. "
        r"Source: \texttt{results/gbm\_strong.csv}.", "tab:cost",
        r"Method & $N$ & Strong error & Upper 95\% & $C=MN$ & Steps/path", body, "@{}lrrrrr@{}"))

    # ---- E3 wall clock (machine dependent) ------------------------------------------------------
    e3_all = rows(R / "extensions" / "ext_wallclock.csv")
    e3 = {(r["method"], r["N"]): r for r in e3_all if r["array_layout"].startswith("column")}
    e3c = {(r["method"], r["N"]): r for r in e3_all if r["array_layout"].startswith("row")}
    mac("wcCEM", f(1e3 * float(e3c[("EM", "512")]["median_seconds"]), 2))
    mac("wcCMil", f(1e3 * float(e3c[("Milstein", "8")]["median_seconds"]), 3))
    mac("wcCRatio", f(float(e3c[("EM", "512")]["median_seconds"]) / float(e3c[("Milstein", "8")]["median_seconds"]), 0))
    mac("wcCSlowEM", f(float(e3c[("EM", "512")]["median_seconds"]) / float(e3[("EM", "512")]["median_seconds"]), 1))
    mac("wcCSlowMil", f(float(e3c[("Milstein", "8")]["median_seconds"]) / float(e3[("Milstein", "8")]["median_seconds"]), 1))
    med = lambda m, n: float(e3[(m, n)]["median_seconds"])
    mac("wcEM", f(1e3 * med("EM", "512"), 2))
    mac("wcMil", f(1e3 * med("Milstein", "8"), 3))
    mac("wcRatio", f(med("EM", "512") / med("Milstein", "8"), 0))
    mac("wcPaths", f"{int(e3[('EM', '8')]['paths']):,}".replace(",", "{,}"))
    mac("wcRepeats", e3[("EM", "8")]["repeats"])
    for m, tag in (("EM", "EM"), ("Milstein", "Mil")):
        for n, nt in (("8", "Eight"), ("512", "FiveTwelve")):
            mac(f"wc{tag}{nt}", f(1e3 * med(m, n), 3))
            mac(f"wcPerStep{tag}{nt}", sci(float(e3[(m, n)]["seconds_per_path_step"])))
    mac("wcStepRatioEight", f(med("Milstein", "8") / med("EM", "8"), 2))
    mac("wcStepRatioFiveTwelve", f(med("Milstein", "512") / med("EM", "512"), 2))
    mac("wcPython", e3[("EM", "8")]["python"]); mac("wcNumpy", e3[("EM", "8")]["numpy"])
    mac("wcMachine", e3[("EM", "8")]["machine"].replace("_", r"\_"))

    # ---- E1b misaligned aggregation ---------------------------------------------------------------
    mis = rows(R / "extensions" / "ext_misaligned.csv")
    mk = {(r["model"], r["aggregation"], r["N"]): r for r in mis}
    mNs = sorted({r["N"] for r in mis}, key=int)
    cols = [(mo_, ag) for mo_ in ("GBM_EM", "GBM_Milstein", "nonaffine_EM") for ag in ("correct", "misaligned")]
    body = [f"{n} & " + " & ".join(pm(float(mk[(mo_, ag, n)]["estimate"]), float(mk[(mo_, ag, n)]["se"])) for mo_, ag in cols)
            for n in mNs]
    body.append(r"\midrule OLS slope & " + " & ".join(f(float(mk[(mo_, ag, mNs[0])]["ols_slope_all_grids"]), 3) for mo_, ag in cols))
    (out / "tab_e1b_misaligned.tex").write_text(table(
        r"Experiment E1b ($M=" + f"{int(mk[cols[0] + (mNs[0],)]['M']):,}".replace(",", "{,}") + r"$, seed child 5): strong error ($\pm1$ SE) with the correct "
        r"block-sum aggregation (`ok') and with a deliberately strided aggregation (`bad': coarse step $j$ sums fine steps $j,j+N,j+2N,\dots$). "
        r"GBM: error against the exact path. Non-affine: discrepancy against an EM reference with $N=" + mk[("nonaffine_EM", "correct", mNs[0])]["reference"].split("=")[1]
        + r"$ built from the correctly ordered fine increments. For GBM the fine grid is $N=512$, so `ok' and `bad' coincide there. Source: \texttt{results/extensions/ext\_misaligned.csv}.",
        "tab:e1b", r"$N$ & GBM EM ok & GBM EM bad & GBM Mil.\ ok & GBM Mil.\ bad & Non-aff.\ ok & Non-aff.\ bad",
        body, "@{}rllllll@{}", r"\scriptsize\setlength{\tabcolsep}{2pt}"))
    for mo_, tag in (("GBM_EM", "GbmEM"), ("GBM_Milstein", "GbmMil"), ("nonaffine_EM", "Nonaff")):
        for ag, at in (("correct", "Ok"), ("misaligned", "Bad")):
            mac(f"eOneB{tag}{at}Slope", f(float(mk[(mo_, ag, mNs[0])]["ols_slope_all_grids"]), 3))
    mac("eOneBNonaffOkErr", f(float(mk[("nonaffine_EM", "correct", mNs[-1])]["estimate"]), 3))
    mac("eOneBNonaffBadErr", f(float(mk[("nonaffine_EM", "misaligned", mNs[-1])]["estimate"]), 3))
    mac("eOneBNonaffBadErrCoarse", f(float(mk[("nonaffine_EM", "misaligned", mNs[0])]["estimate"]), 3))
    mac("eOneBPaths", f"{int(mk[cols[0] + (mNs[0],)]['M']):,}".replace(",", "{,}"))
    (out / "macros.tex").write_text("".join(rf"\newcommand{{\{k}}}{{{v}}}" + "\n" for k, v in macros.items()))
    print(f"Wrote tables and {len(macros)} macros to {out}")


if __name__ == "__main__":
    main()
