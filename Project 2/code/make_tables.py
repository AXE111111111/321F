"""Rebuild LaTeX tables from CSV results into ../report/tables/."""
from __future__ import annotations
import csv, os


def main():
    HERE = os.path.dirname(os.path.abspath(__file__))
    RES = os.path.join(HERE, "..", "results")
    TAB = os.path.join(HERE, "..", "report", "tables")
    os.makedirs(TAB, exist_ok=True)

    R = list(csv.DictReader(open(os.path.join(RES, "convergence.csv"))))
    X = {(r["bc"], r["closure"], int(r["N"])): r
         for r in csv.DictReader(open(os.path.join(RES, "mms_checks.csv")))}


    def e(v):
        m, ex = f"{float(v):.3e}".split("e")
        return f"${m}\\times10^{{{int(ex)}}}$"


    def e2(v):
        m, ex = f"{float(v):.1e}".split("e")
        return f"${m}\\times10^{{{int(ex)}}}$"


    def rng(lo, hi):
        """min--max with a shared power of ten, e.g. $3.17$--$3.87\\times10^{-4}$."""
        ex = int(f"{float(hi):.2e}".split("e")[1])
        return f"${float(lo)/10**ex:.2f}$--${float(hi)/10**ex:.2f}\\times10^{{{ex}}}$"


    def p(v):
        return "---" if v in ("nan", "", None) else f"{float(v):.2f}"


    for bc in ["Neumann", "Robin"]:
        L = [
            r"\begin{tabular}{@{}llrccccccc@{}}\toprule",
            r"Closure & $N$ & $h$ & $\|e\|_{L^2_h}$ & $p$ & $\|e\|_{\infty}$ & $p$ & $\|e\|_{\infty,\Gamma_B}$ & $\|e\|_{\infty,\mathrm{int}}$ & $\rho$\\\midrule",
        ]
        for cl in ["first", "ghost"]:
            for r in [r for r in R if r["bc"] == bc and r["closure"] == cl]:
                lab = "first-order" if cl == "first" else "ghost-point"
                L.append(
                    f"{lab} & {r['N']} & 1/{r['N']} & {e(r['L2'])} & {p(r['p_L2'])} & "
                    f"{e(r['Linf'])} & {p(r['p_Linf'])} & {e(r['Linf_boundary'])} & {e(r['Linf_interior'])} & "
                    f"{e2(X[(bc, cl, int(r['N']))]['scaled_res'])}\\\\"
                )
            L.append(r"\midrule" if cl == "first" else r"\bottomrule")
        L.append(r"\end{tabular}")
        open(os.path.join(TAB, f"conv_{bc}.tex"), "w").write("\n".join(L))

    L = [
        r"\begin{tabular}{@{}lrccccc@{}}\toprule",
        r"Closure & $N$ & $\max|\tau_B|/h$ & $\max|\tau_I|/h^2$ & $\eta(A)$ & $\eta(DA)$ & scaled residual\\\midrule",
    ]
    for cl in ["first", "ghost"]:
        for N in [16, 64, 256]:
            x = X[("Robin", cl, N)]
            lab = "first-order" if cl == "first" else "ghost-point"
            L.append(
                f"{lab} & {N} & {float(x['tauB'])/float(x['h']):.4f} & "
                f"{float(x['tauI'])/float(x['h'])**2:.4f} & {float(x['etaA']):.4f} & "
                f"{float(x['etaS']):.0f} & {e(x['scaled_res'])}\\\\"
            )
        L.append(r"\midrule" if cl == "first" else r"\bottomrule")
    L.append(r"\end{tabular}")
    open(os.path.join(TAB, "trunc.tex"), "w").write("\n".join(L))

    # Cost table (if present)
    cost_path = os.path.join(RES, "cost_accuracy.csv")
    if os.path.isfile(cost_path):
        C = list(csv.DictReader(open(cost_path)))
        L = [
            r"\begin{tabular}{@{}llrrccc@{}}\toprule",
            r"BC & Closure & $N_\mathrm{min}$ & unknowns & $\|e\|_\infty$ & median time (s) & min--max (s)\\\midrule",
        ]
        for r in C:
            if r["N"] in ("", None, "None"):
                L.append(
                    f"{r['bc']} & {'first-order' if r['closure']=='first' else 'ghost-point'} & "
                    r"-- & -- & -- & -- & --\\"
                )
            else:
                L.append(
                    f"{r['bc']} & {'first-order' if r['closure']=='first' else 'ghost-point'} & "
                    f"{r['N']} & {r['M']} & {e(r['Linf'])} & {e(r['t_spsolve_s'])} & "
                    f"{rng(r['t_min_s'], r['t_max_s'])}\\\\"
                )
        L.append(r"\bottomrule\end{tabular}")
        open(os.path.join(TAB, "cost.tex"), "w").write("\n".join(L))
        print("wrote cost.tex")
        # timing-spread statistics quoted in the report text (macros, so text == CSV)
        def spread(r):
            t = float(r["t_spsolve_s"])
            return max((float(r["t_max_s"]) - t) / t, (t - float(r["t_min_s"])) / t)
        hit = [r for r in C if r["N"] not in ("", None, "None")]
        A = list(csv.DictReader(open(os.path.join(RES, "cost_accuracy_all.csv"))))
        w = max(A, key=lambda r: float(r["t_max_s"]) / float(r["t_spsolve_s"]))
        cl = "first-order" if w["closure"] == "first" else "ghost"
        M = [
            "% written by make_tables.py from results/cost_accuracy*.csv",
            f"\\newcommand{{\\CostSpreadMatched}}{{{100 * max(spread(r) for r in hit):.0f}}}",
            f"\\newcommand{{\\CostWorstRatio}}{{{float(w['t_max_s']) / float(w['t_spsolve_s']):.2f}}}",
            f"\\newcommand{{\\CostWorstCase}}{{{w['bc']}, {cl}, $N={w['N']}$}}",
        ]
        open(os.path.join(TAB, "cost_stats.tex"), "w").write("\n".join(M) + "\n")

    print("tables written to", TAB)


if __name__ == "__main__":
    main()
