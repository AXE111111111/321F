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


    def p(v):
        return "---" if v in ("nan", "", None) else f"{float(v):.2f}"


    for bc in ["Neumann", "Robin"]:
        L = [
            r"\begin{tabular}{@{}llrcccccc@{}}\toprule",
            r"Closure & $N$ & $h$ & $\|e\|_{L^2_h}$ & $p$ & $\|e\|_{\infty}$ & $p$ & $\|e\|_{\infty,\Gamma_B}$ & $\|e\|_{\infty,\mathrm{int}}$\\\midrule",
        ]
        for cl in ["first", "ghost"]:
            for r in [r for r in R if r["bc"] == bc and r["closure"] == cl]:
                lab = "first-order" if cl == "first" else "ghost-point"
                L.append(
                    f"{lab} & {r['N']} & 1/{r['N']} & {e(r['L2'])} & {p(r['p_L2'])} & "
                    f"{e(r['Linf'])} & {p(r['p_Linf'])} & {e(r['Linf_boundary'])} & {e(r['Linf_interior'])}\\\\"
                )
            L.append(r"\midrule" if cl == "first" else r"\bottomrule")
        L.append(r"\end{tabular}")
        open(os.path.join(TAB, f"conv_{bc}.tex"), "w").write("\n".join(L))

    L = [
        r"\begin{tabular}{@{}lrcccccc@{}}\toprule",
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
            r"\begin{tabular}{@{}llrrrc@{}}\toprule",
            r"BC & Closure & $N_\mathrm{min}$ & unknowns & $\|e\|_\infty$ & spsolve time (s)\\\midrule",
        ]
        for r in C:
            if r["N"] in ("", None, "None"):
                L.append(
                    f"{r['bc']} & {'first-order' if r['closure']=='first' else 'ghost-point'} & "
                    r"-- & -- & -- & --\\"
                )
            else:
                L.append(
                    f"{r['bc']} & {'first-order' if r['closure']=='first' else 'ghost-point'} & "
                    f"{r['N']} & {r['M']} & {e(r['Linf'])} & {float(r['t_spsolve_s']):.4e}\\\\"
                )
        L.append(r"\bottomrule\end{tabular}")
        open(os.path.join(TAB, "cost.tex"), "w").write("\n".join(L))
        print("wrote cost.tex")

    print("tables written to", TAB)


if __name__ == "__main__":
    main()
