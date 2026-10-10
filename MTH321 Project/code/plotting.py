"""Publication-ready PDF figures rendered solely from saved CSV data."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import lognorm


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _style() -> None:
    plt.rcParams.update({"font.size": 10, "axes.labelsize": 11, "axes.titlesize": 12,
                         "legend.fontsize": 9, "pdf.fonttype": 42,
                         "figure.dpi": 150, "savefig.bbox": "tight"})


def convergence_figure(csv_path: Path, pdf_path: Path, *, title: str,
                       ylabel: str, group_key: str, weak: bool) -> None:
    """Show every measured point; filled markers identify fitted points."""
    _style()
    rows = read_rows(csv_path)
    groups = list(dict.fromkeys(row[group_key] for row in rows))
    if weak:
        fig, (ax, signed_ax) = plt.subplots(2, 1, figsize=(7.4, 7.0),
                                            sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    else:
        fig, ax = plt.subplots(figsize=(7.4, 5.0))
        signed_ax = None
    colors = plt.get_cmap("tab10")
    for i, group in enumerate(groups):
        selected = sorted((r for r in rows if r[group_key] == group), key=lambda r: float(r["h"]))
        h = np.array([float(r["h"]) for r in selected])
        estimate = np.array([float(r["estimate"]) for r in selected])
        se = np.array([float(r["se"]) for r in selected])
        values = np.abs(estimate) if weak else estimate
        fitted = np.array([r["fit_included"] == "1" for r in selected])
        color = colors(i)
        label = f"reference N={group}" if group_key == "reference_N" else group
        slope_text = selected[0].get("fitted_slope", "")
        fit_count = int(selected[0].get("fit_point_count", "0"))
        eligible_count = int(selected[0].get("eligible_point_count", "0"))
        if slope_text and fit_count >= 4:
            label += f" (fit slope {float(slope_text):.3f}; {fit_count} points)"
        elif eligible_count >= 2:
            label += f" (unresolved; {eligible_count} points)"
        else:
            label += " (no resolved fit)"
        ax.plot(h, values, color=color, linewidth=1.3, alpha=0.65, label=label)
        if np.any(fitted):
            # On weak plots the fit rule ensures the lower error bar is positive.
            ax.errorbar(h[fitted], values[fitted], yerr=1.96 * se[fitted], fmt="o",
                        color=color, capsize=2.5, markersize=5)
        if np.any(~fitted):
            ax.scatter(h[~fitted], values[~fitted], facecolors="white", edgecolors=[color],
                       marker="o", s=34, zorder=4)
        if signed_ax is not None:
            signed_ax.errorbar(h, estimate, yerr=1.96 * se, fmt="o-", markersize=3.5,
                               color=color, capsize=2.3, linewidth=1.0, label=label)
        if slope_text and np.count_nonzero(fitted) >= 4:
            slope = float(slope_text)
            xfit = h[fitted]
            # Anchor guide at geometric center of fitted measured data.
            intercept = float(np.mean(np.log(values[fitted]) - slope * np.log(xfit)))
            ax.plot(xfit, np.exp(intercept) * xfit**slope, "--", color=color, linewidth=1.2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    if signed_ax is None:
        ax.set_xlabel(r"Time step $h=\Delta t$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best")
    if signed_ax is not None:
        if "analytic_bias" in rows[0]:
            analytic = sorted((r for r in rows if r[group_key] == groups[0]),
                              key=lambda r: float(r["h"]))
            analytic_h = np.array([float(r["h"]) for r in analytic])
            analytic_y = np.array([float(r["analytic_bias"]) for r in analytic])
            ax.plot(analytic_h, np.abs(analytic_y), ":", color="black", linewidth=1.5,
                    label="analytic |bias|")
            signed_ax.plot(analytic_h, analytic_y, ":", color="black", linewidth=1.5,
                           label="analytic signed bias")
            ax.legend(loc="best")
        signed_ax.set_xscale("log")
        signed_ax.axhline(0.0, color="black", linewidth=0.9, alpha=0.6)
        signed_ax.set_xlabel(r"Time step $h=\Delta t$")
        signed_ax.set_ylabel("Signed paired difference")
        signed_ax.grid(True, which="both", alpha=0.25)
        signed_ax.text(0.01, -0.40,
                       f"M={int(rows[0]['M']):,}; seed={rows[0]['seed']}. Top: absolute estimates (open = unresolved). "
                       "Bottom: signed estimates with 95% CIs.",
                       transform=signed_ax.transAxes, fontsize=8, va="top")
    else:
        ax.text(0.01, -0.21, f"M={int(rows[0]['M']):,}; seed={rows[0]['seed']}. "
                "Open = excluded from fit; bars = 95% CI.",
                transform=ax.transAxes, fontsize=8, va="top")
    if group_key == "reference_N":
        target = signed_ax if signed_ax is not None else ax
        target.text(0.01, -0.52 if weak else -0.30,
                    "References are numerical EM solutions, not exact paths; slopes are empirical.",
                    transform=target.transAxes, fontsize=8, va="top")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path)
    plt.close(fig)


def distribution_figure(csv_path: Path, pdf_path: Path) -> None:
    _style()
    rows = read_rows(csv_path)
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    colors = {"EM": "tab:blue", "Milstein": "tab:orange"}
    for method in ("EM", "Milstein"):
        hist = [r for r in rows if r["record_type"] == "histogram" and r["method"] == method]
        edges = np.array([float(r["bin_left"]) for r in hist] + [float(hist[-1]["bin_right"])])
        heights = np.array([float(r["density"]) for r in hist])
        ax.stairs(heights, edges, label=f"{method} (N={rows[0]['N']})",
                  color=colors[method], linewidth=1.2, alpha=0.8)
    s0, mu, sigma, t = (float(rows[0][key]) for key in ("s0", "mu", "sigma", "T"))
    x = np.linspace(max(edges[0], 1e-10), edges[-1], 700)
    scale = np.exp(np.log(s0) + (mu - 0.5 * sigma**2) * t)
    ax.plot(x, lognorm.pdf(x, s=sigma * np.sqrt(t), scale=scale),
            color="black", linewidth=2.0, label="Exact lognormal density")
    ax.set_xlabel(r"Terminal asset price $S_T$")
    ax.set_ylabel("Probability density")
    ax.set_title(f"GBM terminal distribution; M={int(rows[0]['M']):,}; seed={rows[0]['seed']}")
    ax.grid(True, alpha=0.2)
    ax.legend()
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path)
    plt.close(fig)


def make_all_figures(results: Path, figures: Path) -> list[Path]:
    figures.mkdir(parents=True, exist_ok=True)
    specs = [
        ("gbm_strong", "GBM strong error against exact path", r"$\mathbb{E}|S_N-S_T|$", "method", False),
        ("gbm_weak", "GBM weak bias of terminal mean", r"$|\mathbb{E}[S_N-S_T]|$", "method", True),
        ("nonaffine_strong", "Non-affine coupled strong refinement", r"$\mathbb{E}|S_N-S_{N_{\rm ref}}|$", "reference_N", False),
        ("nonaffine_weak", "Non-affine coupled payoff refinement", r"$|\mathbb{E}[\phi(S_N)-\phi(S_{N_{\rm ref}})]|$", "reference_N", True),
    ]
    made = []
    for stem, title, ylabel, group, weak in specs:
        destination = figures / f"{stem}_convergence.pdf"
        convergence_figure(results / f"{stem}.csv", destination, title=title,
                           ylabel=ylabel, group_key=group, weak=weak)
        made.append(destination)
    destination = figures / "gbm_distribution.pdf"
    distribution_figure(results / "gbm_distribution.csv", destination)
    made.append(destination)
    return made
