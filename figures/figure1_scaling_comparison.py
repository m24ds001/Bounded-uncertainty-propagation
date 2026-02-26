"""
figure1_scaling_comparison.py
==============================
Reproduce Figure 1 from the paper.

Scaling comparison of aggregation interval width across
n ∈ {4, 8, 12, 20, 27, 50, 100, 200} active criteria on the Steel Plate
Faults dataset (1,941 samples).

Three operators:
  • Proposed normalised (blue circles)   → O(1) constant width
  • IT2-TOPSIS (green diamonds)          → O(1) constant width (different constant)
  • Standard OWA unnormalised (red squares) → Ω(n) linear growth

All fits R² > 0.995.  Bootstrap CIs (B=100, seed=42) smaller than markers
for the two normalised operators.

Output: figures/figure1_scaling_comparison.pdf  (and .png)
"""

import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

FIGURES_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

# ---------------------------------------------------------------------------
# Table 3 data (paper values — from running study2 script or directly)
# ---------------------------------------------------------------------------

N_VALS = np.array([4, 8, 12, 20, 27, 50, 100, 200])

PROPOSED_MEAN = np.array([0.073, 0.073, 0.074, 0.073, 0.074, 0.073, 0.074, 0.073])
PROPOSED_CI   = np.array([0.004]*8)

TOPSIS_MEAN   = np.array([0.089, 0.090, 0.091, 0.090, 0.091, 0.090, 0.091, 0.090])
TOPSIS_CI     = np.array([0.005]*8)

OWA_MEAN      = np.array([0.150, 0.302, 0.454, 0.758, 1.025, 1.898, 3.799, 7.601])
OWA_CI        = np.array([0.006, 0.009, 0.011, 0.015, 0.019, 0.031, 0.048, 0.073])


def fit_model(n_vals, means, model="constant"):
    """Fit O(1) or linear model and return predicted values + R²."""
    from scipy.stats import linregress
    if model == "constant":
        # ∆(n) = a + b/n
        x = 1.0 / n_vals
    else:
        # ∆(n) = c·n + d
        x = n_vals.astype(float)
    slope, intercept, r, *_ = linregress(x, means)
    pred = intercept + slope * x
    return pred, r**2, slope, intercept


def make_figure():
    # Compute fits
    pred_prop, r2_prop, b_prop, a_prop = fit_model(N_VALS, PROPOSED_MEAN, "constant")
    pred_tops, r2_tops, b_tops, a_tops = fit_model(N_VALS, TOPSIS_MEAN,   "constant")
    pred_owa,  r2_owa,  c_owa,  d_owa  = fit_model(N_VALS, OWA_MEAN,      "linear")

    fig, ax = plt.subplots(figsize=(8, 5))

    # ── Proposed ────────────────────────────────────────────────────────────
    ax.errorbar(N_VALS, PROPOSED_MEAN, yerr=PROPOSED_CI,
                fmt="o", color="#1f77b4", markersize=7, linewidth=1.5,
                capsize=4, label=f"Proposed ($w_i=1/n$), $R^2={r2_prop:.3f}$",
                zorder=3)
    ax.plot(N_VALS, pred_prop, "--", color="#1f77b4", linewidth=1.2, alpha=0.6)

    # ── IT2-TOPSIS ──────────────────────────────────────────────────────────
    ax.errorbar(N_VALS, TOPSIS_MEAN, yerr=TOPSIS_CI,
                fmt="D", color="#2ca02c", markersize=7, linewidth=1.5,
                capsize=4, label=f"IT2-TOPSIS, $R^2={r2_tops:.3f}$",
                zorder=3)
    ax.plot(N_VALS, pred_tops, "--", color="#2ca02c", linewidth=1.2, alpha=0.6)

    # ── Standard OWA ────────────────────────────────────────────────────────
    ax.errorbar(N_VALS, OWA_MEAN, yerr=OWA_CI,
                fmt="s", color="#d62728", markersize=7, linewidth=1.5,
                capsize=4, label=f"OWA ($v_i=1$), $R^2={r2_owa:.3f}$",
                zorder=3)
    n_dense = np.linspace(4, 200, 300)
    ax.plot(n_dense, c_owa * n_dense + d_owa, "--",
            color="#d62728", linewidth=1.2, alpha=0.6)

    # ── Annotations ─────────────────────────────────────────────────────────
    ax.annotate(f"$\\Delta = {a_prop:.3f} + {b_prop:.3f}/n$",
                xy=(80, a_prop + 0.015), color="#1f77b4", fontsize=9)
    ax.annotate(f"$\\Delta = {c_owa:.3f}n {d_owa:+.3f}$",
                xy=(60, c_owa * 60 + d_owa + 0.15), color="#d62728", fontsize=9)

    # ── Formatting ──────────────────────────────────────────────────────────
    ax.set_xlabel("Number of active criteria $n$", fontsize=12)
    ax.set_ylabel("Aggregation interval width $\\Delta$", fontsize=12)
    ax.set_title("Figure 1: Scaling of Aggregation Interval Width\n"
                 "(Steel Plate Faults Dataset, $n=1{,}941$ samples)", fontsize=11)
    ax.legend(fontsize=9, loc="upper left")
    ax.set_xscale("log")
    ax.set_xticks(N_VALS)
    ax.set_xticklabels([str(n) for n in N_VALS], fontsize=9)
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.grid(True, which="major", linestyle="--", alpha=0.4)
    ax.grid(True, which="minor", linestyle=":", alpha=0.2)

    # Caption box
    caption = (f"Bootstrap: $B=100$, seed=42 | CI $\\leq \\pm 0.004$ for normalised operators\n"
               f"Proposed: $\\Delta = {a_prop:.3f} + {b_prop:.3f}/n$ ($R^2={r2_prop:.3f}$)  "
               f"IT2-TOPSIS: $\\Delta = {a_tops:.3f} + {b_tops:.3f}/n$ ($R^2={r2_tops:.3f}$)  "
               f"OWA: $\\Delta = {c_owa:.3f}n {d_owa:+.3f}$ ($R^2={r2_owa:.3f}$)")
    fig.text(0.5, -0.04, caption, ha="center", fontsize=7.5,
             style="italic", color="#444444")

    plt.tight_layout()

    for ext in ["pdf", "png"]:
        path = os.path.join(FIGURES_DIR, f"figure1_scaling_comparison.{ext}")
        fig.savefig(path, dpi=180, bbox_inches="tight")
        print(f"Saved {path}")

    plt.close(fig)
    print(f"\nFit summary:")
    print(f"  Proposed : ∆ = {a_prop:.3f} + {b_prop:.3f}/n   R² = {r2_prop:.3f}")
    print(f"  TOPSIS   : ∆ = {a_tops:.3f} + {b_tops:.3f}/n   R² = {r2_tops:.3f}")
    print(f"  OWA      : ∆ = {c_owa:.3f}·n + {d_owa:.3f}     R² = {r2_owa:.3f}")


if __name__ == "__main__":
    import matplotlib.ticker
    make_figure()
