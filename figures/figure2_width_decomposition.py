"""
figure2_width_decomposition.py
================================
Reproduce Figure 2 from the paper.

Decomposition of the certified width bound (Theorem 8) into:
  • Centre-uncertainty component  :  k_avg · ϵ_c / 4  (blue)
  • Steepness-uncertainty component: M · ϵ_k / 4       (coral)

As ϵ_c increases from 0 → 0.3, the total bound grows linearly while the
steepness contribution remains constant.

Parameters match the four-criteria supplier example (Section 4.4 / Example 7):
  k_avg = 3.30,  M = 0.5,  ϵ_k = 0.5.

Output: figures/figure2_width_decomposition.pdf  (and .png)
"""

import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

FIGURES_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Parameters (Section 4.4 / Example 7) ───────────────────────────────────
K_AVG = 3.30
M     = 0.5
EPS_K = 0.5
EPS_C_RANGE = np.linspace(0.0, 0.30, 300)

# ── Components ──────────────────────────────────────────────────────────────
centre_component    = K_AVG * EPS_C_RANGE / 4.0        # grows with ϵ_c
steepness_component = np.full_like(EPS_C_RANGE, M * EPS_K / 4.0)   # constant
total_bound         = centre_component + steepness_component


def make_figure():
    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Stacked area
    ax.fill_between(EPS_C_RANGE, 0, steepness_component,
                    color="#e07b54", alpha=0.75, label="Steepness component $M\\epsilon_k/4$")
    ax.fill_between(EPS_C_RANGE, steepness_component, total_bound,
                    color="#4c9be8", alpha=0.75, label="Centre component $k_{\\mathrm{avg}}\\epsilon_c/4$")
    ax.plot(EPS_C_RANGE, total_bound, color="#1a1a2e", linewidth=2.0,
            label="Total bound $\\Delta_{\\mathrm{cert}}$")

    # Reference line at ϵ_c = 0.10 (corporate weight example)
    eps_ref = 0.10
    y_ref   = K_AVG * eps_ref / 4.0 + M * EPS_K / 4.0
    ax.axvline(eps_ref, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.annotate(f"$\\epsilon_c = {eps_ref}$\n$\\Delta = {y_ref:.3f}$",
                xy=(eps_ref, y_ref), xytext=(eps_ref + 0.02, y_ref + 0.02),
                fontsize=9, color="gray",
                arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))

    # Critical tolerance ϵ*_c from Example 7
    eps_star = 0.167
    ax.axvline(eps_star, color="#c0392b", linestyle=":", linewidth=1.2)
    ax.text(eps_star + 0.005, 0.01,
            f"$\\epsilon^*_c \\approx {eps_star}$\n(cert. collapses)",
            fontsize=8.5, color="#c0392b", va="bottom")

    # ── Formatting ──────────────────────────────────────────────────────────
    ax.set_xlabel("Centre uncertainty $\\epsilon_c$", fontsize=12)
    ax.set_ylabel("Certified width bound $\\Delta_{\\mathrm{cert}}$", fontsize=12)
    ax.set_title("Figure 2: Width Bound Decomposition (Theorem 8)\n"
                 "$k_{\\mathrm{avg}}=3.30$, $M=0.5$, $\\epsilon_k=0.5$",
                 fontsize=11)
    ax.legend(fontsize=9, loc="upper left")
    ax.set_xlim(0, 0.30)
    ax.set_ylim(0, None)
    ax.grid(True, linestyle="--", alpha=0.35)

    # Annotation: steepness component is flat
    ax.annotate("Steepness term constant\n$(M\\epsilon_k/4 = {:.4f})$".format(M * EPS_K / 4),
                xy=(0.25, M * EPS_K / 8.0),
                fontsize=8.5, color="#e07b54",
                ha="center")

    plt.tight_layout()

    for ext in ["pdf", "png"]:
        path = os.path.join(FIGURES_DIR, f"figure2_width_decomposition.{ext}")
        fig.savefig(path, dpi=180, bbox_inches="tight")
        print(f"Saved {path}")

    plt.close(fig)


if __name__ == "__main__":
    make_figure()
