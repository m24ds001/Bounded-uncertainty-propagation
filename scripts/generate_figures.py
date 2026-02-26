"""
generate_figures.py
===================
Generates Figure 1 (scaling laws) and Figure 2 (width bound decomposition)
from Ramesh & Mehreen 2025.

Usage:
  python scripts/generate_figures.py

Outputs:
  figures/Fig1.pdf   — Scaling Law Comparison (log-log)
  figures/Fig2.pdf   — Width Bound Decomposition bar chart
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError:
    print("matplotlib not available — cannot generate figures.")
    sys.exit(0)

FIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# ── Figure 1: Scaling Law Comparison ────────────────────────────────────────

def figure1():
    """Reproduce Figure 1 from paper (log-log scaling law comparison)."""
    ns = np.array([4, 8, 12, 20, 27, 50, 100, 200])

    # Theoretical values (Table 3 values)
    proposed = np.full_like(ns, 0.073, dtype=float)
    topsis   = np.full_like(ns, 0.090, dtype=float)
    owa       = 0.038 * ns - 0.002

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(ns, proposed, "b-o", label="Proposed (O(1))",
            linewidth=2.2, markersize=8, zorder=4)
    ax.plot(ns, topsis, "g-D", label="IT2-TOPSIS (O(1))",
            linewidth=2.2, markersize=8, zorder=4)
    ax.plot(ns, owa, "r--s", label="Standard OWA (O(n))",
            linewidth=2.2, markersize=8, zorder=4)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of Sources $n$", fontsize=13)
    ax.set_ylabel(r"Aggregation Width $\Delta$", fontsize=13)
    ax.set_title("Scaling Law Comparison", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(True, which="both", linestyle=":", alpha=0.45)
    ax.set_xticks([4, 10, 20, 50, 100, 200])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "Fig1.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure 1 saved → {out}")


# ── Figure 2: Width Bound Decomposition ─────────────────────────────────────

def figure2():
    """Reproduce Figure 2 from paper (stacked bar chart of width components)."""
    k_avg = 5.0
    M = 0.5
    eps_k = 0.5

    eps_c_values = np.array([0.02, 0.05, 0.10, 0.15, 0.20])
    center_comp = k_avg * eps_c_values / 2.0      # kavg * eps_c / 2
    steep_comp  = np.full_like(eps_c_values, M * eps_k / 2.0)  # M * eps_k / 2

    x = np.arange(len(eps_c_values))
    width = 0.5

    fig, ax = plt.subplots(figsize=(7, 5))
    bars_center = ax.bar(x, center_comp, width, label=r"Center ($k_\mathrm{avg}\varepsilon_c/4$)",
                          color="#4472C4", alpha=0.85)
    bars_steep  = ax.bar(x, steep_comp, width, bottom=center_comp,
                          label=r"Steepness ($M\varepsilon_k/4$)",
                          color="#ED7D31", alpha=0.75,
                          hatch="///")

    # Dashed total lines
    for xi, (cc, sc) in enumerate(zip(center_comp, steep_comp)):
        ax.hlines(cc + sc, xi - width/2, xi + width/2,
                  colors="k", linestyles="--", linewidths=1.2)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{e:.2f}" for e in eps_c_values], fontsize=11)
    ax.set_xlabel(r"Center Uncertainty $\varepsilon_c$", fontsize=13)
    ax.set_ylabel("Width Contribution", fontsize=13)
    ax.set_title("Width Bound Decomposition", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_ylim(0, 0.30)
    ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.025))
    ax.grid(True, axis="y", which="major", linestyle=":", alpha=0.4)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "Fig2.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure 2 saved → {out}")


if __name__ == "__main__":
    import matplotlib.ticker
    figure1()
    figure2()
    print("All figures generated.")
