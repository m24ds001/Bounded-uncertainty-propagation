"""
example_supplier.py
===================
Worked example: four-criteria supplier selection
(Section 4.4, Example 4.1 of Ramesh & Mehreen 2025).

Reproduces all numerical results including:
  - Width certificate (Theorem 3.8)
  - Width-optimal weight allocation (Theorem 3.9)
  - Certified ranking of two alternatives (Remark 4.2)
  - Critical tolerance ε_c* computation
  - Indeterminate-case recovery (dominant wisi identification)

Usage:
  python scripts/example_supplier.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from it2_aggregation import (
    IT2Source, aggregate_it2, certify_decision, certify_ranking,
    certified_width_bound, sensitivity_coefficients,
    width_optimal_weights, sigmoid,
)


def run_supplier_example():
    print("=" * 60)
    print("Example 4.1: Four-criteria supplier selection")
    print("=" * 60)

    # ── Problem setup ─────────────────────────────────────────────
    tau = [0.82, 0.71, 0.68, 0.65]
    c   = [0.60, 0.65, 0.55, 0.60]
    k   = [3.0,  4.0,  3.0,  3.0]
    w   = np.array([0.40, 0.30, 0.20, 0.10])
    eps_c, eps_k = 0.10, 0.50
    labels = ["Technical compat.", "Regulatory comply.",
              "Delivery consist.", "Cost effectiveness"]

    sources_A = [
        IT2Source(tau=tau[i], c=c[i], k=k[i],
                  eps_c=eps_c, eps_k=eps_k, label=labels[i])
        for i in range(4)
    ]

    # ── Type-1 aggregate ──────────────────────────────────────────
    mu_nom = [sigmoid(tau[i], c[i], k[i]) for i in range(4)]
    E_type1 = float(np.dot(w, mu_nom))
    print("\nSigmoid membership grades (nominal):")
    for i, name in enumerate(labels):
        print(f"  µ_{i+1} = {mu_nom[i]:.3f}   ({name})")
    print(f"\nType-1 aggregate  E = {E_type1:.4f}")

    # ── IT2 aggregation ───────────────────────────────────────────
    result_A = aggregate_it2(sources_A, weights=w)
    M = max(abs(tau[i] - c[i]) for i in range(4))
    k_avg = float(np.dot(w, k))

    print(f"\nIT2 interval  [E̲, Ē] = [{result_A.E_lower:.4f}, {result_A.E_upper:.4f}]")
    print(f"Empirical width        = {result_A.width_empirical:.4f}")
    print(f"Certified width bound  = {result_A.width_certified:.4f}")
    print(f"kavg = {k_avg:.2f},  M = {M:.2f}")

    cert_bound_paper = 2.0 * (k_avg * eps_c / 4.0 + M * eps_k / 4.0)
    print(f"\nPaper Theorem 3.8 check:")
    print(f"  2*(kavg*ε_c/4 + M*ε_k/4) = 2*({k_avg:.2f}*{eps_c}/4 + "
          f"{M:.2f}*{eps_k}/4) = {cert_bound_paper:.3f}")

    # ── Decision at threshold 0.50 ────────────────────────────────
    theta = 0.50
    half_w = result_A.width_certified / 2.0
    E_lo = E_type1 - half_w
    print(f"\nAt threshold θ = {theta}:")
    print(f"  E̲ ≥ E - half_width = {E_type1:.4f} - {half_w:.4f} = {E_lo:.4f}")
    dec_A = certify_decision(result_A, threshold=theta + half_w)
    print(f"  Decision: {dec_A}  (margin is narrow — see paper)")

    # ── Reducing ε_c to 0.09 ─────────────────────────────────────
    print("\nReducing ε_c → 0.09:")
    eps_c_new = 0.09
    sources_A09 = [
        IT2Source(tau=tau[i], c=c[i], k=k[i],
                  eps_c=eps_c_new, eps_k=eps_k)
        for i in range(4)
    ]
    res09 = aggregate_it2(sources_A09, weights=w)
    print(f"  New certified width = {res09.width_certified:.4f}")
    print(f"  E̲ ≥ {E_type1 - res09.width_certified/2:.4f}  "
          f"({'> 0.50 ✓' if E_type1 - res09.width_certified/2 > 0.50 else '≤ 0.50'})")

    # ── Width-optimal weight allocation (Theorem 3.9) ─────────────
    print("\nWidth-optimal weight allocation (Theorem 3.9):")
    s = sensitivity_coefficients(sources_A, M)
    for i, name in enumerate(labels):
        print(f"  s_{i+1} = {s[i]:.4f}   ({name})")
    w_opt, min_width = width_optimal_weights(sources_A, M)
    print(f"\n  Optimal degenerate weights: {w_opt}")
    print(f"  Minimum certified width:    {min_width:.4f}")
    print(f"  Corporate weight width:     {result_A.width_certified:.4f}")
    print(f"  Premium over optimum:       {result_A.width_certified - min_width:.4f}")

    # ── Critical tolerance ε_c* ───────────────────────────────────
    print("\nCritical tolerance ε_c* (balance point at θ=0.50):")
    # E_type1 - (kavg*eps_c*/4 + M*eps_k/4) = 0.50
    # kavg*eps_c*/4 = E_type1 - 0.50 - M*eps_k/4
    rhs = E_type1 - 0.50 - M * eps_k / 4.0
    eps_c_star = rhs * 4.0 / k_avg
    print(f"  ε_c* = (E - θ - M·ε_k/4) · 4/kavg "
          f"= ({E_type1:.4f} - 0.50 - {M*eps_k/4:.4f}) · 4/{k_avg:.2f} "
          f"= {eps_c_star:.4f}")

    # ── Second alternative ────────────────────────────────────────
    print("\n" + "-" * 60)
    print("Second alternative B:")
    tau_B = [0.55, 0.68, 0.72, 0.70]
    mu_B = [sigmoid(tau_B[i], c[i], k[i]) for i in range(4)]
    E_B = float(np.dot(w, mu_B))
    print(f"  µ' = {[f'{m:.3f}' for m in mu_B]}")
    print(f"  E' = {E_B:.4f}")

    sources_B = [
        IT2Source(tau=tau_B[i], c=c[i], k=k[i],
                  eps_c=eps_c, eps_k=eps_k)
        for i in range(4)
    ]
    result_B = aggregate_it2(sources_B, weights=w)
    print(f"  IT2 interval B = [{result_B.E_lower:.3f}, {result_B.E_upper:.3f}]")
    print(f"  IT2 interval A = [{result_A.E_lower:.3f}, {result_A.E_upper:.3f}]")

    rank = certify_ranking(result_A, result_B)
    print(f"  Certified ranking: {rank}")
    print(f"  Overlap: intervals {'DO' if rank == 'indeterminate' else 'do NOT'} overlap")

    # Critical frontier condition
    diff = E_type1 - E_B
    print(f"\nCritical frontier (Remark 4.2):")
    print(f"  E_A - E_B = {diff:.4f}")
    print(f"  2*(kavg*ε_c/4 + M*ε_k/4) per side = {result_A.width_certified/2:.4f}")
    print(f"  Ranking requires E_A - E_B > total_width = "
          f"{result_A.width_certified:.4f}")
    print(f"  Floor from ε_k alone: 2 * M*ε_k/2 = {2*M*eps_k/2:.4f}")
    if 2 * M * eps_k / 2 > diff:
        print(f"  Cannot certify ranking by reducing ε_c alone ✓")
    eps_c_cert = diff / k_avg if k_avg > 0 else np.nan
    print(f"  If ε_k=0: need ε_c ≤ (E_A-E_B)/kavg = {diff:.4f}/{k_avg:.2f} = {eps_c_cert:.4f}")

    # ── Indeterminate recovery: wisi decomposition ─────────────────
    print("\n" + "-" * 60)
    print("Indeterminate recovery — per-source wisi decomposition:")
    ws_i = 2.0 * w * s
    for i, name in enumerate(labels):
        print(f"  2·w_{i+1}·s_{i+1} = 2·{w[i]:.2f}·{s[i]:.4f} = {ws_i[i]:.4f}"
              f"  ({ws_i[i]/ws_i.sum()*100:.1f}%)")
    print(f"  Sum = {ws_i.sum():.4f}  (≈ certified width {result_A.width_certified:.4f})")

    dom_idx = int(np.argmax(ws_i))
    print(f"\n  Dominant criterion: #{dom_idx+1} ({labels[dom_idx]})")
    eps_c_reduced = eps_c / 2.0
    s_new = k[dom_idx] * eps_c_reduced / 4.0 + M * eps_k / 4.0
    delta_hw = w[dom_idx] * (s[dom_idx] - s_new)
    print(f"  Reduce ε_c for criterion {dom_idx+1}: {eps_c} → {eps_c_reduced}")
    print(f"  Half-width reduction: {delta_hw:.4f}")
    print(f"  New E̲ ≥ {E_type1 - half_w + delta_hw:.4f}"
          f"  ({'> 0.50 ✓' if E_type1 - half_w + delta_hw > 0.50 else '≤ 0.50'})")

    print("\n[Example 4.1 complete]")


if __name__ == "__main__":
    run_supplier_example()
