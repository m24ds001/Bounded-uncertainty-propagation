"""
concentration.py
================
Sub-Gaussian concentration bounds for IT2 weighted aggregation.

Implements:
  - Theorem 15    : Bernstein-type bound for IT2 aggregate
  - Remark 5      : Comparison with Hoeffding (when Bernstein is tighter)
  - Corollary 16  : Combined deviation bound; residual = (k_avg·ϵ_c + M·ϵ_k)/8

The key innovation over generic Hoeffding: the sigmoid steepness kᵢ appears
explicitly through σ²ᵢ ≤ kᵢ²σ̃²ᵢ/16, making the bound tighter whenever
inputs are concentrated near their centres.

Reference
---------
Ramesh & Mehreen (2025), FODM (submitted).
Bernstein, S. (1924).  On a modification of Chebyshev's inequality.
Hoeffding, W. (1963).  JASA 58(301), 13–30.
"""

import numpy as np
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Theorem 15 — Bernstein sub-Gaussian bound
# ---------------------------------------------------------------------------

def bernstein_bound(
    weights: np.ndarray,
    sigma2_i: np.ndarray,
    delta: float,
) -> Tuple[float, float]:
    """
    Bernstein-type tail bound for the IT2 aggregate  En = Σ wᵢµᵢ(τᵢ)
    (Theorem 15).

    The i-th summand wᵢµᵢ(τᵢ) has:
      - Variance:   w²ᵢ · σ²ᵢ
      - Range:      [0, wᵢ]   (since µᵢ ∈ [0,1])

    Bernstein's inequality gives:

        P(|En - E[En]| ≥ t) ≤ 2·exp( -t²/2 / (σ²_w + Bw·t/3) )

    Inverting at level δ produces the explicit bound returned here.

    Parameters
    ----------
    weights  : (n,) normalised weight vector.
    sigma2_i : (n,) per-source membership variances Var(µᵢ(τᵢ)).
               Upper bound: σ²ᵢ ≤ kᵢ²σ̃²ᵢ/16 where σ̃²ᵢ = Var(τᵢ).
    delta    : Confidence level; bound holds with probability ≥ 1 - δ.

    Returns
    -------
    (t_bernstein, sigma2_w) :
        t_bernstein — width of the 1-δ confidence interval.
        sigma2_w    — weighted variance Σ w²ᵢσ²ᵢ.

    Notes
    -----
    The bound is obtained by solving the quadratic:
        t² / 2 = log(2/δ) · (σ²_w + Bw·t/3)
    giving:
        t = Bw·log(2/δ)/3 + sqrt((Bw·log(2/δ)/3)² + 2·σ²_w·log(2/δ))
    """
    weights  = np.asarray(weights, float)
    sigma2_i = np.asarray(sigma2_i, float)

    sigma2_w = float(np.dot(weights ** 2, sigma2_i))    # Σ w²ᵢσ²ᵢ
    Bw       = float(weights.max())                      # max wᵢ
    log2d    = np.log(2.0 / delta)

    # Solve t² / 2 = log(2/δ) · (σ²_w + Bw·t/3)
    # ⟺ t² - (2·Bw·log2d/3)·t - 2·σ²_w·log2d = 0
    # Positive root:
    b_coeff = 2.0 * Bw * log2d / 3.0
    c_coeff = 2.0 * sigma2_w * log2d
    t       = (b_coeff + np.sqrt(b_coeff ** 2 + 4.0 * c_coeff)) / 2.0

    return float(t), sigma2_w


def hoeffding_bound(
    weights: np.ndarray,
    delta: float,
) -> float:
    """
    Distribution-free Hoeffding bound for comparison with Theorem 15.

    For bounded summands wᵢµᵢ ∈ [0, wᵢ], Hoeffding gives:

        P(|En - E[En]| ≥ t) ≤ 2·exp(-2t² / Σ wᵢ²)

    Inverting:  t_hoeffding = sqrt( Σ wᵢ² · log(2/δ) / 2 ).

    Parameters
    ----------
    weights : (n,) normalised weights.
    delta   : Confidence level.

    Returns
    -------
    float : Hoeffding bound on |En - E[En]|.
    """
    weights = np.asarray(weights, float)
    return float(np.sqrt(np.sum(weights ** 2) * np.log(2.0 / delta) / 2.0))


def bernstein_vs_hoeffding(
    weights: np.ndarray,
    sigma2_i: np.ndarray,
    delta: float,
) -> dict:
    """
    Compare Bernstein and Hoeffding bounds (Remark 5).

    Bernstein is strictly tighter when:
        σ²_w < (max wᵢ)² / 4

    which holds when membership function variance is below its worst-case value.

    Returns
    -------
    dict with 'bernstein', 'hoeffding', 'improvement_factor', 'bernstein_tighter'.
    """
    t_b, sigma2_w = bernstein_bound(weights, sigma2_i, delta)
    t_h           = hoeffding_bound(weights, delta)
    Bw            = float(np.asarray(weights).max())
    condition_met = sigma2_w < (Bw ** 2) / 4.0

    return {
        "bernstein"         : t_b,
        "hoeffding"         : t_h,
        "improvement_factor": t_h / t_b if t_b > 0 else np.inf,
        "bernstein_tighter" : t_b < t_h,
        "condition_met"     : condition_met,
        "sigma2_w"          : sigma2_w,
        "Bw_squared_over_4" : (Bw ** 2) / 4.0,
    }


# ---------------------------------------------------------------------------
# Corollary 16 — Combined deviation bound
# ---------------------------------------------------------------------------

def combined_deviation_bound(
    weights: np.ndarray,
    sigma2_i: np.ndarray,
    delta: float,
    k_avg: float,
    eps_c: float,
    M: float,
    eps_k: float,
) -> dict:
    """
    Combined deviation bound from Theorem 15 + Theorem 10(i) (Corollary 16):

        | (E + Ē)/2 - E[En] | ≤  t_bernstein  +  (k_avg·ϵ_c + M·ϵ_k) / 8

    simultaneously with probability ≥ 1 - δ.

    As n → ∞ under uniform weights:
      - σ²_w  → 0   (statistical uncertainty vanishes)
      - residual = (k_avg·ϵ_c + M·ϵ_k)/8  (irreducible epistemic imprecision)

    Parameters
    ----------
    weights  : (n,) normalised weights.
    sigma2_i : (n,) per-source membership variances.
    delta    : Confidence level.
    k_avg    : Average steepness.
    eps_c    : Centre uncertainty.
    M        : Domain bound.
    eps_k    : Steepness uncertainty.

    Returns
    -------
    dict with 'statistical_term', 'epistemic_residual', 'total_bound'.
    """
    t_b, sigma2_w  = bernstein_bound(weights, sigma2_i, delta)
    epistemic_res  = (k_avg * eps_c + M * eps_k) / 8.0

    return {
        "statistical_term"  : t_b,
        "epistemic_residual": epistemic_res,
        "total_bound"       : t_b + epistemic_res,
        "sigma2_w"          : sigma2_w,
    }


# ---------------------------------------------------------------------------
# Variance from steepness (Remark 5 approximation)
# ---------------------------------------------------------------------------

def sigma2_from_steepness(
    k: np.ndarray,
    sigma2_tau: np.ndarray,
) -> np.ndarray:
    """
    Approximate membership variance from sigmoid steepness and input variance.

    For τᵢ ~ N(cᵢ, σ̃²ᵢ) and small kᵢσ̃ᵢ (Remark 5):

        Var(µᵢ(τᵢ)) ≈ kᵢ² · σ̃²ᵢ / 16

    The approximation is tight for kᵢσ̃ᵢ ≪ 1 and conservative otherwise.

    Parameters
    ----------
    k         : (n,) steepness parameters.
    sigma2_tau: (n,) input variances σ̃²ᵢ = Var(τᵢ).

    Returns
    -------
    np.ndarray : (n,) upper bounds on membership variances.
    """
    k         = np.asarray(k, float)
    sigma2_tau= np.asarray(sigma2_tau, float)
    return (k ** 2) * sigma2_tau / 16.0


# ---------------------------------------------------------------------------
# Corollary 16 — Uniform weight asymptotic
# ---------------------------------------------------------------------------

def corollary16_asymptotic(
    n: int,
    sigma2_max: float,
    k_avg: float,
    eps_c: float,
    M: float,
    eps_k: float,
    delta: float,
) -> dict:
    """
    Illustrate Corollary 16: as n → ∞ under uniform weights, only the
    epistemic residual (k_avg·ϵ_c + M·ϵ_k)/8 survives.

    Returns statistical and epistemic components for a range of n values.
    """
    ns       = np.geomspace(1, n, 50, dtype=int)
    stat_terms = []
    for ni in ns:
        w         = np.full(ni, 1.0 / ni)
        sigma2_i  = np.full(ni, sigma2_max)
        t_b, _    = bernstein_bound(w, sigma2_i, delta)
        stat_terms.append(t_b)

    return {
        "ns"               : ns,
        "statistical_terms": np.array(stat_terms),
        "epistemic_residual": (k_avg * eps_c + M * eps_k) / 8.0,
    }
