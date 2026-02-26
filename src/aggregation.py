"""
aggregation.py
==============
IT2 weighted aggregation operators and the unnormalised lower-bound result.

Implements:
  - Definition 8  : Unified aggregate score E = Σ wᵢµᵢ
  - Theorem 3     : Weighted average properties (idempotence, bounds, convexity)
  - Theorem 13    : Width growth lower bound for unnormalised operators (Ω(n))
  - Corollary 14  : Normalisation necessary & sufficient for O(1) width
  - Lemma 19      : Parameter sensitivity bounds (|∂E/∂cⱼ|, |∂E/∂kⱼ|)
  - Theorem 20    : Stability under simultaneous perturbation

Reference
---------
Ramesh & Mehreen (2025), FODM (submitted).
"""

import numpy as np
from typing import Tuple, Optional
from .membership import sigmoid, sigmoid_lipschitz_constant


# ---------------------------------------------------------------------------
# Normalised weighted aggregation (Definition 8 / Theorem 3)
# ---------------------------------------------------------------------------

def normalised_aggregate(
    mu: np.ndarray,
    weights: np.ndarray,
) -> float:
    """
    Compute the normalised weighted aggregate E = Σ wᵢµᵢ  (Definition 8).

    Satisfies all five Theorem 3 properties:
      (i)   0 ≤ E ≤ 1
      (ii)  Convex combination of membership grades
      (iii) minᵢ µᵢ ≤ E ≤ maxᵢ µᵢ
      (iv)  Idempotent: µᵢ = c ∀i ⟹ E = c

    Parameters
    ----------
    mu      : (n,) array of membership grades µᵢ ∈ [0, 1].
    weights : (n,) array of non-negative weights summing to 1.

    Returns
    -------
    float : Aggregate score E ∈ [0, 1].

    Raises
    ------
    ValueError : If weights do not sum to 1 (tolerance 1e-8) or µᵢ ∉ [0,1].
    """
    mu      = np.asarray(mu, float)
    weights = np.asarray(weights, float)

    if not np.all(weights >= -1e-12):
        raise ValueError("All weights must be non-negative (Axiom i).")
    if abs(weights.sum() - 1.0) > 1e-8:
        raise ValueError(
            f"Weights must sum to 1 (Axiom ii); got sum = {weights.sum():.6f}."
        )
    if not (np.all(mu >= -1e-12) and np.all(mu <= 1.0 + 1e-12)):
        raise ValueError("Membership grades must be in [0, 1].")

    return float(np.dot(weights, mu))


def it2_aggregate_interval(
    mu_lo: np.ndarray,
    mu_hi: np.ndarray,
    weights: np.ndarray,
) -> Tuple[float, float]:
    """
    Aggregate an IT2 source collection to produce the output interval [E, Ē].

    Given per-source interval memberships [µ_lo_i, µ_hi_i] and normalised
    weights, the aggregate interval is:

        E  = Σ wᵢ µ_lo_i   (lower bound)
        Ē  = Σ wᵢ µ_hi_i   (upper bound)

    Parameters
    ----------
    mu_lo   : (n,) lower membership bounds.
    mu_hi   : (n,) upper membership bounds.
    weights : (n,) normalised weights.

    Returns
    -------
    (E_lo, E_hi) : float tuple, the IT2 output interval.
    """
    E_lo = normalised_aggregate(mu_lo, weights)
    E_hi = normalised_aggregate(mu_hi, weights)
    return E_lo, E_hi


# ---------------------------------------------------------------------------
# Unnormalised operator (Theorem 13 / Example 1)
# ---------------------------------------------------------------------------

def unnormalised_aggregate(
    mu: np.ndarray,
    v: float = 1.0,
) -> float:
    """
    Unnormalised aggregate F_n(µ) = Σ vᵢµᵢ with fixed vᵢ = v > 0.

    For n sources, the width grows as Ω(n) (Theorem 13).
    """
    return float(v * np.asarray(mu, float).sum())


def unnormalised_width_lower_bound(
    n: int,
    v: float,
    k_min: float,
    eps_c: float,
) -> float:
    """
    Lower bound on the width of the unnormalised operator (Theorem 13):

        Δ_unnorm ≥ n · v · k_min · ϵ_c / 4

    Parameters
    ----------
    n     : Number of sources.
    v     : Fixed weight per source.
    k_min : Minimum steepness parameter (Assumption 1).
    eps_c : Centre uncertainty magnitude.

    Returns
    -------
    float : Width lower bound (grows without bound as n → ∞).

    Example
    -------
    >>> unnormalised_width_lower_bound(50, 1.0, 2.0, 0.1)
    2.5   # versus normalised bound of 0.05
    """
    return n * v * k_min * eps_c / 4.0


# ---------------------------------------------------------------------------
# Lemma 19 + Theorem 20 — Sensitivity bounds and stability under perturbation
# ---------------------------------------------------------------------------

def stability_constant(
    weights: np.ndarray,
    k: np.ndarray,
    M: float,
) -> float:
    """
    Stability constant C from Theorem 20:

        |E(θ + δ) - E(θ)| ≤ C · ‖δ‖_∞

    where  C = (1/4) Σ wᵢ (kᵢ + M).

    Derived from Lemma 19 (parameter sensitivity bounds) via the multivariate
    mean value theorem (Theorem 20).

    Parameters
    ----------
    weights : (n,) normalised weight vector.
    k       : (n,) steepness parameters.
    M       : Domain bound sup_i |τᵢ - cᵢ|.

    Returns
    -------
    float : Lipschitz stability constant C.
    """
    weights = np.asarray(weights, float)
    k       = np.asarray(k, float)
    return float(0.25 * np.dot(weights, k + M))


def verify_stability(
    tau: np.ndarray,
    c: np.ndarray,
    k: np.ndarray,
    weights: np.ndarray,
    delta_c: np.ndarray,
    delta_k: np.ndarray,
    M: float,
) -> dict:
    """
    Numerically verify Theorem 20: compute actual vs certified perturbation bound.

    Returns
    -------
    dict with keys 'actual_change', 'certified_bound', 'satisfied'.
    """
    E_base    = normalised_aggregate(sigmoid(tau, c, k), weights)
    E_perturb = normalised_aggregate(sigmoid(tau, c + delta_c, k + delta_k), weights)
    actual    = abs(E_perturb - E_base)

    eps       = max(np.abs(delta_c).max(), np.abs(delta_k).max())
    C         = stability_constant(weights, k, M)
    certified = C * eps

    return {
        "actual_change"  : actual,
        "certified_bound": certified,
        "satisfied"      : actual <= certified + 1e-10,
        "tightness_ratio": actual / certified if certified > 0 else np.nan,
    }


