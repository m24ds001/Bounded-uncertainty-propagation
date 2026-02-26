"""
yager.py
========
Yager λ-aggregation with certified O(1) width  (Theorem 6 / Definition 9).

The Yager power-mean family:

    E_λ = ( Σ wᵢ µᵢ(τᵢ)^λ )^{1/λ}   for λ ≥ 1

reduces to the arithmetic mean at λ=1 and to the maximum as λ → ∞.

Theorem 6 extends the O(1) width certificate from Theorem 10(iii) to the
full Yager family via a subadditivity argument on the ℓ_λ norm:

    Δ(Ã^λ_n) ≤ λ^{1/λ} · (K')^{1/λ}

where K' = max_i { k_i/4 · ϵ_c + M/4 · ϵ_k }.

The factor λ^{1/λ} achieves its maximum e^{1/e} ≈ 1.445 at λ=e and → 1
as λ → ∞.  For λ=1 the bound reduces to K', recovering Theorem 10(iii).

Reference
---------
Ramesh & Mehreen (2025), FODM (submitted).
Beliakov, G., Pradera, A., & Calvo, T. (2007). Aggregation Functions:
  A Guide for Practitioners. Springer.
Yager, R.R. (1988). IEEE SMC 18(1), 183–190.
"""

import numpy as np
from typing import Union
from .width_bounds import per_source_sensitivity


# ---------------------------------------------------------------------------
# Definition 9 — Yager λ-aggregate
# ---------------------------------------------------------------------------

def yager_aggregate(
    mu: np.ndarray,
    weights: np.ndarray,
    lam: float,
) -> float:
    """
    Yager λ-aggregate  E_λ = (Σ wᵢµᵢ^λ)^{1/λ}  (Definition 9).

    Parameters
    ----------
    mu      : (n,) membership grades µᵢ ∈ [0, 1].
    weights : (n,) normalised weights.
    lam     : Power parameter λ ≥ 1.
              λ=1 → arithmetic mean;  λ→∞ → maximum.

    Returns
    -------
    float : Yager aggregate score E_λ ∈ [0, 1].
    """
    mu      = np.asarray(mu, float)
    weights = np.asarray(weights, float)
    if lam < 1.0:
        raise ValueError(f"λ must be ≥ 1; got {lam}.")
    return float((np.dot(weights, mu ** lam)) ** (1.0 / lam))


def yager_aggregate_interval(
    mu_lo: np.ndarray,
    mu_hi: np.ndarray,
    weights: np.ndarray,
    lam: float,
) -> tuple:
    """
    IT2 Yager aggregate interval [E_λ^{lo}, E_λ^{hi}].

    E_λ^{lo} = (Σ wᵢ µ_lo_i^λ)^{1/λ}
    E_λ^{hi} = (Σ wᵢ µ_hi_i^λ)^{1/λ}
    """
    E_lo = yager_aggregate(mu_lo, weights, lam)
    E_hi = yager_aggregate(mu_hi, weights, lam)
    return E_lo, E_hi


# ---------------------------------------------------------------------------
# Theorem 6 — Certified O(1) width for Yager λ-aggregation
# ---------------------------------------------------------------------------

def yager_width_bound(
    k: np.ndarray,
    eps_c: float,
    M: float,
    eps_k: float,
    lam: float,
) -> dict:
    """
    Certified O(1) width for Yager λ-aggregation (Theorem 6):

        Δ(Ã^λ_n) ≤ λ^{1/λ} · (K')^{1/λ}

    where K' = max_i { kᵢ/4 · ϵ_c + M/4 · ϵ_k }.

    Proof steps (as in paper):
      Step 1: Subadditivity of f(t) = t^{1/λ} gives A^{1/λ} - B^{1/λ} ≤ (A-B)^{1/λ}.
      Step 2: Mean value theorem for g(t) = t^λ gives A - B ≤ λ·K'.
      Step 3: Assembly gives λ^{1/λ}·(K')^{1/λ}.

    Parameters
    ----------
    k       : (n,) steepness parameters.
    eps_c   : Centre uncertainty.
    M       : Domain bound.
    eps_k   : Steepness uncertainty.
    lam     : Yager parameter λ ≥ 1.

    Returns
    -------
    dict with 'K_prime', 'lam_factor', 'width_bound'.
    """
    if lam < 1.0:
        raise ValueError(f"λ must be ≥ 1; got {lam}.")

    s       = per_source_sensitivity(k, eps_c, M, eps_k)
    K_prime = float(s.max())

    lam_factor  = lam ** (1.0 / lam)
    width_bound = lam_factor * (K_prime ** (1.0 / lam))

    return {
        "K_prime"    : K_prime,
        "lam_factor" : lam_factor,
        "width_bound": width_bound,
        "lambda"     : lam,
        "is_O1"      : True,   # K' < ∞ by Assumption 1; bound is finite
    }


def lambda_factor_curve(
    lam_range: np.ndarray,
) -> np.ndarray:
    """
    Compute the inflation factor λ^{1/λ} over a range of λ values.

    Key values:
      λ = 1   → factor = 1.000  (recovers linear case)
      λ = e   → factor = e^{1/e} ≈ 1.445  (maximum)
      λ → ∞  → factor → 1.000  (converges back to 1)

    Parameters
    ----------
    lam_range : 1-D array of λ values ≥ 1.

    Returns
    -------
    np.ndarray : λ^{1/λ} values.
    """
    lam_range = np.asarray(lam_range, float)
    return lam_range ** (1.0 / lam_range)


# ---------------------------------------------------------------------------
# Proposition 4 — Yager bounds (Beliakov et al. 2007)
# ---------------------------------------------------------------------------

def verify_yager_bounds(
    mu: np.ndarray,
    weights: np.ndarray,
    lam: float,
) -> dict:
    """
    Verify Proposition 4: for all λ ≥ 1,  min µᵢ ≤ E_λ ≤ max µᵢ.
    """
    E_lam = yager_aggregate(mu, weights, lam)
    return {
        "E_lambda"          : E_lam,
        "min_mu"            : float(mu.min()),
        "max_mu"            : float(mu.max()),
        "lower_bound_holds" : mu.min() <= E_lam + 1e-10,
        "upper_bound_holds" : E_lam    <= mu.max() + 1e-10,
    }
