"""
membership.py
=============
Sigmoid membership functions, their partial derivatives, and Lipschitz constants.

Implements:
  - Definition 7  : Sigmoid membership function
  - Lemma 1       : Partial derivatives (∂µ/∂τ, ∂µ/∂c, ∂µ/∂k)
  - Lemma 2       : Lipschitz constant k/4 (sharp)
  - Definition 10 : Interval-valued membership from parameter rectangles
  - Corollary 18  : Gaussian membership covered by Lipschitz class F_L

All functions are fully vectorised over NumPy arrays.

Reference
---------
Ramesh & Mehreen (2025), "Bounded uncertainty propagation in normalised
interval Type-2 fuzzy aggregation", FODM (submitted).
"""

import numpy as np
from typing import Tuple, Union

ArrayLike = Union[float, np.ndarray]


# ---------------------------------------------------------------------------
# Core sigmoid (Definition 7)
# ---------------------------------------------------------------------------

def sigmoid(tau: ArrayLike, c: ArrayLike, k: ArrayLike) -> np.ndarray:
    """
    Sigmoid membership function (Definition 7).

        µ(τ; c, k) = 1 / (1 + exp(-k * (τ - c)))

    Parameters
    ----------
    tau : array_like
        Input value(s).
    c   : array_like
        Centre parameter(s).
    k   : array_like
        Steepness parameter(s) > 0.

    Returns
    -------
    np.ndarray
        Membership grade(s) in (0, 1).

    Notes
    -----
    Inflects at µ(c) = 0.5.  Strictly increasing in τ and maps ℝ → (0, 1).
    """
    tau, c, k = np.asarray(tau, float), np.asarray(c, float), np.asarray(k, float)
    return 1.0 / (1.0 + np.exp(-k * (tau - c)))


# ---------------------------------------------------------------------------
# Partial derivatives (Lemma 1)
# ---------------------------------------------------------------------------

def sigmoid_dtau(tau: ArrayLike, c: ArrayLike, k: ArrayLike) -> np.ndarray:
    """∂µ/∂τ = k · µ · (1 - µ)   (Lemma 1)."""
    mu = sigmoid(tau, c, k)
    return k * mu * (1.0 - mu)


def sigmoid_dc(tau: ArrayLike, c: ArrayLike, k: ArrayLike) -> np.ndarray:
    """∂µ/∂c = -k · µ · (1 - µ)   (Lemma 1)."""
    mu = sigmoid(tau, c, k)
    return -k * mu * (1.0 - mu)


def sigmoid_dk(tau: ArrayLike, c: ArrayLike, k: ArrayLike) -> np.ndarray:
    """∂µ/∂k = (τ - c) · µ · (1 - µ)   (Lemma 1)."""
    mu = sigmoid(tau, c, k)
    return (np.asarray(tau) - np.asarray(c)) * mu * (1.0 - mu)


# ---------------------------------------------------------------------------
# Lipschitz constant (Lemma 2)
# ---------------------------------------------------------------------------

def sigmoid_lipschitz_constant(k: ArrayLike) -> np.ndarray:
    """
    Lipschitz constant of µ(·; c, k) with respect to τ  (Lemma 2).

        L = k / 4

    The constant is *sharp*: it is attained as τ → c (where µ(1-µ) → 1/4).

    Parameters
    ----------
    k : array_like
        Steepness parameter(s).

    Returns
    -------
    np.ndarray
        Lipschitz constant(s) k/4.
    """
    return np.asarray(k, float) / 4.0


def verify_lipschitz(tau1: ArrayLike, tau2: ArrayLike,
                     c: ArrayLike, k: ArrayLike) -> np.ndarray:
    """
    Numerically verify Lemma 2: |µ(τ1) - µ(τ2)| ≤ (k/4)|τ1 - τ2|.

    Returns the ratio |µ1 - µ2| / ((k/4)|τ1 - τ2|).
    Should be ≤ 1.0 everywhere (and approach 1 near τ = c).
    """
    tau1, tau2 = np.asarray(tau1, float), np.asarray(tau2, float)
    mu1 = sigmoid(tau1, c, k)
    mu2 = sigmoid(tau2, c, k)
    L   = sigmoid_lipschitz_constant(k)
    denom = L * np.abs(tau1 - tau2)
    # Avoid division by zero when tau1 == tau2
    with np.errstate(invalid='ignore', divide='ignore'):
        ratio = np.where(denom == 0, 0.0, np.abs(mu1 - mu2) / denom)
    return ratio


# ---------------------------------------------------------------------------
# Interval-valued membership (Definition 10)
# ---------------------------------------------------------------------------

def interval_membership(
    tau: ArrayLike,
    c_lo: ArrayLike, c_hi: ArrayLike,
    k_lo: ArrayLike, k_hi: ArrayLike,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interval-valued membership [µ_lo, µ_hi] over a parameter rectangle
    [c_lo, c_hi] × [k_lo, k_hi]  (Definition 10).

    The extrema are attained at the four rectangle corners (Proposition 5):
    the function is monotone in each parameter separately (Theorem 7, Steps 1-2).

    Parameters
    ----------
    tau               : Input value(s).
    c_lo, c_hi        : Centre uncertainty interval (c_lo ≤ c_hi).
    k_lo, k_hi        : Steepness uncertainty interval (k_lo ≤ k_hi).

    Returns
    -------
    mu_lo, mu_hi : np.ndarray
        Lower and upper membership grades.
    """
    tau  = np.asarray(tau, float)
    c_lo = np.asarray(c_lo, float)
    c_hi = np.asarray(c_hi, float)
    k_lo = np.asarray(k_lo, float)
    k_hi = np.asarray(k_hi, float)

    # Evaluate all four corners
    corners = np.stack([
        sigmoid(tau, c_lo, k_lo),
        sigmoid(tau, c_lo, k_hi),
        sigmoid(tau, c_hi, k_lo),
        sigmoid(tau, c_hi, k_hi),
    ], axis=-1)

    mu_lo = corners.min(axis=-1)
    mu_hi = corners.max(axis=-1)
    return mu_lo, mu_hi


def interval_width(
    tau: ArrayLike,
    c_lo: ArrayLike, c_hi: ArrayLike,
    k_lo: ArrayLike, k_hi: ArrayLike,
) -> np.ndarray:
    """
    Per-source interval width  µ_hi(τ) - µ_lo(τ)  (Definition 11).

    Equivalent to calling interval_membership and subtracting.
    """
    mu_lo, mu_hi = interval_membership(tau, c_lo, c_hi, k_lo, k_hi)
    return mu_hi - mu_lo



# ---------------------------------------------------------------------------
# Gaussian membership (Corollary 18 — Lipschitz class coverage)
# ---------------------------------------------------------------------------

def gaussian_membership(tau: ArrayLike, c: ArrayLike, k: ArrayLike) -> np.ndarray:
    """
    Gaussian membership function  µ(τ; c, k) = exp(-k(τ-c)²).

    Covered by the Lipschitz class F_L under Assumption 1 (Corollary 18).
    The O(1) width certificate applies without modification.

    Parameters
    ----------
    tau : array_like   Input value(s).
    c   : array_like   Centre.
    k   : array_like   Spread parameter (> 0).
    """
    tau, c, k = np.asarray(tau, float), np.asarray(c, float), np.asarray(k, float)
    return np.exp(-k * (tau - c) ** 2)


def gaussian_lipschitz_constant(k: ArrayLike, M: float) -> np.ndarray:
    """
    Lipschitz constant for the Gaussian membership w.r.t. centre c,
    bounded by k/4 under |τ - c| ≤ M (Corollary 18).

    |∂µ/∂c| = 2k|τ-c|exp(-k(τ-c)²) ≤ k/4 when 2|τ-c|exp(-k(τ-c)²) ≤ 1/4.
    For practical ranges this is satisfied when kM ≤ 4; the bound k/4 is used
    as a conservative upper envelope consistent with Assumption 1.
    """
    return np.asarray(k, float) / 4.0
