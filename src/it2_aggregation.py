"""
it2_aggregation.py
==================
Core module for bounded uncertainty propagation in normalised
interval Type-2 (IT2) fuzzy aggregation.

Implements:
  - Sigmoid membership functions and their derivatives (Lemma 2.8)
  - Lipschitz constant computation (Lemma 2.9)
  - Corner evaluation for interval endpoints (Theorem 3.6)
  - Width certificates for uniform and general weights (Theorem 3.11)
  - Yager lambda-aggregation (Theorem 3.4)
  - Width-optimal weight allocation (Theorem 3.9)
  - Sub-Gaussian concentration bounds (Theorem 3.21)
  - Power-law uncertainty decay (Theorem 3.13)

Reference:
  Ramesh & Mehreen, "Bounded uncertainty propagation in normalised
  interval Type-2 fuzzy aggregation", 2025.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class IT2Source:
    """
    Single IT2 fuzzy source with sigmoid membership function.

    Parameters
    ----------
    tau : float
        Input value.
    c : float
        Nominal centre parameter.
    k : float
        Nominal steepness parameter (k > 0).
    eps_c : float
        Half-width of centre uncertainty interval  |delta_c| <= eps_c.
    eps_k : float
        Half-width of steepness uncertainty interval |delta_k| <= eps_k.
    label : str, optional
        Human-readable source name.
    """
    tau: float
    c: float
    k: float
    eps_c: float
    eps_k: float
    label: str = ""

    def __post_init__(self):
        if self.k <= 0:
            raise ValueError("Steepness k must be positive.")
        if self.eps_c < 0 or self.eps_k < 0:
            raise ValueError("Uncertainty half-widths must be non-negative.")


@dataclass
class IT2AggregationResult:
    """Output from :func:`aggregate_it2`."""
    E_lower: float          # Lower bound of aggregation interval
    E_upper: float          # Upper bound of aggregation interval
    width_empirical: float  # E_upper - E_lower (empirical)
    width_certified: float  # Closed-form upper bound on width (Theorem 3.8)
    E_type1: float          # Type-1 aggregate at nominal parameters
    source_contributions: np.ndarray  # Per-source certified half-widths
    weights: np.ndarray


# ---------------------------------------------------------------------------
# Membership function
# ---------------------------------------------------------------------------

def sigmoid(tau: float, c: float, k: float) -> float:
    """
    Sigmoid membership function (Definition 2.7).

    mu(tau; c, k) = 1 / (1 + exp(-k * (tau - c)))
    """
    return 1.0 / (1.0 + np.exp(-k * (tau - c)))


def sigmoid_array(tau: np.ndarray, c: float, k: float) -> np.ndarray:
    """Vectorised sigmoid over an array of inputs."""
    return 1.0 / (1.0 + np.exp(-k * (tau - c)))


def sigmoid_derivatives(tau: float, c: float, k: float
                        ) -> Tuple[float, float, float]:
    """
    Partial derivatives of sigmoid (Lemma 2.8).

    Returns
    -------
    (dmu/dtau, dmu/dc, dmu/dk)
    """
    mu = sigmoid(tau, c, k)
    common = mu * (1.0 - mu)
    return k * common, -k * common, (tau - c) * common


def lipschitz_constant(k: float) -> float:
    """
    Lipschitz constant of sigmoid w.r.t. tau (Lemma 2.9).

    L = k / 4  (sharp, attained at tau = c)
    """
    return k / 4.0


# ---------------------------------------------------------------------------
# Corner evaluation
# ---------------------------------------------------------------------------

def _membership_bounds_single(
    tau: float,
    c: float,
    k: float,
    eps_c: float,
    eps_k: float,
) -> Tuple[float, float]:
    """
    Compute [mu_lower, mu_upper] for a single source by evaluating
    all four parameter rectangle corners (Theorem 3.6 / Proposition 3.3).

    Returns
    -------
    (mu_lower, mu_upper)
    """
    corners = [
        sigmoid(tau, c + dc, k + dk)
        for dc in (-eps_c, +eps_c)
        for dk in (-eps_k, +eps_k)
    ]
    return min(corners), max(corners)


def corner_evaluate(sources: list[IT2Source]
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate lower and upper membership grades for all sources via
    corner evaluation (O(4n) procedure, Moore 1966; Theorem 3.6).

    Returns
    -------
    mu_lower, mu_upper : np.ndarray of shape (n,)
    """
    mu_lo = np.empty(len(sources))
    mu_hi = np.empty(len(sources))
    for i, src in enumerate(sources):
        mu_lo[i], mu_hi[i] = _membership_bounds_single(
            src.tau, src.c, src.k, src.eps_c, src.eps_k
        )
    return mu_lo, mu_hi


# ---------------------------------------------------------------------------
# Certified width bound
# ---------------------------------------------------------------------------

def certified_width_bound(
    sources: list[IT2Source],
    weights: np.ndarray,
    M: Optional[float] = None,
) -> Tuple[float, np.ndarray]:
    """
    Closed-form certified width bound from Theorem 3.8.

    bound = 2 * sum_i  w_i * (k_i * eps_c_i / 4 + M * eps_k_i / 4)

    Parameters
    ----------
    sources : list of IT2Source
    weights : normalised weight vector (must sum to 1)
    M : supremum of |tau_i - c_i|; computed from data if None.

    Returns
    -------
    bound : float
    per_source : np.ndarray  (per-source contribution w_i * s_i * 2)
    """
    weights = np.asarray(weights, dtype=float)
    _check_normalised(weights)

    if M is None:
        M = max(abs(src.tau - src.c) for src in sources)

    s = np.array([
        src.k * src.eps_c / 4.0 + M * src.eps_k / 4.0
        for src in sources
    ])
    per_source = 2.0 * weights * s
    return float(per_source.sum()), per_source


def certified_width_uniform(
    sources: list[IT2Source],
    M: Optional[float] = None,
) -> float:
    """
    Theorem 3.11(i) — certified width under uniform weights wi = 1/n.

    bound = kavg * eps_c / 2 + M * eps_k / 2

    Assumes all sources share the same eps_c, eps_k.  For per-source
    uncertainties use :func:`certified_width_bound` directly.
    """
    n = len(sources)
    w = np.ones(n) / n
    return certified_width_bound(sources, w, M)[0]


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_it2(
    sources: list[IT2Source],
    weights: Optional[np.ndarray] = None,
    M: Optional[float] = None,
) -> IT2AggregationResult:
    """
    Compute the IT2 aggregation interval and its certified width.

    Parameters
    ----------
    sources : list of IT2Source
    weights : normalised weights; defaults to uniform 1/n.
    M : domain bound; inferred from data if None.

    Returns
    -------
    IT2AggregationResult
    """
    n = len(sources)
    if weights is None:
        weights = np.ones(n) / n
    weights = np.asarray(weights, dtype=float)
    _check_normalised(weights)

    # Type-1 aggregate at nominal parameters
    mu_nom = np.array([sigmoid(src.tau, src.c, src.k) for src in sources])
    E_type1 = float(weights @ mu_nom)

    # IT2 interval via corner evaluation
    mu_lo, mu_hi = corner_evaluate(sources)
    E_lower = float(weights @ mu_lo)
    E_upper = float(weights @ mu_hi)

    if M is None:
        M = max(abs(src.tau - src.c) for src in sources)

    bound, per_source = certified_width_bound(sources, weights, M)

    return IT2AggregationResult(
        E_lower=E_lower,
        E_upper=E_upper,
        width_empirical=E_upper - E_lower,
        width_certified=bound,
        E_type1=E_type1,
        source_contributions=per_source,
        weights=weights,
    )


# ---------------------------------------------------------------------------
# Yager lambda-aggregation
# ---------------------------------------------------------------------------

def aggregate_yager(
    sources: list[IT2Source],
    weights: np.ndarray,
    lam: float = 2.0,
    M: Optional[float] = None,
) -> Tuple[float, float, float]:
    """
    Yager lambda-aggregation IT2 interval and certified width
    (Theorem 3.4).

    E_lam = (sum_i w_i * mu_i^lam)^(1/lam)

    Returns
    -------
    E_lower, E_upper, width_certified_bound
    """
    if lam < 1:
        raise ValueError("Lambda must be >= 1 for Yager power mean.")
    weights = np.asarray(weights, dtype=float)
    _check_normalised(weights)

    mu_lo, mu_hi = corner_evaluate(sources)
    E_lower = float((weights @ mu_lo ** lam) ** (1.0 / lam))
    E_upper = float((weights @ mu_hi ** lam) ** (1.0 / lam))

    # Certified bound: lambda^(1/lambda) * (2*K')^(1/lambda)
    if M is None:
        M = max(abs(src.tau - src.c) for src in sources)
    K_prime = max(
        src.k * src.eps_c / 4.0 + M * src.eps_k / 4.0 for src in sources
    )
    bound = (lam ** (1.0 / lam)) * ((2.0 * K_prime) ** (1.0 / lam))
    return E_lower, E_upper, bound


# ---------------------------------------------------------------------------
# Width-optimal weight allocation  (Theorem 3.9)
# ---------------------------------------------------------------------------

def sensitivity_coefficients(
    sources: list[IT2Source],
    M: float,
) -> np.ndarray:
    """
    Per-source one-sided sensitivity coefficients s_i = k_i*eps_c/4 + M*eps_k/4.
    """
    return np.array([
        src.k * src.eps_c / 4.0 + M * src.eps_k / 4.0 for src in sources
    ])


def width_optimal_weights(
    sources: list[IT2Source],
    M: float,
    alpha: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float]:
    """
    Minimum certified-width weight allocation (Theorem 3.9).

    Parameters
    ----------
    sources  : list of IT2Source
    M        : domain bound
    alpha    : optional floor weights (must sum to < 1)

    Returns
    -------
    w_opt : np.ndarray  (degenerate or constrained minimiser)
    min_width : float
    """
    n = len(sources)
    s = sensitivity_coefficients(sources, M)
    j_star = int(np.argmin(s))

    if alpha is None:
        w_opt = np.zeros(n)
        w_opt[j_star] = 1.0
    else:
        alpha = np.asarray(alpha, dtype=float)
        if alpha.sum() >= 1.0:
            raise ValueError("Floor weights must sum to strictly less than 1.")
        sigma = 1.0 - alpha.sum()
        w_opt = alpha.copy()
        w_opt[j_star] += sigma

    min_width = 2.0 * float(w_opt @ s)
    return w_opt, min_width


# ---------------------------------------------------------------------------
# Sub-Gaussian concentration bound  (Theorem 3.21)
# ---------------------------------------------------------------------------

def bernstein_concentration_bound(
    sources: list[IT2Source],
    weights: np.ndarray,
    sigma2_i: np.ndarray,
    delta: float = 0.05,
    M: Optional[float] = None,
) -> Tuple[float, float]:
    """
    Bernstein-type sub-Gaussian bound (Theorem 3.21).

    Returns
    -------
    stat_bound  : sampling deviation bound (holds w.p. >= 1-delta)
    total_bound : stat_bound + half certified width (combined bound)
    """
    weights = np.asarray(weights, dtype=float)
    _check_normalised(weights)

    sigma2_w = float(np.sum(weights ** 2 * sigma2_i))
    B_w = float(np.max(weights))  # range of each term is w_i

    log2d = np.log(2.0 / delta)
    stat_bound = np.sqrt(2.0 * sigma2_w * log2d) + B_w * log2d / 3.0

    if M is None:
        M = max(abs(src.tau - src.c) for src in sources)
    _, per_source = certified_width_bound(sources, weights, M)
    half_cert = per_source.sum() / 2.0  # half-width

    return float(stat_bound), float(stat_bound + half_cert)


# ---------------------------------------------------------------------------
# Power-law uncertainty decay  (Theorem 3.13)
# ---------------------------------------------------------------------------

def width_powerlaw_decay(
    k_avg: float,
    eps_c0: float,
    eps_k0: float,
    M: float,
    n: int,
    beta: float = 0.0,
    gamma: float = 0.0,
    concentrated_domain: bool = False,
    M0: Optional[float] = None,
) -> float:
    """
    Certified width under power-law parameter uncertainty decay
    (Theorem 3.13).

    eps_c(n) = eps_c0 * n^{-beta},  eps_k(n) = eps_k0 * n^{-gamma}

    Parameters
    ----------
    concentrated_domain : if True, applies Assumption 2.16 (M <= M0/n)
    M0 : required when concentrated_domain=True
    """
    eps_c_n = eps_c0 * n ** (-beta)
    eps_k_n = eps_k0 * n ** (-gamma)

    if concentrated_domain:
        if M0 is None:
            raise ValueError("M0 must be supplied for concentrated domain.")
        return k_avg * eps_c_n / 2.0 + M0 * eps_k_n / (2.0 * n)
    else:
        return k_avg * eps_c_n / 2.0 + M * eps_k_n / 2.0


# ---------------------------------------------------------------------------
# Decision certification
# ---------------------------------------------------------------------------

def certify_decision(
    result: IT2AggregationResult,
    threshold: float,
) -> str:
    """
    Classify aggregate score relative to threshold.

    Returns
    -------
    'certified_accept'  if E_lower > threshold
    'certified_reject'  if E_upper < threshold
    'indeterminate'     otherwise
    """
    if result.E_lower > threshold:
        return "certified_accept"
    elif result.E_upper < threshold:
        return "certified_reject"
    return "indeterminate"


def certify_ranking(
    result_a: IT2AggregationResult,
    result_b: IT2AggregationResult,
) -> str:
    """
    Attempt to certify a ranking between two alternatives (Remark 4.2).

    Returns
    -------
    'A > B', 'B > A', or 'indeterminate'
    """
    if result_a.E_lower > result_b.E_upper:
        return "A > B"
    elif result_b.E_lower > result_a.E_upper:
        return "B > A"
    return "indeterminate"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_normalised(w: np.ndarray, tol: float = 1e-9) -> None:
    if abs(w.sum() - 1.0) > tol:
        raise ValueError(f"Weights must sum to 1.  Got {w.sum():.6f}.")
    if np.any(w < 0):
        raise ValueError("All weights must be non-negative.")
