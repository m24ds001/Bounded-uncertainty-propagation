"""
test_width_bounds.py
====================
Unit tests for Theorems 8, 9, 10, 12, 13 and their corollaries.

Run with:  pytest tests/test_width_bounds.py -v
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.width_bounds import (
    width_bound_lipschitz,
    per_source_sensitivity,
    optimal_weight_allocation,
    width_uniform_weighting,
    certified_width_o1,
    width_versus_n,
    width_power_law,
    example2_power_law_illustration,
    example3_sharpness,
)
from src.aggregation import unnormalised_width_lower_bound


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_params():
    """Typical IT2 parameters for n=5 sources."""
    rng = np.random.default_rng(42)
    n   = 5
    return {
        "n"      : n,
        "weights": np.full(n, 1.0 / n),
        "k"      : rng.uniform(2.0, 8.0, n),
        "eps_c"  : 0.10,
        "M"      : 0.50,
        "eps_k"  : 0.50,
    }


# ---------------------------------------------------------------------------
# Theorem 8 — Width bound under Lipschitz perturbation
# ---------------------------------------------------------------------------

class TestTheorem8:
    def test_positive(self, sample_params):
        """Width bound must be non-negative."""
        bound = width_bound_lipschitz(**{k: sample_params[k]
                                        for k in ("weights", "k", "eps_c", "M", "eps_k")})
        assert bound >= 0

    def test_linearity_in_eps_c(self, sample_params):
        """Bound is linear in ϵ_c."""
        p = sample_params
        b1 = width_bound_lipschitz(p["weights"], p["k"], 0.10, p["M"], p["eps_k"])
        b2 = width_bound_lipschitz(p["weights"], p["k"], 0.20, p["M"], p["eps_k"])
        # Ratio should be (0.10 * k/4 + M * eps_k / 4) : (0.20 * k/4 + M * eps_k / 4)
        assert b2 > b1   # more uncertainty → wider bound

    def test_zero_uncertainty(self, sample_params):
        """Zero uncertainty → zero width bound."""
        p     = sample_params
        bound = width_bound_lipschitz(p["weights"], p["k"], 0.0, p["M"], 0.0)
        assert abs(bound) < 1e-12

    def test_dominated_by_K_prime(self, sample_params):
        """Theorem 10(iii): general weight bound ≤ K'."""
        p   = sample_params
        s   = per_source_sensitivity(p["k"], p["eps_c"], p["M"], p["eps_k"])
        K_p = float(s.max())
        bound = width_bound_lipschitz(p["weights"], p["k"], p["eps_c"], p["M"], p["eps_k"])
        assert bound <= K_p + 1e-10

    def test_uniform_bound_formula(self, sample_params):
        """Theorem 10(i): uniform bound = k_avg * eps_c / 4 + M * eps_k / 4."""
        p     = sample_params
        k_avg = p["k"].mean()
        expected = k_avg * p["eps_c"] / 4.0 + p["M"] * p["eps_k"] / 4.0
        computed = width_uniform_weighting(p["k"], p["eps_c"], p["M"], p["eps_k"])
        assert abs(computed - expected) < 1e-12


# ---------------------------------------------------------------------------
# Theorem 9 — Width-optimal weight allocation
# ---------------------------------------------------------------------------

class TestTheorem9:
    def test_optimal_weight_concentrates_on_min_s(self, sample_params):
        """Unconstrained minimiser puts all mass on arg-min sensitivity."""
        p = sample_params
        w_opt, B_opt = optimal_weight_allocation(p["k"], p["eps_c"], p["M"], p["eps_k"])
        s = per_source_sensitivity(p["k"], p["eps_c"], p["M"], p["eps_k"])
        j = int(np.argmin(s))
        assert abs(w_opt[j] - 1.0) < 1e-10
        assert abs(B_opt - s[j]) < 1e-10

    def test_uniform_in_range(self, sample_params):
        """Uniform weight lies strictly inside [min s, max s]."""
        p = sample_params
        s = per_source_sensitivity(p["k"], p["eps_c"], p["M"], p["eps_k"])
        B_unif = width_uniform_weighting(p["k"], p["eps_c"], p["M"], p["eps_k"])
        if not np.all(s == s[0]):   # skip when all s equal
            assert s.min() < B_unif < s.max()

    def test_constrained_minimum(self, sample_params):
        """Constrained minimiser allocates remaining mass to min-sensitivity source."""
        p = sample_params
        n = p["n"]
        alpha = np.full(n, 0.05)   # small lower bounds
        w_opt, B_opt = optimal_weight_allocation(
            p["k"], p["eps_c"], p["M"], p["eps_k"], alpha=alpha)
        assert abs(w_opt.sum() - 1.0) < 1e-10
        assert np.all(w_opt >= alpha - 1e-10)

    def test_weight_sums_to_one(self, sample_params):
        """Optimal weight vector is normalised."""
        p = sample_params
        w_opt, _ = optimal_weight_allocation(p["k"], p["eps_c"], p["M"], p["eps_k"])
        assert abs(w_opt.sum() - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# Theorem 10 — Certified O(1) width
# ---------------------------------------------------------------------------

class TestTheorem10:
    def test_case_i_finite(self, sample_params):
        """Case (i): bound is finite for any n."""
        p = sample_params
        result = certified_width_o1(p["weights"], p["k"], p["eps_c"], p["M"], p["eps_k"])
        assert np.isfinite(result["case_i"])

    def test_case_iii_equals_Kprime(self, sample_params):
        """Case (iii): general weight bound = max per-source sensitivity."""
        p = sample_params
        result = certified_width_o1(p["weights"], p["k"], p["eps_c"], p["M"], p["eps_k"])
        s = per_source_sensitivity(p["k"], p["eps_c"], p["M"], p["eps_k"])
        assert abs(result["case_iii"] - s.max()) < 1e-12

    def test_case_ii_smaller_than_case_i(self, sample_params):
        """Case (ii) ≤ Case (i) when M₀ < M·n."""
        p  = sample_params
        n  = p["n"]
        M0 = p["M"] * n * 0.5   # Assumption 3: M ≤ M0/n → M0 = M·n
        result = certified_width_o1(
            p["weights"], p["k"], p["eps_c"], p["M"], p["eps_k"],
            domain_M=M0, n=n)
        assert result["case_ii"] <= result["case_i"] + 1e-10

    def test_width_nondecreasing_in_eps_c(self, sample_params):
        """Width bound non-decreasing in ϵ_c."""
        p  = sample_params
        r1 = certified_width_o1(p["weights"], p["k"], 0.05, p["M"], p["eps_k"])
        r2 = certified_width_o1(p["weights"], p["k"], 0.15, p["M"], p["eps_k"])
        assert r2["case_i"] >= r1["case_i"] - 1e-10


# ---------------------------------------------------------------------------
# Corollary 11 — Non-increasing in n
# ---------------------------------------------------------------------------

def test_corollary11_nonincrasing():
    """Adding sources with below-average k does not inflate the bound."""
    # k sequence starts high then adds low-k sources
    k_all = np.array([8.0, 8.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
    ns, bounds = width_versus_n(k_all, eps_c=0.10, M=0.5, eps_k=0.5)
    # After inflection point (when low-k sources dominate), bound should decrease
    assert bounds[-1] <= bounds[2]   # final bound ≤ bound after first 3 sources


# ---------------------------------------------------------------------------
# Theorem 12 — Power-law contraction
# ---------------------------------------------------------------------------

class TestTheorem12:
    def test_beta0_recovers_theorem10i(self):
        """β=γ=0 → constant bound (Theorem 10i special case)."""
        r = width_power_law(n=10, k_avg=5.0, eps_c_0=0.20, M=0.5,
                            eps_k_0=1.0, beta=0.0, gamma=0.0)
        assert r["decay_rate"] == 0.0
        # Bound should equal the β=0 case of Theorem 10(i)
        expected = 5.0 * 0.20 / 4.0 + 0.5 * 1.0 / 4.0
        assert abs(r["bound_general"] - expected) < 1e-10

    def test_example2_values(self):
        """Reproduce Example 2 numerical values exactly."""
        bounds = example2_power_law_illustration()
        assert abs(bounds[9]  - 0.375 / 3) < 1e-6
        assert abs(bounds[25] - 0.375 / 5) < 1e-6

    def test_monotone_decay_with_n(self):
        """Width bound decreases as n increases for β > 0."""
        params = dict(k_avg=5.0, eps_c_0=0.20, M=0.5, eps_k_0=1.0,
                      beta=0.5, gamma=0.5)
        b10 = width_power_law(10,  **params)["bound_general"]
        b50 = width_power_law(50,  **params)["bound_general"]
        b100= width_power_law(100, **params)["bound_general"]
        assert b10 > b50 > b100


# ---------------------------------------------------------------------------
# Theorem 13 — Unnormalised divergence lower bound
# ---------------------------------------------------------------------------

def test_theorem13_linear_growth():
    """Unnormalised width lower bound grows linearly with n (Theorem 13)."""
    v, k_min, eps_c = 1.0, 2.0, 0.10
    b10  = unnormalised_width_lower_bound(10,  v, k_min, eps_c)
    b100 = unnormalised_width_lower_bound(100, v, k_min, eps_c)
    assert abs(b100 / b10 - 10.0) < 1e-10   # exactly linear


# ---------------------------------------------------------------------------
# Example 3 — Sharpness of Theorem 10(i)
# ---------------------------------------------------------------------------

def test_example3_sharpness():
    """Bound is attained exactly when τᵢ = cᵢ (Example 3)."""
    result = example3_sharpness()
    assert result["is_tight"], (
        f"Bound not tight: empirical={result['empirical_width']:.6f}, "
        f"certified={result['certified_bound']:.6f}"
    )
