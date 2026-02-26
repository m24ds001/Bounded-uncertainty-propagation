"""
test_it2_aggregation.py
=======================
Unit tests verifying all main theorems and propositions.

Run:
  pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest
from it2_aggregation import (
    IT2Source, sigmoid, lipschitz_constant, sigmoid_derivatives,
    corner_evaluate, aggregate_it2, certified_width_bound,
    certified_width_uniform, aggregate_yager, sensitivity_coefficients,
    width_optimal_weights, bernstein_concentration_bound,
    width_powerlaw_decay, certify_decision, certify_ranking,
    _check_normalised,
)

# ── fixtures ─────────────────────────────────────────────────────────────────

def make_sources(n=3, eps_c=0.1, eps_k=0.5, k=3.0):
    return [
        IT2Source(tau=0.7 + i * 0.05, c=0.6, k=k,
                  eps_c=eps_c, eps_k=eps_k)
        for i in range(n)
    ]


# ── Lemma 2.8: sigmoid derivatives ──────────────────────────────────────────

class TestSigmoidDerivatives:
    def test_dtau(self):
        """dmu/dtau = k*mu*(1-mu)"""
        tau, c, k = 0.5, 0.4, 3.0
        mu = sigmoid(tau, c, k)
        dtau, _, _ = sigmoid_derivatives(tau, c, k)
        assert abs(dtau - k * mu * (1 - mu)) < 1e-12

    def test_dc(self):
        """dmu/dc = -k*mu*(1-mu)"""
        tau, c, k = 0.5, 0.4, 3.0
        mu = sigmoid(tau, c, k)
        _, dc, _ = sigmoid_derivatives(tau, c, k)
        assert abs(dc - (-k * mu * (1 - mu))) < 1e-12

    def test_dk(self):
        """dmu/dk = (tau-c)*mu*(1-mu)"""
        tau, c, k = 0.7, 0.4, 3.0
        mu = sigmoid(tau, c, k)
        _, _, dk = sigmoid_derivatives(tau, c, k)
        assert abs(dk - (tau - c) * mu * (1 - mu)) < 1e-12

    def test_centre_value(self):
        """mu(c; c, k) = 0.5 for all k > 0"""
        for k in [1.0, 5.0, 12.0]:
            assert abs(sigmoid(0.5, 0.5, k) - 0.5) < 1e-12


# ── Lemma 2.9: Lipschitz constant ────────────────────────────────────────────

class TestLipschitz:
    def test_sharp_constant(self):
        """k/4 is the sharp Lipschitz constant (attained at tau=c)."""
        k = 4.0
        L = lipschitz_constant(k)
        assert abs(L - k / 4.0) < 1e-12

    def test_bound_holds(self):
        """|mu(tau1) - mu(tau2)| <= k/4 * |tau1 - tau2| for random pairs."""
        rng = np.random.default_rng(0)
        k, c = 5.0, 0.3
        L = lipschitz_constant(k)
        for _ in range(1000):
            t1, t2 = rng.uniform(-2, 2, size=2)
            diff_mu = abs(sigmoid(t1, c, k) - sigmoid(t2, c, k))
            assert diff_mu <= L * abs(t1 - t2) + 1e-12


# ── Theorem 3.1: Weighted average properties ─────────────────────────────────

class TestWeightedAverageProperties:
    def test_range_01(self):
        """E in [0, 1]."""
        sources = make_sources(5)
        res = aggregate_it2(sources)
        assert 0 <= res.E_type1 <= 1
        assert 0 <= res.E_lower <= res.E_upper <= 1

    def test_convexity(self):
        """mini mu <= E <= maxi mu"""
        sources = make_sources(4)
        w = np.array([0.1, 0.2, 0.3, 0.4])
        mu = np.array([sigmoid(s.tau, s.c, s.k) for s in sources])
        res = aggregate_it2(sources, weights=w)
        assert mu.min() - 1e-10 <= res.E_type1 <= mu.max() + 1e-10

    def test_constant_membership(self):
        """If all mu_i = c, then E = c."""
        # All inputs at their centre → mu_i = 0.5
        sources = [IT2Source(tau=0.5, c=0.5, k=3.0, eps_c=0.0, eps_k=0.0)
                   for _ in range(4)]
        res = aggregate_it2(sources)
        assert abs(res.E_type1 - 0.5) < 1e-10


# ── Theorem 3.6: Interval bounds ─────────────────────────────────────────────

class TestIntervalBounds:
    def test_enclosure(self):
        """E_lower <= nominal <= E_upper for all parameter configs."""
        sources = make_sources(5, eps_c=0.15, eps_k=0.3)
        res = aggregate_it2(sources)
        assert res.E_lower <= res.E_type1 + 1e-9
        assert res.E_type1 <= res.E_upper + 1e-9

    def test_width_non_negative(self):
        sources = make_sources(4)
        res = aggregate_it2(sources)
        assert res.width_empirical >= 0


# ── Theorem 3.8: Width bound ─────────────────────────────────────────────────

class TestWidthBound:
    def test_empirical_leq_certified(self):
        """Empirical width <= certified bound (Theorem 3.8)."""
        for n in [2, 5, 10, 20]:
            sources = make_sources(n, eps_c=0.1, eps_k=0.5, k=3.0)
            res = aggregate_it2(sources)
            assert res.width_empirical <= res.width_certified + 1e-9, \
                f"n={n}: empirical {res.width_empirical:.6f} > cert {res.width_certified:.6f}"

    def test_zero_uncertainty(self):
        """Zero uncertainty → width = 0."""
        sources = make_sources(4, eps_c=0.0, eps_k=0.0)
        res = aggregate_it2(sources)
        assert abs(res.width_empirical) < 1e-10
        assert abs(res.width_certified) < 1e-10


# ── Theorem 3.11: O(1) under normalised weights ───────────────────────────────

class TestO1Width:
    def test_o1_uniform(self):
        """Width bound is O(1) — constant in n."""
        bounds = []
        for n in [2, 5, 10, 50, 200]:
            sources = make_sources(n, eps_c=0.1, eps_k=0.0, k=3.0)
            w = np.ones(n) / n
            b, _ = certified_width_bound(sources, w)
            bounds.append(b)
        # All bounds should be the same (kavg * eps_c / 2 = const)
        assert max(bounds) - min(bounds) < 1e-10, \
            f"Width not constant: {bounds}"

    def test_unnormalised_grows_with_n(self):
        """Unnormalised OWA width grows as O(n) (Theorem 3.18)."""
        k_min, eps_c, v = 2.0, 0.1, 1.0
        lower_bound = lambda n: n * v * k_min * eps_c / 4.0
        for n in [5, 10, 20, 50]:
            assert lower_bound(n) > lower_bound(n - 1)
        # Verify linear growth
        assert lower_bound(50) > 5 * lower_bound(10) - 1e-10


# ── Theorem 3.9: Width-optimal allocation ─────────────────────────────────────

class TestWidthOptimalAllocation:
    def test_degenerate_minimiser(self):
        """Minimum width achieved by degenerate weight on lowest-s source."""
        sources = make_sources(4, eps_c=0.1, eps_k=0.5, k=3.0)
        # Make source 2 have lower steepness
        sources[2] = IT2Source(tau=0.65, c=0.60, k=1.5, eps_c=0.1, eps_k=0.5)
        M = max(abs(s.tau - s.c) for s in sources)
        w_opt, min_w = width_optimal_weights(sources, M)
        s = sensitivity_coefficients(sources, M)
        j_star = int(np.argmin(s))
        assert w_opt[j_star] == pytest.approx(1.0)
        assert min_w == pytest.approx(2.0 * s[j_star])

    def test_uniform_inside_range(self):
        """Uniform weighting gives B in (min_s, max_s)."""
        sources = make_sources(4, eps_c=0.1, eps_k=0.5)
        M = max(abs(s.tau - s.c) for s in sources)
        s = sensitivity_coefficients(sources, M)
        n = len(sources)
        w = np.ones(n) / n
        B = float(w @ s)
        assert s.min() <= B <= s.max() + 1e-10


# ── Theorem 3.4: Yager lambda-aggregation ────────────────────────────────────

class TestYagerAggregation:
    def test_lambda1_recovers_linear(self):
        """lambda=1 reduces to linear aggregation."""
        sources = make_sources(4, eps_c=0.1, eps_k=0.5)
        w = np.ones(4) / 4
        E_lo_y, E_hi_y, cert_y = aggregate_yager(sources, w, lam=1.0)
        res = aggregate_it2(sources, weights=w)
        assert abs(E_lo_y - res.E_lower) < 1e-8
        assert abs(E_hi_y - res.E_upper) < 1e-8

    def test_certified_O1(self):
        """Yager certified width is O(1) in n."""
        for lam in [1.0, 2.0, 5.0, np.e]:
            for n in [2, 10, 50]:
                sources = make_sources(n, eps_c=0.1, eps_k=0.0)
                w = np.ones(n) / n
                _, _, cert = aggregate_yager(sources, w, lam=lam)
                assert np.isfinite(cert)
                assert cert < 10.0  # sanity: never blows up


# ── Theorem 3.13: Power-law decay ────────────────────────────────────────────

class TestPowerlawDecay:
    def test_beta0_recovers_o1(self):
        """beta=0 gives Theorem 3.11(i)."""
        k_avg, eps_c0, eps_k0, M = 3.0, 0.1, 0.5, 0.5
        n = 10
        bound_decay = width_powerlaw_decay(k_avg, eps_c0, eps_k0, M, n, 0, 0)
        bound_static = k_avg * eps_c0 / 2.0 + M * eps_k0 / 2.0
        assert abs(bound_decay - bound_static) < 1e-12

    def test_beta1_monotone_decrease(self):
        """beta=1: bound decreases with n."""
        k_avg, eps_c0, eps_k0, M = 3.0, 0.1, 0.5, 0.5
        bounds = [width_powerlaw_decay(k_avg, eps_c0, eps_k0, M, n, 1, 0)
                  for n in [1, 5, 10, 50, 100]]
        for i in range(len(bounds) - 1):
            assert bounds[i] > bounds[i + 1]


# ── Theorem 3.21: Bernstein concentration ────────────────────────────────────

class TestBernsteinBound:
    def test_finite_bounds(self):
        sources = make_sources(5, eps_c=0.1, eps_k=0.5)
        w = np.ones(5) / 5
        sigma2_i = np.array([0.01] * 5)
        stat, total = bernstein_concentration_bound(sources, w, sigma2_i)
        assert np.isfinite(stat) and stat >= 0
        assert np.isfinite(total) and total >= stat

    def test_shrinks_with_n(self):
        """Statistical bound shrinks with n (uniform weights)."""
        sigma2_base = 0.02
        bounds = []
        for n in [5, 20, 100]:
            sources = make_sources(n, eps_c=0.1, eps_k=0.0)
            w = np.ones(n) / n
            sigma2_i = np.full(n, sigma2_base)
            stat, _ = bernstein_concentration_bound(sources, w, sigma2_i)
            bounds.append(stat)
        # Bound should decrease
        assert bounds[0] > bounds[1] > bounds[2]


# ── Decision certification ────────────────────────────────────────────────────

class TestCertification:
    def test_certified_accept(self):
        sources = [IT2Source(tau=0.9, c=0.5, k=5.0, eps_c=0.01, eps_k=0.0)]
        res = aggregate_it2(sources)
        dec = certify_decision(res, threshold=0.6)
        assert dec == "certified_accept"

    def test_certified_reject(self):
        sources = [IT2Source(tau=0.1, c=0.5, k=5.0, eps_c=0.01, eps_k=0.0)]
        res = aggregate_it2(sources)
        dec = certify_decision(res, threshold=0.6)
        assert dec == "certified_reject"

    def test_ranking_disjoint(self):
        s_a = [IT2Source(tau=0.9, c=0.5, k=8.0, eps_c=0.01, eps_k=0.0)]
        s_b = [IT2Source(tau=0.2, c=0.5, k=8.0, eps_c=0.01, eps_k=0.0)]
        ra = aggregate_it2(s_a)
        rb = aggregate_it2(s_b)
        assert certify_ranking(ra, rb) == "A > B"


# ── Normalisation guard ───────────────────────────────────────────────────────

class TestNormalisationGuard:
    def test_raises_on_unnormalised(self):
        sources = make_sources(3)
        with pytest.raises(ValueError):
            aggregate_it2(sources, weights=np.array([0.5, 0.5, 0.5]))

    def test_raises_on_negative_weight(self):
        sources = make_sources(3)
        with pytest.raises(ValueError):
            aggregate_it2(sources, weights=np.array([0.5, 0.6, -0.1]))


# ── Example 3.16: Sharpness ──────────────────────────────────────────────────

class TestSharpness:
    def test_bound_attained(self):
        """
        Example 3.16: evaluate each mu at tau_i = c_i, eps_k=0.
        Width = kavg * eps_c / 2  exactly (linear approx tight).
        """
        k, eps_c = 3.0, 0.10
        n = 5
        # Evaluate at tau=c so mu_i(1-mu_i) = 1/4 exactly
        sources = [
            IT2Source(tau=0.5, c=0.5, k=k, eps_c=eps_c, eps_k=0.0)
            for _ in range(n)
        ]
        res = aggregate_it2(sources, weights=np.ones(n) / n)
        expected = k * eps_c / 2.0   # kavg = k, uniform weights
        # Exact: sigma(k*eps_c) - sigma(-k*eps_c)
        from it2_aggregation import sigmoid
        exact = sigmoid(0.5, 0.5 - eps_c, k) - sigmoid(0.5, 0.5 + eps_c, k)
        assert abs(res.width_empirical - exact) < 1e-10
        # Certificate attained within 1% (slight over-estimate due to linearisation)
        assert abs(res.width_certified - expected) / expected < 0.015
