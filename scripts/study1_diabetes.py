"""
study1_diabetes.py
==================
Study 1: IT2 diabetes risk scoring replication
(Section 4.2 of Ramesh & Mehreen 2025).

Reproduces:
  - Table 1: IT2 parameters for the diabetes dataset
  - Table 2: Per-criterion certified width decomposition
  - Certified fraction (64.9%) and 5-fold cross-validation

Dataset:
  sklearn.datasets.load_diabetes  (Efron et al. 2004)

Usage:
  python scripts/study1_diabetes.py

Random seed: 42
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import KFold
from scipy.stats import pearsonr

from it2_aggregation import (
    IT2Source, aggregate_it2, certify_decision, certified_width_bound,
)

RNG = np.random.default_rng(42)

# ── helpers ─────────────────────────────────────────────────────────────────

def iqr(x: np.ndarray) -> float:
    return float(np.percentile(x, 75) - np.percentile(x, 25))


def fit_parameters(X: np.ndarray, y: np.ndarray, alpha_eps: float = 0.10,
                   eps_k: float = 0.5):
    """
    Fit IT2 parameters from training data.

    Returns
    -------
    centres, steepnesses, eps_c_vec, weights, correlations
    """
    n_feat = X.shape[1]
    centres = np.median(X, axis=0)
    iqrs = np.array([iqr(X[:, j]) for j in range(n_feat)])
    iqrs = np.where(iqrs == 0, 1e-6, iqrs)          # guard against zero IQR
    steepnesses = 4.0 / iqrs
    eps_c_vec = alpha_eps * iqrs

    # Weights proportional to |Pearson corr with y|
    corrs = np.array([abs(pearsonr(X[:, j], y)[0]) for j in range(n_feat)])
    weights = corrs / corrs.sum()
    return centres, steepnesses, eps_c_vec, weights, corrs


def score_patient(x: np.ndarray, centres, steepnesses, eps_c_vec, weights,
                  eps_k: float = 0.5):
    """
    Build IT2 sources for one patient and run aggregation.
    """
    sources = [
        IT2Source(
            tau=float(x[j]),
            c=float(centres[j]),
            k=float(steepnesses[j]),
            eps_c=float(eps_c_vec[j]),
            eps_k=eps_k,
        )
        for j in range(len(centres))
    ]
    return aggregate_it2(sources, weights=weights)


def compute_M(X: np.ndarray, centres: np.ndarray) -> float:
    return float(np.max(np.abs(X - centres)))


# ── main study ───────────────────────────────────────────────────────────────

def run_study1():
    print("=" * 65)
    print("Study 1: Diabetes risk scoring (Ramesh & Mehreen 2025 §4.2)")
    print("=" * 65)

    # Load data
    data = load_diabetes()
    X, y = data.data, data.target
    feature_names = data.feature_names
    N, P = X.shape
    DPS_THRESHOLD = 140

    # Fit parameters on full dataset (unsupervised — no DPS used)
    EPS_K = 0.5
    ALPHA = 0.10
    centres, steepnesses, eps_c_vec, weights, corrs = fit_parameters(
        X, y, alpha_eps=ALPHA, eps_k=EPS_K
    )
    M_emp = compute_M(X, centres)
    M = 0.20   # round upper bound (see paper §4.2)

    # ── Table 1 ─────────────────────────────────────────────────────────────
    print("\nTable 1: IT2 aggregation parameters")
    header = f"{'Criterion':<8} {'ci':>8} {'IQRi':>8} {'ki':>8} "
    header += f"{'wi':>8} {'|corr|':>8}"
    print(header)
    print("-" * 55)
    iqrs = eps_c_vec / ALPHA
    for j, name in enumerate(feature_names):
        print(
            f"{name:<8} {centres[j]:>8.3f} {iqrs[j]:>8.3f} "
            f"{steepnesses[j]:>8.1f} {weights[j]:>8.3f} {corrs[j]:>8.3f}"
        )

    # ── Certified width (full dataset) ──────────────────────────────────────
    # Because k_i * eps_c_i = 0.40 for all i (see paper), use Theorem 3.8
    # with M = 0.20 and eps_k = 0.5
    dummy_sources = [
        IT2Source(tau=float(X[0, j]), c=float(centres[j]),
                  k=float(steepnesses[j]), eps_c=float(eps_c_vec[j]),
                  eps_k=EPS_K)
        for j in range(P)
    ]
    cert_bound, per_source = certified_width_bound(dummy_sources, weights, M=M)
    half_w = cert_bound / 2.0
    theta_lo = 0.50 - half_w          # = 0.45  (certification band)
    theta_hi = 0.50 + half_w          # = 0.55

    print(f"\nCertified width bound  ∆(Ã_10) ≤ {cert_bound:.3f}")
    print(f"Certification band     [{theta_lo:.2f}, {theta_hi:.2f}]  "
          f"(threshold 0.50 ± {half_w:.3f})")
    print(f"Observed M_emp = {M_emp:.4f}  (paper uses M = {M:.2f})")

    # ── Table 2 ─────────────────────────────────────────────────────────────
    print("\nTable 2: Per-criterion certified width decomposition")
    header2 = f"{'Criterion':<8} {'ki':>6} {'wi':>8} {'bi (cert)':>12} {'% total':>8}"
    print(header2)
    print("-" * 48)
    for j, name in enumerate(feature_names):
        pct = per_source[j] / cert_bound * 100
        print(f"{name:<8} {steepnesses[j]:>6.1f} {weights[j]:>8.3f} "
              f"{per_source[j]:>12.4f} {pct:>7.1f}%")
    print(f"{'Sum':<8} {'':>6} {weights.sum():>8.3f} "
          f"{per_source.sum():>12.4f} {'100.0%':>8}")

    # ── Score all 442 patients ───────────────────────────────────────────────
    decisions = []
    widths = []
    for i in range(N):
        res = score_patient(X[i], centres, steepnesses, eps_c_vec, weights,
                            EPS_K)
        # Use classification thresholds E > 0.55 → high, E < 0.45 → low
        dec = certify_decision(res, threshold=0.55)
        if dec == "certified_accept":
            d = "high_risk"
        elif res.E_upper < 0.45:
            d = "low_risk"
        else:
            d = "indeterminate"
        decisions.append(d)
        widths.append(res.width_empirical)

    widths = np.array(widths)
    n_cert = sum(1 for d in decisions if d != "indeterminate")
    cert_frac = n_cert / N

    print(f"\nClassification results (n={N})")
    print(f"  Certified (high or low risk):  {n_cert}  ({cert_frac*100:.1f}%)")
    print(f"  Indeterminate:                 {N-n_cert}  ({(1-cert_frac)*100:.1f}%)")
    print(f"  Empirical width: {widths.mean():.3f} ± {widths.std():.3f}")
    print(f"  Empirical-to-certified ratio:  {widths.mean()/cert_bound:.3f}")

    # ── 5-fold cross-validation ──────────────────────────────────────────────
    print("\n5-fold cross-validation of certified fraction:")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_fracs = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te = X[test_idx]
        c_tr, k_tr, ec_tr, w_tr, _ = fit_parameters(X_tr, y_tr,
                                                      alpha_eps=ALPHA,
                                                      eps_k=EPS_K)
        frac_fold = 0
        for i in range(len(X_te)):
            res = score_patient(X_te[i], c_tr, k_tr, ec_tr, w_tr, EPS_K)
            if res.E_lower > 0.55 or res.E_upper < 0.45:
                frac_fold += 1
        cv_fracs.append(frac_fold / len(X_te))
        print(f"  Fold {fold+1}: certified fraction = {cv_fracs[-1]*100:.1f}%")
    cv_arr = np.array(cv_fracs)
    print(f"  Mean ± SD: {cv_arr.mean()*100:.1f} ± {cv_arr.std()*100:.1f}%")

    return {
        "certified_fraction": cert_frac,
        "cert_bound": cert_bound,
        "width_mean": widths.mean(),
        "width_std": widths.std(),
        "cv_mean": cv_arr.mean(),
        "cv_std": cv_arr.std(),
    }


if __name__ == "__main__":
    run_study1()
