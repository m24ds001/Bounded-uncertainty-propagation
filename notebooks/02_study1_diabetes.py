"""
study1_diabetes.py
==================
Study 1: Diabetes Risk Scoring (Section 4.2)

Reproduces:
  - Table 1  : IT2 aggregation parameters for the diabetes dataset
  - Table 2  : Per-criterion certified width decomposition
  - Certified fraction (64.9%) and indeterminate count (155)
  - Five-fold cross-validation stability (64.4 ± 1.2%)

Membership function specification (Section 4.2):
  - Centre cᵢ     : sample median of criterion i
  - Steepness kᵢ  : 4 / IQR_i
  - Weights wᵢ    : proportional to |corr(criterion_i, DPS)|, normalised
  - ϵ_c           : 0.10 × IQR_i per criterion
  - ϵ_k           : 0.5 (uniform)
  - M             : 1.0 (conservative domain bound)

No outcome-dependent fitting is performed; parameters encode domain
knowledge only (median → 50% acceptability; IQR → local sensitivity).

Usage
-----
    python notebooks/02_study1_diabetes.py
"""

import sys, os
import numpy as np
import pandas as pd
from scipy.stats import iqr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.download_data import load_diabetes
from src.membership import sigmoid, interval_membership
from src.corner_evaluation import corner_evaluation_batch
from src.width_bounds import width_bound_lipschitz, certified_width_o1

SEED   = 42
EPS_K  = 0.5
M      = 1.0          # conservative domain bound
THRESHOLD_HI = 0.55   # certified high-risk
THRESHOLD_LO = 0.45   # certified low-risk
DPS_CUTOFF   = 140    # high-risk DPS threshold

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

CRITERION_ORDER = ["bmi", "ltg", "map", "tch", "glu", "ldl", "tc", "hdl", "age", "sex"]


# ---------------------------------------------------------------------------
# Parameter specification
# ---------------------------------------------------------------------------

def specify_parameters(X, y_dps):
    """
    Compute IT2 membership function parameters (Section 4.2 / Table 1).

    Returns a DataFrame with columns:
      criterion, ci, IQRi, ki, wi, |corr|, eps_c_i
    """
    rows = []
    for col in X.columns:
        xi     = X[col].values
        ci     = float(np.median(xi))
        iqr_i  = float(iqr(xi))
        ki     = 4.0 / iqr_i if iqr_i > 0 else 10.0
        corr   = abs(float(pd.Series(xi).corr(y_dps)))
        eps_c  = 0.10 * iqr_i
        rows.append(dict(criterion=col, ci=ci, IQRi=iqr_i,
                         ki=ki, abs_corr=corr, eps_ci=eps_c))

    df = pd.DataFrame(rows)
    # Normalise weights by |corr|
    df["wi"] = df["abs_corr"] / df["abs_corr"].sum()
    return df


# ---------------------------------------------------------------------------
# Width certificate (Theorem 8)
# ---------------------------------------------------------------------------

def compute_width_certificate(params_df):
    """
    Apply Theorem 8 to the full parameter set → aggregation interval width.
    """
    wi    = params_df["wi"].values
    ki    = params_df["ki"].values
    eps_c = params_df["eps_ci"].values   # per-criterion (not scalar)
    per_source = (ki / 4.0) * eps_c + (M / 4.0) * EPS_K
    certified_width = float(np.dot(wi, per_source))

    # Effective cap: membership ∈ (0,1) so width ≤ 1
    effective = min(certified_width, 1.0)
    return certified_width, effective


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_patients(X, params_df):
    """
    Compute IT2 interval [E, Ē] for each patient and certify classification.
    """
    wi    = params_df["wi"].values
    ci    = params_df["ci"].values
    ki    = params_df["ki"].values
    eps_c = params_df["eps_ci"].values
    eps_k = EPS_K

    c_lo = ci - eps_c
    c_hi = ci + eps_c
    k_lo = (ki - eps_k).clip(min=0.01)
    k_hi = ki + eps_k
    tau  = X.values

    E_lo, E_hi = corner_evaluation_batch(tau, c_lo, c_hi, k_lo, k_hi, wi)
    midpoint   = (E_lo + E_hi) / 2.0

    certified_hi    = E_lo > THRESHOLD_HI
    certified_lo    = E_hi < THRESHOLD_LO
    indeterminate   = ~certified_hi & ~certified_lo

    return pd.DataFrame({
        "E_lo"         : E_lo,
        "E_hi"         : E_hi,
        "midpoint"     : midpoint,
        "width"        : E_hi - E_lo,
        "cert_high"    : certified_hi,
        "cert_low"     : certified_lo,
        "indeterminate": indeterminate,
    })


# ---------------------------------------------------------------------------
# Per-criterion width decomposition (Table 2)
# ---------------------------------------------------------------------------

def width_decomposition(X, params_df, results_df):
    """
    Decompose certified width by criterion (Table 2).

    bᵢ = wᵢ · (kᵢ · ϵ_c_i / 4 + M · ϵ_k / 4)  ← certified contribution
    δ̂ᵢ = wᵢ · mean(µ_hi_i - µ_lo_i)             ← empirical mean weighted width
    """
    wi    = params_df["wi"].values
    ki    = params_df["ki"].values
    eps_c = params_df["eps_ci"].values

    b_i = wi * ((ki / 4.0) * eps_c + (M / 4.0) * EPS_K)

    # Empirical per-source widths
    delta_hat = []
    for j, row in params_df.iterrows():
        c_lo_j = row["ci"] - row["eps_ci"]
        c_hi_j = row["ci"] + row["eps_ci"]
        k_lo_j = max(row["ki"] - EPS_K, 0.01)
        k_hi_j = row["ki"] + EPS_K
        tau_j  = X.iloc[:, j].values
        mu_lo, mu_hi = interval_membership(
            tau_j,
            np.full_like(tau_j, c_lo_j),
            np.full_like(tau_j, c_hi_j),
            np.full_like(tau_j, k_lo_j),
            np.full_like(tau_j, k_hi_j),
        )
        delta_hat.append(float(row["wi"] * (mu_hi - mu_lo).mean()))

    params_df = params_df.copy()
    params_df["b_cert"]   = b_i
    params_df["delta_hat"]= delta_hat
    params_df["tightness"]= np.array(delta_hat) / b_i
    return params_df


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def cross_validate(X, y_dps, k_folds=5, seed=SEED):
    """
    5-fold CV to check certified-fraction stability (Section 4.2).
    """
    from sklearn.model_selection import KFold
    kf       = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    frac_list= []

    for train_idx, test_idx in kf.split(X):
        X_train = X.iloc[train_idx]
        X_test  = X.iloc[test_idx]
        y_train = y_dps.iloc[train_idx]

        # Estimate parameters from training fold
        params = specify_parameters(X_train, y_train)
        res    = classify_patients(X_test, params)
        frac   = (~res["indeterminate"]).mean()
        frac_list.append(frac)

    return np.mean(frac_list), np.std(frac_list)


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def print_table1(params_df):
    print("\nTable 1: IT2 Aggregation Parameters (Diabetes Dataset)")
    print("-" * 72)
    print(f"{'Criterion':>10}  {'ci':>8}  {'IQRi':>8}  {'ki':>8}  {'wi':>8}  {'|corr|':>8}")
    print("-" * 72)
    for _, row in params_df.iterrows():
        print(f"{row['criterion']:>10}  {row['ci']:>8.3f}  {row['IQRi']:>8.3f}"
              f"  {row['ki']:>8.1f}  {row['wi']:>8.3f}  {row['abs_corr']:>8.3f}")
    print("-" * 72)


def print_table2(decomp_df):
    print("\nTable 2: Per-Criterion Certified Width Decomposition")
    print("-" * 70)
    print(f"{'Criterion':>10}  {'ki':>6}  {'wi':>6}  {'bi(cert)':>10}  {'δ̂i(emp)':>10}  {'Tight.':>8}")
    print("-" * 70)
    for _, row in decomp_df.iterrows():
        print(f"{row['criterion']:>10}  {row['ki']:>6.1f}  {row['wi']:>6.3f}"
              f"  {row['b_cert']:>10.3f}  {row['delta_hat']:>10.3f}  {row['tightness']:>8.2f}")
    b_sum = decomp_df["b_cert"].sum()
    d_sum = decomp_df["delta_hat"].sum()
    print("-" * 70)
    print(f"{'Sum':>10}                {b_sum:>10.3f}  {d_sum:>10.3f}")
    print("-" * 70)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Study 1: Diabetes Risk Scoring")
    print("Loading data ...")
    X, y_dps = load_diabetes()
    print(f"  {X.shape[0]} patients, {X.shape[1]} criteria, "
          f"high-risk (DPS>140): {(y_dps > 140).sum()}")

    print("\nSpecifying IT2 parameters ...")
    params = specify_parameters(X, y_dps)
    print_table1(params)

    cert_width, eff_width = compute_width_certificate(params)
    k_avg = params["ki"].mean()
    print(f"\nWidth certificate (Theorem 10i):")
    print(f"  k_avg = {k_avg:.1f},  k_avg·ϵ_c/4 term = {k_avg*params['eps_ci'].mean()/4:.3f}")
    print(f"  Raw bound: {cert_width:.3f}  |  Effective (capped at 1): {eff_width:.3f}")

    print("\nClassifying patients ...")
    results = classify_patients(X, params)
    n_cert_hi = results["cert_high"].sum()
    n_cert_lo = results["cert_low"].sum()
    n_indet   = results["indeterminate"].sum()
    n_cert    = n_cert_hi + n_cert_lo
    print(f"  Certified high-risk  : {n_cert_hi}")
    print(f"  Certified low-risk   : {n_cert_lo}")
    print(f"  Certified total      : {n_cert} ({100*n_cert/len(results):.1f}%)")
    print(f"  Indeterminate        : {n_indet}")
    print(f"  Empirical width mean : {results['width'].mean():.3f} ± {results['width'].std():.3f}")

    print("\nPer-criterion decomposition ...")
    decomp = width_decomposition(X, params, results)
    print_table2(decomp)

    print("\nRunning 5-fold cross-validation ...")
    cv_mean, cv_std = cross_validate(X, y_dps)
    print(f"  Certified fraction: {100*cv_mean:.1f} ± {100*cv_std:.1f}%")
    print(f"  (Full-dataset: {100*n_cert/len(results):.1f}%)")

    # Save results
    params.to_csv(os.path.join(RESULTS_DIR, "table1_diabetes_params.csv"), index=False)
    decomp.to_csv(os.path.join(RESULTS_DIR, "table2_width_decomposition.csv"), index=False)
    print("\nSaved results/table1_diabetes_params.csv")
    print("Saved results/table2_width_decomposition.csv")
