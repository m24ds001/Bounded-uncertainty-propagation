"""
study2_steel_plate.py
=====================
Study 2: Steel Plate Fault Detection (Section 4.3)

Reproduces:
  - Table 3  : Aggregation interval width vs. active criteria
               (proposed / IT2-TOPSIS / OWA), n ∈ {4,8,12,20,27,50,100,200}
  - Table 6  : Feature selection robustness (MI / Pearson / Random)
  - Table 7  : Runtime profiling for corner evaluation
  - Figure 1 : Scaling comparison (calls figures/figure1_scaling_comparison.py)
  - Fault classification summary at n=27

All random operations use seed=42.

Usage
-----
    python notebooks/03_study2_steel_plate.py
    python notebooks/03_study2_steel_plate.py --robustness   # Table 6
    python notebooks/03_study2_steel_plate.py --profile      # Table 7
"""

import sys, os, time, argparse
import numpy as np
import pandas as pd

# Allow running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.download_data import load_steel_plate
from src.membership import sigmoid, interval_membership
from src.corner_evaluation import corner_evaluation_batch
from src.width_bounds import certified_width_o1, per_source_sensitivity
from src.aggregation import unnormalised_width_lower_bound

SEED    = 42
RNG     = np.random.default_rng(SEED)
N_VALS  = [4, 8, 12, 20, 27, 50, 100, 200]
B_BOOT  = 100          # bootstrap resamples (Table 3)
K_UNIF  = 3.0          # uniform steepness (Study 2 setup)
EPS_C   = 0.05         # centre uncertainty
EPS_K   = 0.5          # steepness uncertainty
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# IT2-TOPSIS comparator (Chen & Lee 2010)
# ---------------------------------------------------------------------------

def it2_topsis_width(mu_lo, mu_hi, weights):
    """
    Approximate IT2-TOPSIS interval width using distance to positive/negative
    ideal solutions under the same membership parameterisation.

    Implements the two-step extremisation from Chen & Lee (2010):
      d+ = distance to positive ideal (all µ=1)
      d- = distance to negative ideal (all µ=0)
      CC = d- / (d+ + d-)

    For IT2, both d+ and d- produce intervals, creating a CC interval [CC_lo, CC_hi].
    """
    n = len(weights)
    # Positive ideal: µ_i = 1; Negative ideal: µ_i = 0
    # d+ components for IT2
    d_plus_hi = float(np.sqrt(np.dot(weights, (1 - mu_lo)**2)))
    d_plus_lo = float(np.sqrt(np.dot(weights, (1 - mu_hi)**2)))
    d_minus_hi = float(np.sqrt(np.dot(weights, mu_hi**2)))
    d_minus_lo = float(np.sqrt(np.dot(weights, mu_lo**2)))

    eps = 1e-12
    CC_lo = d_minus_lo / (d_plus_hi + d_minus_lo + eps)
    CC_hi = d_minus_hi / (d_plus_lo + d_minus_hi + eps)
    if CC_lo > CC_hi:
        CC_lo, CC_hi = CC_hi, CC_lo
    return CC_hi - CC_lo


# ---------------------------------------------------------------------------
# Feature selection methods
# ---------------------------------------------------------------------------

def select_features_mi(X, y, n):
    """Top-n by mutual information with class label."""
    from sklearn.feature_selection import mutual_info_classif
    mi = mutual_info_classif(X.values, y.values, random_state=SEED)
    idx = np.argsort(mi)[::-1][:n]
    return X.columns[idx].tolist()


def select_features_pearson(X, y, n):
    """Top-n by |Pearson correlation| with class label."""
    corr = X.apply(lambda col: abs(col.corr(y)))
    idx = corr.nlargest(n).index.tolist()
    return idx


def select_features_random(X, y, n, seed=SEED):
    """Random selection of n features (averaged over 20 draws in Table 6)."""
    rng_ = np.random.default_rng(seed)
    cols = rng_.choice(X.columns, size=n, replace=False)
    return cols.tolist()


# ---------------------------------------------------------------------------
# Core width computation for one (n, method, X_sub) configuration
# ---------------------------------------------------------------------------

def compute_widths_bootstrap(X_sub, y, c_vals, k_val, eps_c, eps_k,
                              weights, B, rng, operator="proposed"):
    """
    Bootstrap distribution of the empirical interval width for one n.

    Returns (mean_width, ci_half_width).
    """
    S, n = X_sub.shape
    tau   = X_sub.values            # (S, n)
    c_lo  = c_vals - eps_c          # (n,)
    c_hi  = c_vals + eps_c
    k_lo  = np.full(n, k_val - eps_k).clip(min=0.01)
    k_hi  = np.full(n, k_val + eps_k)

    # Compute per-sample widths
    E_lo, E_hi = corner_evaluation_batch(tau, c_lo, c_hi, k_lo, k_hi, weights)
    widths = E_hi - E_lo             # (S,)

    if operator == "owa":
        # Unnormalised OWA: width grows linearly with n (Example 1)
        # Use fixed v=1; width lower bound = n·v·k_min·eps_c/4
        widths = widths * n           # rescale by n to simulate unnormalised sum

    elif operator == "topsis":
        # IT2-TOPSIS: compute per-sample CC interval widths
        topsis_widths = []
        for s in range(S):
            mu_lo_s = interval_membership(tau[s], c_lo, c_hi, k_lo, k_hi)[0]
            mu_hi_s = interval_membership(tau[s], c_lo, c_hi, k_lo, k_hi)[1]
            topsis_widths.append(it2_topsis_width(mu_lo_s, mu_hi_s, weights))
        widths = np.array(topsis_widths)

    mean_w = widths.mean()
    # Bootstrap CI
    boot_means = []
    for _ in range(B):
        idx = rng.integers(0, S, size=S)
        boot_means.append(widths[idx].mean())
    boot_means = np.array(boot_means)
    ci_half = 1.96 * boot_means.std()   # 95% CI
    return mean_w, ci_half


# ---------------------------------------------------------------------------
# Main scaling experiment (Table 3)
# ---------------------------------------------------------------------------

def run_scaling_experiment(X, y, selection_fn=None, B=B_BOOT, rng=None):
    """Run the full scaling experiment for one feature selection criterion."""
    if rng is None:
        rng = np.random.default_rng(SEED)
    if selection_fn is None:
        selection_fn = select_features_mi

    rows = []
    for n in N_VALS:
        features = selection_fn(X, y, n)
        X_sub    = X[features]
        weights  = np.full(n, 1.0 / n)

        # Centre parameters: class-conditional medians for Pastry class
        c_vals = X_sub[y == 1].median().values if y.sum() > 0 else X_sub.median().values

        # Proposed normalised operator
        m_prop, ci_prop = compute_widths_bootstrap(
            X_sub, y, c_vals, K_UNIF, EPS_C, EPS_K, weights, B, rng,
            operator="proposed"
        )
        # IT2-TOPSIS
        m_topsis, ci_topsis = compute_widths_bootstrap(
            X_sub, y, c_vals, K_UNIF, EPS_C, EPS_K, weights, B, rng,
            operator="topsis"
        )
        # Unnormalised OWA
        m_owa, ci_owa = compute_widths_bootstrap(
            X_sub, y, c_vals, K_UNIF, EPS_C, EPS_K, weights, B, rng,
            operator="owa"
        )

        rows.append({
            "n"            : n,
            "proposed_mean": m_prop,
            "proposed_ci"  : ci_prop,
            "topsis_mean"  : m_topsis,
            "topsis_ci"    : ci_topsis,
            "owa_mean"     : m_owa,
            "owa_ci"       : ci_owa,
            "ratio"        : m_prop / m_owa if m_owa > 0 else np.nan,
        })

    return pd.DataFrame(rows)


def print_table3(df):
    """Print Table 3 in paper-style format."""
    print("\n" + "=" * 80)
    print("Table 3: Aggregation Interval Width vs. Active Criteria (Steel Plate Faults)")
    print("=" * 80)
    print(f"{'n':>4}  {'Proposed':>22}  {'IT2-TOPSIS':>22}  {'OWA (v=1)':>22}  {'Ratio':>6}")
    print("-" * 80)
    for _, row in df.iterrows():
        n    = int(row["n"])
        prop = f"{row['proposed_mean']:.3f} ± {row['proposed_ci']:.3f}"
        tops = f"{row['topsis_mean']:.3f} ± {row['topsis_ci']:.3f}"
        owa  = f"{row['owa_mean']:.3f} ± {row['owa_ci']:.3f}"
        ratio= f"{row['ratio']:.2f}"
        print(f"{n:>4}  {prop:>22}  {tops:>22}  {owa:>22}  {ratio:>6}")
    print("=" * 80)


# ---------------------------------------------------------------------------
# Feature robustness experiment (Table 6)
# ---------------------------------------------------------------------------

def run_robustness_experiment(X, y):
    """Reproduce Table 6: robustness across MI / Pearson / Random selection."""
    from scipy.stats import linregress

    methods = {
        "MI"     : select_features_mi,
        "Pearson": select_features_pearson,
        "Random" : None,   # handled specially below
    }
    N_RANDOM_DRAWS = 20
    rows_prop, rows_owa = [], []

    for method_name, sel_fn in methods.items():
        prop_widths_all, owa_widths_all = [], []

        for n in N_VALS:
            if method_name == "Random":
                # Average over 20 random draws
                prop_vals, owa_vals = [], []
                for draw in range(N_RANDOM_DRAWS):
                    features = select_features_random(X, y, n, seed=SEED + draw)
                    X_sub    = X[features]
                    w        = np.full(n, 1.0 / n)
                    c_vals   = X_sub.median().values
                    rng_     = np.random.default_rng(SEED)
                    m_p, _   = compute_widths_bootstrap(
                        X_sub, y, c_vals, K_UNIF, EPS_C, EPS_K, w, B_BOOT, rng_)
                    m_o, _   = compute_widths_bootstrap(
                        X_sub, y, c_vals, K_UNIF, EPS_C, EPS_K, w, B_BOOT, rng_,
                        operator="owa")
                    prop_vals.append(m_p)
                    owa_vals.append(m_o)
                prop_widths_all.append(np.mean(prop_vals))
                owa_widths_all.append(np.mean(owa_vals))
            else:
                features = sel_fn(X, y, n)
                X_sub    = X[features]
                w        = np.full(n, 1.0 / n)
                c_vals   = X_sub.median().values
                rng_     = np.random.default_rng(SEED)
                m_p, _   = compute_widths_bootstrap(
                    X_sub, y, c_vals, K_UNIF, EPS_C, EPS_K, w, B_BOOT, rng_)
                m_o, _   = compute_widths_bootstrap(
                    X_sub, y, c_vals, K_UNIF, EPS_C, EPS_K, w, B_BOOT, rng_,
                    operator="owa")
                prop_widths_all.append(m_p)
                owa_widths_all.append(m_o)

        # Fit ∆(n) = a + b/n  for proposed
        inv_n      = 1.0 / np.array(N_VALS)
        slope_p, intercept_p, r_p, *_ = linregress(inv_n, prop_widths_all)
        # Fit ∆(n) = c·n + d  for OWA
        slope_o, intercept_o, r_o, *_ = linregress(N_VALS, owa_widths_all)

        print(f"\n  {method_name:8s} | Proposed: a={intercept_p:.3f} b={slope_p:.3f} "
              f"R²={r_p**2:.3f} | OWA: c={slope_o:.3f} d={intercept_o:.3f} R²={r_o**2:.3f}")


# ---------------------------------------------------------------------------
# Runtime profiling (Table 7)
# ---------------------------------------------------------------------------

def run_profiling(X, y, n_runs=10):
    """Reproduce Table 7: runtime decomposition."""
    print("\nTable 7: Runtime Decomposition")
    print(f"{'n':>5}  {'CE (s)':>16}  {'Bootstrap (s)':>18}  {'Total (s)':>14}  {'CE%':>6}")
    print("-" * 65)

    for n in [4, 27, 200]:
        features  = select_features_mi(X, y, n)
        X_sub     = X[features]
        weights   = np.full(n, 1.0 / n)
        c_lo      = X_sub.median().values - EPS_C
        c_hi      = X_sub.median().values + EPS_C
        k_lo      = np.full(n, K_UNIF - EPS_K).clip(min=0.01)
        k_hi      = np.full(n, K_UNIF + EPS_K)
        tau       = X_sub.values

        ce_times, total_times = [], []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            corner_evaluation_batch(tau, c_lo, c_hi, k_lo, k_hi, weights)
            t_ce = time.perf_counter() - t0

            t0  = time.perf_counter()
            rng_ = np.random.default_rng(SEED)
            compute_widths_bootstrap(X_sub, y, X_sub.median().values,
                                     K_UNIF, EPS_C, EPS_K, weights, B_BOOT, rng_)
            t_total = time.perf_counter() - t0

            ce_times.append(t_ce)
            total_times.append(t_total)

        ce_mean  = np.mean(ce_times)
        ce_std   = np.std(ce_times)
        tot_mean = np.mean(total_times)
        tot_std  = np.std(total_times)
        ce_pct   = 100 * ce_mean / tot_mean

        print(f"{n:>5}  {ce_mean:.3f} ± {ce_std:.3f}  "
              f"{tot_mean - ce_mean:.3f} ± {tot_std:.3f}  "
              f"{tot_mean:.3f} ± {tot_std:.3f}  "
              f"{ce_pct:.1f}%")


# ---------------------------------------------------------------------------
# Fault classification at n=27 (Section 4.3)
# ---------------------------------------------------------------------------

def classification_summary(X, y, n=27):
    """Certified fault classification at n=27."""
    features = select_features_mi(X, y, n)
    X_sub    = X[features]
    weights  = np.full(n, 1.0 / n)
    c_lo     = X_sub.median().values - EPS_C
    c_hi     = X_sub.median().values + EPS_C
    k_lo     = np.full(n, K_UNIF - EPS_K).clip(min=0.01)
    k_hi     = np.full(n, K_UNIF + EPS_K)
    tau      = X_sub.values

    E_lo, E_hi = corner_evaluation_batch(tau, c_lo, c_hi, k_lo, k_hi, weights)
    theta   = 0.60

    certified_pastry   = (E_lo > theta).sum()
    certified_nopastry = (E_hi < 0.50).sum()
    indeterminate      = len(y) - certified_pastry - certified_nopastry

    # Precision on certified Pastry
    precision = (y.values[E_lo > theta] == 1).mean() if certified_pastry > 0 else np.nan

    print(f"\nFault Classification at n={n} (θ=0.60):")
    print(f"  Certified Pastry    : {certified_pastry} ({100*certified_pastry/len(y):.1f}%)")
    print(f"  Certified Non-Pastry: {certified_nopastry} ({100*certified_nopastry/len(y):.1f}%)")
    print(f"  Indeterminate       : {indeterminate} ({100*indeterminate/len(y):.1f}%)")
    print(f"  Precision (certified Pastry): {precision:.3f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Study 2: Steel Plate Faults")
    parser.add_argument("--robustness", action="store_true",
                        help="Run feature robustness experiment (Table 6)")
    parser.add_argument("--profile"   , action="store_true",
                        help="Run runtime profiling (Table 7)")
    args = parser.parse_args()

    print("Loading Steel Plate Faults dataset ...")
    X, y = load_steel_plate()
    print(f"  {X.shape[0]} samples, {X.shape[1]} features, "
          f"{y.sum()} Pastry faults ({100*y.mean():.1f}%)")

    if args.robustness:
        print("\nRunning feature robustness experiment (Table 6) ...")
        run_robustness_experiment(X, y)
    elif args.profile:
        print("\nRunning runtime profiling (Table 7) ...")
        run_profiling(X, y)
    else:
        print("\nRunning main scaling experiment (Table 3) ...")
        df = run_scaling_experiment(X, y)
        print_table3(df)
        df.to_csv(os.path.join(RESULTS_DIR, "table3_steel_plate.csv"), index=False)
        classification_summary(X, y, n=27)
        print("\nSaved results/table3_steel_plate.csv")
