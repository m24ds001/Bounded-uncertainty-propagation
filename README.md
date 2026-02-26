# IT2 Evidence Framework

**Bounded uncertainty propagation in normalised interval Type-2 fuzzy aggregation**

> Renikunta Ramesh & Ramsha Mehreen  
> Department of Mathematics and Humanities / Department of Information Technology  
> Kakatiya Institute of Technology and Science, Warangal 506015, Telangana, India

---

## Overview

This repository contains all code, data-download utilities, and reproduction scripts for the paper:

> *Bounded uncertainty propagation in normalised interval Type-2 fuzzy aggregation*,  
> Fuzzy Optimization and Decision Making (submitted).

The paper proves that weight normalisation is **necessary and sufficient** for O(1) aggregation interval width in interval Type-2 (IT2) fuzzy systems, derives closed-form width certificates computable in O(n) time, and validates these guarantees on two publicly available benchmark datasets.

---

## Repository Structure

```
it2-evidence-framework/
├── README.md                   ← This file
├── requirements.txt            ← Python dependencies
├── environment.yml             ← Conda environment spec
├── LICENSE                     ← MIT License
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── membership.py           ← Sigmoid & Lipschitz membership functions (Lemma 1, 2)
│   ├── aggregation.py          ← IT2 weighted aggregation (Theorem 3, 13, 20)
│   ├── width_bounds.py         ← Width certificates (Theorems 8, 9, 10, 12; Cor. 11, 14)
│   ├── corner_evaluation.py    ← Corner evaluation O(n) (Theorem 7, Proposition 5)
│   ├── yager.py                ← Yager λ-aggregation O(1) certificate (Theorem 6)
│   ├── concentration.py        ← Bernstein sub-Gaussian bound (Theorem 15, Corollary 16)
│   └── utils.py                ← Shared utilities and plotting helpers
│
├── data/
│   ├── download_data.py        ← Downloads UCI + sklearn diabetes datasets
│   └── README.md               ← Dataset descriptions and licenses
│
├── notebooks/
│   ├── 02_study1_diabetes.py   ← Study 1: UCI Diabetes (Section 4.2, Tables 1 & 2)
│   ├── 03_study2_steel_plate.py← Study 2: Steel Plate Faults (Section 4.3, Table 3)
│   └── 05_bootstrap_stability.py ← Appendix B bootstrap stability
│
├── tests/
│   └── test_width_bounds.py    ← Unit tests for all major theorems
│
├── figures/
│   ├── figure1_scaling_comparison.py   ← Reproduces Figure 1
│   └── figure2_width_decomposition.py  ← Reproduces Figure 2
│
└── results/
    ├── table3_steel_plate.csv          ← Table 3 numerical results
    ├── table1_diabetes_params.csv      ← Table 1 IT2 parameters
    └── table2_width_decomposition.csv  ← Table 2 per-criterion decomposition
```

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/rramesh/it2-evidence-framework.git
cd it2-evidence-framework

# Option A — pip
pip install -r requirements.txt

# Option B — conda
conda env create -f environment.yml
conda activate it2-fuzzy
```

### 2. Download datasets

```bash
python data/download_data.py
```

This downloads:
- **Diabetes dataset** via `sklearn.datasets.load_diabetes` (mirrors the lars R package).
- **Steel Plate Faults dataset** from the [UCI ML Repository](https://archive.ics.uci.edu/dataset/198/steel+plates+faults).

### 3. Reproduce all results

```bash
# Study 1 — Diabetes risk scoring
python notebooks/02_study1_diabetes.py

# Study 2 — Steel plate fault detection (all scaling experiments)
python notebooks/03_study2_steel_plate.py

# Figures
python figures/figure1_scaling_comparison.py
python figures/figure2_width_decomposition.py
```

All scripts write outputs to `results/` and figures to `figures/`.

### 4. Run tests

```bash
pytest tests/ -v
```

---

## Reproducing Individual Tables and Figures

| Output | Script | Runtime (approx.) |
|--------|--------|-------------------|
| Table 1 (IT2 parameters, diabetes) | `notebooks/02_study1_diabetes.py` | < 5 s |
| Table 2 (per-criterion width decomposition) | `notebooks/02_study1_diabetes.py` | < 5 s |
| Table 3 (steel plate scaling, B=100) | `notebooks/03_study2_steel_plate.py` | ~35 s |
| Feature robustness check | `notebooks/03_study2_steel_plate.py --robustness` | ~2 min |
| Runtime profiling | `notebooks/03_study2_steel_plate.py --profile` | ~1 min |
| Figure 1 (scaling comparison) | `figures/figure1_scaling_comparison.py` | ~5 s |
| Figure 2 (width decomposition) | `figures/figure2_width_decomposition.py` | < 5 s |

All random operations use **seed 42** throughout; results are fully deterministic.

---

## Key Theoretical Results Implemented

| Theorem / Result | Function | File |
|-----------------|----------|------|
| Lemma 1 — Sigmoid partial derivatives | `sigmoid_dtau()`, `sigmoid_dc()`, `sigmoid_dk()` | `src/membership.py` |
| Lemma 2 — Sigmoid Lipschitz constant `k/4` | `sigmoid_lipschitz_constant()` | `src/membership.py` |
| Theorem 7 — Corner evaluation enclosure | `corner_evaluation()`, `corner_evaluation_batch()` | `src/corner_evaluation.py` |
| Theorem 8 — Width bound under Lipschitz perturbation | `width_bound_lipschitz()` | `src/width_bounds.py` |
| Theorem 9 — Width-optimal weight allocation | `optimal_weight_allocation()` | `src/width_bounds.py` |
| Theorem 10 — Certified O(1) width | `certified_width_o1()` | `src/width_bounds.py` |
| Theorem 12 — Power-law contraction | `width_power_law()` | `src/width_bounds.py` |
| Theorem 13 — Unnormalised divergence lower bound | `unnormalised_width_lower_bound()` | `src/aggregation.py` |
| Theorem 15 — Bernstein sub-Gaussian bound | `bernstein_bound()` | `src/concentration.py` |
| Theorem 6 — Yager λ O(1) certificate | `yager_width_bound()` | `src/yager.py` |
| Lemma 19 + Theorem 20 — Stability under perturbation | `stability_constant()`, `verify_stability()` | `src/aggregation.py` |
| Proposition 5 — Corners are worst case | `proposition5_interior_vs_corner()` | `src/corner_evaluation.py` |
| Proposition 22 — Equivalence to iterative centroid | `counterexample_asymmetric_divergence()` | `src/corner_evaluation.py` |
| Corollary 14 — Normalisation necessary & sufficient | `test_theorem13_linear_growth()` | `tests/test_width_bounds.py` |

---

## Computational Environment

Results reported in the paper were produced on:

```
CPU:    Intel Core i7-1165G7 (2.80 GHz base, 4 cores)
RAM:    16 GB DDR4
OS:     Ubuntu 22.04 LTS
Python: 3.11.4
NumPy:  1.25.2
scikit-learn: 1.3.0
```

Corner evaluation at n=27 on 1,941 samples: **0.004 ± 0.001 s**.  
Full Study 2 pipeline (n=27, B=100): **0.311 ± 0.014 s**.

---

## Datasets

| Dataset | Source | Licence | Rows | Features |
|---------|--------|---------|------|----------|
| Diabetes (lars) | Efron et al. (2004) / sklearn | BSD-3 | 442 | 10 |
| Steel Plate Faults | Buscema et al. (2010) / UCI | CC BY 4.0 | 1,941 | 27 |

DOI for steel plate faults: [10.24432/C5J88N](https://doi.org/10.24432/C5J88N)

---

## Citation

```bibtex
@article{ramesh2025it2bounded,
  title   = {Bounded uncertainty propagation in normalised interval {Type-2} fuzzy aggregation},
  author  = {Ramesh, Renikunta and Mehreen, Ramsha},
  journal = {Fuzzy Optimization and Decision Making},
  year    = {2025},
  note    = {Submitted}
}
```

---

## License

This repository is released under the [MIT License](LICENSE).  
Dataset licenses are documented in `data/README.md`.

---

## Contact

- **Renikunta Ramesh** — rr.mh@kitsw.ac.in  
- **Ramsha Mehreen** — ramshamehreen2208@gmail.com
