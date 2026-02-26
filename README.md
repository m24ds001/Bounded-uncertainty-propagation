# IT2 Evidence Framework

**Bounded uncertainty propagation in normalised interval Type-2 fuzzy aggregation**

Official code repository for:

> Ramesh, R. & Mehreen, R. (2026). *Bounded uncertainty propagation in normalised
> interval Type-2 fuzzy aggregation.* Kakatiya Institute of Technology and Science,
> Warangal. Correspondence: rr.mh@kitsw.ac.in

---

## Overview

This repository reproduces every theoretical result and empirical study in the
paper. The core contribution is a **closed-form, tight, O(1) width certificate**
for IT2 weighted aggregation under weight normalisation — a triple combination
absent from prior IT2 aggregation work.

Key results implemented:

| Result | Location | Description |
|--------|----------|-------------|
| Lemma 2.8–2.9 | `src/it2_aggregation.py` | Sigmoid derivatives & Lipschitz constant |
| Theorem 3.6 | `src/it2_aggregation.py` | Corner evaluation for interval bounds |
| Theorem 3.8 | `src/it2_aggregation.py` | Closed-form width bound |
| Theorem 3.9 | `src/it2_aggregation.py` | Width-optimal weight allocation |
| Theorem 3.11 | `src/it2_aggregation.py` | O(1) certified width under normalised weights |
| Corollary 3.19 | `tests/` | Necessity and sufficiency of normalisation |
| Theorem 3.4 | `src/it2_aggregation.py` | O(1) for Yager λ-family |
| Theorem 3.13 | `src/it2_aggregation.py` | Power-law uncertainty decay |
| Theorem 3.21 | `src/it2_aggregation.py` | Bernstein concentration bound |
| Study 1 | `scripts/study1_diabetes.py` | Diabetes risk scoring (Table 1–2) |
| Study 2 | `scripts/study2_steel_plates.py` | Steel plate scaling laws (Table 3, Fig. 1) |
| Example 4.1 | `scripts/example_supplier.py` | Four-criteria supplier selection |
| Figures | `scripts/generate_figures.py` | Fig. 1 and Fig. 2 |

---

## Repository structure

```
it2-evidence-framework/
├── src/
│   └── it2_aggregation.py      # Core module (all theorems)
├── scripts/
│   ├── study1_diabetes.py      # Study 1: diabetes risk scoring
│   ├── study2_steel_plates.py  # Study 2: steel plates scaling
│   ├── example_supplier.py     # Example 4.1: supplier selection
│   └── generate_figures.py     # Figures 1 and 2
├── tests/
│   └── test_it2_aggregation.py # Unit tests for all main theorems
├── data/
│   └── README_data.md          # Dataset instructions
├── figures/                    # Output directory for generated figures
├── notebooks/
│   └── demo.ipynb              # Interactive demo (optional)
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/rramesh/it2-evidence-framework.git
cd it2-evidence-framework
pip install -r requirements.txt
```

Tested with Python 3.9–3.12.

---

## Reproducing the results

### Worked example (Example 4.1, §4.4)

```bash
python scripts/example_supplier.py
```

No external data required. Reproduces all numerical values in Example 4.1
including the width certificate, critical tolerance ε*_c, certified ranking
analysis, and indeterminate-case recovery via the wᵢsᵢ decomposition.

### Study 1 — Diabetes risk scoring (§4.2)

```bash
python scripts/study1_diabetes.py
```

Uses `sklearn.datasets.load_diabetes` (no manual download needed).
Reproduces Table 1, Table 2, certified fraction (64.9%), and
5-fold cross-validation.

### Study 2 — Steel plate fault scaling (§4.3)

```bash
# Download dataset first (optional — synthetic proxy used if absent)
# From: https://archive.ics.uci.edu/dataset/198/steel+plates+faults
# Place Faults.NNA in data/

python scripts/study2_steel_plates.py
```

Reproduces Table 3 and Figure 1 (scaling law comparison, R² > 0.995).

### Figures

```bash
python scripts/generate_figures.py
# → figures/Fig1.pdf  (Scaling Law Comparison)
# → figures/Fig2.pdf  (Width Bound Decomposition)
```

### Unit tests

```bash
pytest tests/ -v
```

Tests verify all main theorems (Lemma 2.8–2.9, Theorems 3.4, 3.6, 3.8, 3.9,
3.11, 3.13, 3.21, Corollary 3.19, Example 3.16 sharpness).

---

## Quick API reference

```python
from src.it2_aggregation import (
    IT2Source, aggregate_it2, certify_decision, certify_ranking,
    certified_width_bound, width_optimal_weights,
)

# Define sources
sources = [
    IT2Source(tau=0.82, c=0.60, k=3.0, eps_c=0.10, eps_k=0.5),
    IT2Source(tau=0.71, c=0.65, k=4.0, eps_c=0.10, eps_k=0.5),
]
weights = np.array([0.6, 0.4])

# Aggregate
result = aggregate_it2(sources, weights=weights)
print(f"IT2 interval: [{result.E_lower:.3f}, {result.E_upper:.3f}]")
print(f"Certified width bound: {result.width_certified:.3f}")

# Certify decision
decision = certify_decision(result, threshold=0.50)
# Returns: 'certified_accept', 'certified_reject', or 'indeterminate'

# Find width-optimal weights
M = max(abs(s.tau - s.c) for s in sources)
w_opt, min_width = width_optimal_weights(sources, M)
```

---

## Datasets

| Dataset | Source | Access |
|---------|--------|--------|
| Diabetes (Study 1) | Efron et al. (2004) | `sklearn.datasets.load_diabetes()` |
| Steel Plates Faults (Study 2) | Buscema et al. (2010) | [UCI ML Repository](https://archive.ics.uci.edu/dataset/198/steel+plates+faults) |

See `data/README_data.md` for download instructions.

---

## Citation

```bibtex
@article{ramesh2025it2,
  title   = {Bounded uncertainty propagation in normalised
             interval Type-2 fuzzy aggregation},
  author  = {Ramesh, Renikunta and Mehreen, Ramsha},
  year    = {2025},
  institution = {Kakatiya Institute of Technology and Science, Warangal}
}
```

---

## License

MIT License — see `LICENSE` for details.

## Competing interests

The authors declare no competing interests.

## Funding

No funding was received for this study.
