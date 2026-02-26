# Datasets

## Study 1 — Diabetes Dataset

| Field | Value |
|-------|-------|
| Source | `sklearn.datasets.load_diabetes` (mirrors lars R package) |
| Reference | Efron, B., Hastie, T., Johnstone, I., & Tibshirani, R. (2004). Least angle regression. *Annals of Statistics*, 32(2), 407–499. |
| Rows | 442 patients |
| Features | 10 normalised physiological measurements (age, sex, BMI, MAP, TC, LDL, HDL, TCH, LTG, GLU) |
| Target | Disease progression score (DPS) measured 1 year after baseline |
| Licence | BSD-3 (distributed with scikit-learn) |

**High-risk threshold**: DPS > 140 (228 patients, 51.6%)

Features are already normalised in the sklearn version; IQRs computed on the normalised scale.

---

## Study 2 — Steel Plate Faults Dataset

| Field | Value |
|-------|-------|
| Source | UCI Machine Learning Repository |
| URL | https://archive.ics.uci.edu/dataset/198/steel+plates+faults |
| DOI | [10.24432/C5J88N](https://doi.org/10.24432/C5J88N) |
| Reference | Buscema, M., Terzi, S., & Tastle, W. (2010). UCI ML Repository. |
| Rows | 1,941 coil samples |
| Features | 27 geometric and electromagnetic features |
| Labels | 7 binary fault types (Pastry, Z_Scratch, K_Scratch, Stains, Dirtiness, Bumps, Other) |
| Licence | CC BY 4.0 |

**Target for Study 2**: Binary Pastry fault label.  
Pastry faults: 158 samples (8.1%).

### Manual download (if automatic download fails)

1. Visit https://archive.ics.uci.edu/dataset/198/steel+plates+faults
2. Download `steel+plates+faults.zip`
3. Extract `faults.dat` to this `data/` directory
4. Re-run `python data/download_data.py`
