# Data directory

## Study 1 — Diabetes dataset

No download required. Accessed automatically via:

```python
from sklearn.datasets import load_diabetes
```

This is the lars::diabetes dataset from Efron et al. (2004), 442 patients
× 10 physiological features, standardised to unit ℓ₂ norm per column.

Reference:
> Efron, B., Hastie, T., Johnstone, I., & Tibshirani, R. (2004).
> Least angle regression. Annals of Statistics, 32(2), 407–499.
> https://doi.org/10.1214/009053604000000067

---

## Study 2 — Steel Plates Faults dataset

Download from UCI ML Repository:
https://archive.ics.uci.edu/dataset/198/steel+plates+faults

Place the file `Faults.NNA` in this (`data/`) directory.

If the file is absent, `scripts/study2_steel_plates.py` automatically
generates a synthetic proxy (1941 × 27, ~8.1% Pastry prevalence) so the
pipeline can be validated end-to-end.

Reference:
> Buscema, M., Terzi, S., & Tastle, W. (2010).
> Steel Plates Faults [Dataset]. UCI Machine Learning Repository.
> https://doi.org/10.24432/C5J88N
