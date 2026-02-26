"""
download_data.py
================
Download and prepare the two benchmark datasets used in Studies 1 and 2.

Dataset 1 — Diabetes (lars / sklearn)
--------------------------------------
  Efron, B., Hastie, T., Johnstone, I., & Tibshirani, R. (2004).
  Least angle regression. Annals of Statistics, 32(2), 407–499.
  Available via sklearn.datasets.load_diabetes (BSD-3 licence).

Dataset 2 — Steel Plate Faults (UCI)
--------------------------------------
  Buscema, M., Terzi, S., & Tastle, W. (2010).
  UCI Machine Learning Repository.
  https://archive.ics.uci.edu/dataset/198/steel+plates+faults
  DOI: 10.24432/C5J88N   (CC BY 4.0 licence)

Usage
-----
    python data/download_data.py

Files created
-------------
    data/diabetes_X.csv       — 442 × 10 feature matrix (normalised)
    data/diabetes_y.csv       — 442 × 1 disease progression scores
    data/steel_plate_X.csv    — 1941 × 27 feature matrix
    data/steel_plate_y.csv    — 1941 × 1 binary Pastry fault labels
"""

import os
import sys
import numpy as np
import pandas as pd

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Study 1 — Diabetes dataset
# ---------------------------------------------------------------------------

def download_diabetes() -> None:
    """Download and save the lars/sklearn diabetes dataset."""
    try:
        from sklearn.datasets import load_diabetes
    except ImportError:
        print("[ERROR] scikit-learn is required.  Run: pip install scikit-learn")
        sys.exit(1)

    print("Downloading diabetes dataset from sklearn.datasets ...")
    data = load_diabetes()
    X    = pd.DataFrame(data.data, columns=data.feature_names)
    y    = pd.Series(data.target, name="dps")

    X_path = os.path.join(DATA_DIR, "diabetes_X.csv")
    y_path = os.path.join(DATA_DIR, "diabetes_y.csv")

    X.to_csv(X_path, index=False)
    y.to_csv(y_path, index=False)

    print(f"  Saved {X.shape[0]} samples × {X.shape[1]} features → {X_path}")
    print(f"  Saved targets → {y_path}")
    print(f"  DPS range: [{y.min():.1f}, {y.max():.1f}]  "
          f"| High-risk (DPS>140): {(y > 140).sum()} samples")


# ---------------------------------------------------------------------------
# Study 2 — Steel Plate Faults dataset
# ---------------------------------------------------------------------------

STEEL_PLATE_URL = (
    "https://archive.ics.uci.edu/static/public/198/steel+plates+faults.zip"
)

STEEL_FEATURE_NAMES = [
    "X_Minimum", "X_Maximum", "Y_Minimum", "Y_Maximum",
    "Pixels_Areas", "X_Perimeter", "Y_Perimeter", "Sum_of_Luminosity",
    "Minimum_of_Luminosity", "Maximum_of_Luminosity", "Length_of_Conveyer",
    "TypeOfSteel_A300", "TypeOfSteel_A400", "Steel_Plate_Thickness",
    "Edges_Index", "Empty_Index", "Square_Index", "Outside_X_Index",
    "Edges_X_Index", "Edges_Y_Index", "Outside_Global_Index",
    "LogOfAreas", "Log_X_Index", "Log_Y_Index", "Orientation_Index",
    "Luminosity_Index", "SigmoidOfAreas",
]

STEEL_LABEL_NAMES = [
    "Pastry", "Z_Scratch", "K_Scratch", "Stains",
    "Dirtiness", "Bumps", "Other_Faults",
]


def download_steel_plate() -> None:
    """Download and save the UCI Steel Plate Faults dataset."""
    try:
        import urllib.request, zipfile, io
    except ImportError:
        print("[ERROR] Standard library modules unavailable.")
        sys.exit(1)

    X_path = os.path.join(DATA_DIR, "steel_plate_X.csv")
    y_path = os.path.join(DATA_DIR, "steel_plate_y.csv")

    print(f"Downloading Steel Plate Faults dataset from UCI ...")
    print(f"  URL: {STEEL_PLATE_URL}")

    try:
        with urllib.request.urlopen(STEEL_PLATE_URL, timeout=60) as resp:
            zip_bytes = resp.read()
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        print("  Please download manually from:")
        print("  https://archive.ics.uci.edu/dataset/198/steel+plates+faults")
        print("  and place 'faults.dat' and 'faults.names' in the data/ directory.")
        sys.exit(1)

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        # The archive contains 'faults.dat' with 34 columns (27 features + 7 labels)
        with z.open("faults.dat") as f:
            raw = np.loadtxt(f)

    X = pd.DataFrame(raw[:, :27], columns=STEEL_FEATURE_NAMES)
    labels = pd.DataFrame(raw[:, 27:], columns=STEEL_LABEL_NAMES)

    # Binary Pastry label (Study 2)
    y_pastry = labels["Pastry"].astype(int)
    y_pastry.name = "Pastry_fault"

    X.to_csv(X_path, index=False)
    y_pastry.to_csv(y_path, index=False)

    print(f"  Saved {X.shape[0]} samples × {X.shape[1]} features → {X_path}")
    print(f"  Saved Pastry labels → {y_path}")
    print(f"  Pastry faults: {y_pastry.sum()} ({100*y_pastry.mean():.1f}%)")


# ---------------------------------------------------------------------------
# Data loading helpers (used by study scripts)
# ---------------------------------------------------------------------------

def load_diabetes(data_dir: str = DATA_DIR):
    """Load pre-downloaded diabetes dataset.  Downloads if not present."""
    X_path = os.path.join(data_dir, "diabetes_X.csv")
    y_path = os.path.join(data_dir, "diabetes_y.csv")
    if not os.path.exists(X_path):
        download_diabetes()
    X = pd.read_csv(X_path)
    y = pd.read_csv(y_path).squeeze()
    return X, y


def load_steel_plate(data_dir: str = DATA_DIR):
    """Load pre-downloaded steel plate dataset.  Downloads if not present."""
    X_path = os.path.join(data_dir, "steel_plate_X.csv")
    y_path = os.path.join(data_dir, "steel_plate_y.csv")
    if not os.path.exists(X_path):
        download_steel_plate()
    X = pd.read_csv(X_path)
    y = pd.read_csv(y_path).squeeze()
    return X, y


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("IT2 Evidence Framework — Dataset Download")
    print("=" * 60)
    download_diabetes()
    print()
    download_steel_plate()
    print()
    print("All datasets ready.  Proceed with study scripts.")
