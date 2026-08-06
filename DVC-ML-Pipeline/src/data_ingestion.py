"""
Stage 1: Data Ingestion
------------------------
Loads the scikit-learn Breast Cancer Wisconsin dataset (binary classification:
malignant vs benign tumor) and dumps it as a raw CSV file.

Output:
    data/raw/data.csv
"""

import os
import pandas as pd
from sklearn.datasets import load_breast_cancer


def load_data() -> pd.DataFrame:
    """Load the sklearn breast cancer dataset into a DataFrame."""
    bunch = load_breast_cancer(as_frame=True)
    df = bunch.frame  #includes feature columns + 'target'
    return df


def save_raw_data(df: pd.DataFrame, out_dir: str = "data/raw") -> None:
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "data.csv")
    df.to_csv(out_path, index=False)
    print(f"[data_ingestion] Saved raw data -> {out_path} (shape={df.shape})")


def main():
    df = load_data()
    save_raw_data(df)


if __name__ == "__main__":
    main()