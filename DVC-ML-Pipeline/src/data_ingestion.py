"""
Stage 1: Data Ingestion
------------------------
Loads the Boston Housing dataset from OpenML
and saves it as data/raw/data.csv
"""

import os
import pandas as pd
from sklearn.datasets import fetch_openml


def load_data():
    X, y = fetch_openml(
        name="boston",
        version=1,
        as_frame=True,
        return_X_y=True,
    )

    df = X.copy()
    df["target"] = y.astype(float)

    return df


def save_raw_data(df, out_dir="data/raw"):
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, "data.csv")

    df.to_csv(out_path, index=False)

    print(f"Saved raw dataset -> {out_path}")
    print(df.head())


def main():
    df = load_data()
    save_raw_data(df)


if __name__ == "__main__":
    main()