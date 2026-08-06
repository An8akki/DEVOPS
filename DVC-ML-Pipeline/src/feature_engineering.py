"""
Stage 3: Feature Engineering
-----------------------------
Splits train/test data and scales features.

Input:
    data/processed/data.csv

Outputs:
    data/features/train.csv
    data/features/test.csv
    data/features/scaler.pkl
"""

import os
import yaml
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_params(path="params.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def load_processed_data(path="data/processed/data.csv"):
    df = pd.read_csv(path)
    print(f"[feature_engineering] Loaded processed data (shape={df.shape})")
    return df


def build_features(df, test_size, random_state):

    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    scaler = StandardScaler()

    X_train = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X.columns
    )

    X_test = pd.DataFrame(
        scaler.transform(X_test),
        columns=X.columns
    )

    train_df = X_train.copy()
    train_df["target"] = y_train.reset_index(drop=True)

    test_df = X_test.copy()
    test_df["target"] = y_test.reset_index(drop=True)

    return train_df, test_df, scaler


def save_features(train_df, test_df, scaler):

    os.makedirs("data/features", exist_ok=True)

    train_df.to_csv("data/features/train.csv", index=False)

    test_df.to_csv("data/features/test.csv", index=False)

    joblib.dump(scaler, "data/features/scaler.pkl")

    print("[feature_engineering] Features saved")


def main():

    params = load_params()["feature_engineering"]

    df = load_processed_data()

    train_df, test_df, scaler = build_features(
        df,
        params["test_size"],
        params["random_state"]
    )

    save_features(train_df, test_df, scaler)


if __name__ == "__main__":
    main()