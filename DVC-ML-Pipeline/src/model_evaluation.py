"""
Stage 5 : Model Evaluation

Loads the best trained model,
evaluates it on the test dataset,
logs metrics to MLflow (DagsHub),
and saves metrics.json.

Input
-----
model.pkl
data/features/test.csv

Output
------
metrics.json
"""

import os
import json
import joblib
import pandas as pd
import mlflow

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

# ==========================================================
# MLflow + DagsHub Configuration
# ==========================================================
from dotenv import load_dotenv

load_dotenv()

mlflow.set_tracking_uri(
    "https://dagshub.com/An8akki/DVC-ML-Pipeline.mlflow"
)

mlflow.set_experiment("Boston Housing Regression")


# ==========================================================
# Utility Functions
# ==========================================================

def load_model(path="model.pkl"):
    model = joblib.load(path)
    print("[INFO] Loaded best model.")
    return model


def load_test_data(path="data/features/test.csv"):
    df = pd.read_csv(path)
    print(f"[INFO] Test Shape : {df.shape}")
    return df


def evaluate(model, X_test, y_test):

    predictions = model.predict(X_test)

    rmse = mean_squared_error(
        y_test,
        predictions
    ) ** 0.5

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    return {

        "RMSE": float(rmse),
        "MAE": float(mae),
        "R2": float(r2)

    }


def save_metrics(metrics, path="metrics.json"):

    with open(path, "w") as f:
        json.dump(
            metrics,
            f,
            indent=4
        )

    print(f"\nMetrics saved -> {path}")


# ==========================================================
# Main
# ==========================================================

def main():

    model = load_model()

    df = load_test_data()

    X_test = df.drop(columns=["target"])
    y_test = df["target"]

    metrics = evaluate(
        model,
        X_test,
        y_test
    )

    save_metrics(metrics)

    with mlflow.start_run(run_name="Final Evaluation"):

        # -----------------------
        # Log Metrics
        # -----------------------

        mlflow.log_metric(
            "RMSE",
            metrics["RMSE"]
        )

        mlflow.log_metric(
            "MAE",
            metrics["MAE"]
        )

        mlflow.log_metric(
            "R2",
            metrics["R2"]
        )

        # -----------------------
        # Upload metrics.json
        # -----------------------

        mlflow.log_artifact("metrics.json")

    print("\n===================================")
    print("FINAL MODEL PERFORMANCE")
    print("===================================")

    print(f"RMSE : {metrics['RMSE']:.4f}")
    print(f"MAE  : {metrics['MAE']:.4f}")
    print(f"R²   : {metrics['R2']:.4f}")

    print("===================================")


if __name__ == "__main__":
    main()