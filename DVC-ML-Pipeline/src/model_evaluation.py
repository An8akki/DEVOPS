"""
Stage 5: Model Evaluation
-------------------------
Loads the trained model and test data,
evaluates the regression model,
logs metrics to MLflow,
and saves metrics.json.

Input:
    model.pkl
    data/features/test.csv

Output:
    metrics.json
"""

import json
import joblib
import pandas as pd
import mlflow

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


def load_model(path="model.pkl"):
    model = joblib.load(path)
    print("[model_evaluation] Model loaded")
    return model


def load_test_data(path="data/features/test.csv"):
    df = pd.read_csv(path)
    print(f"[model_evaluation] Loaded test data (shape={df.shape})")
    return df


def evaluate(model, df):
    X_test = df.drop(columns=["target"])
    y_test = df["target"]

    predictions = model.predict(X_test)

    rmse = mean_squared_error(y_test, predictions) ** 0.5
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    metrics = {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2_score": float(r2)
    }

    return metrics


def save_metrics(metrics, path="metrics.json"):
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)

    print("[model_evaluation] Metrics saved -> metrics.json")
    print(json.dumps(metrics, indent=4))


def main():

    model = load_model()

    df = load_test_data()

    metrics = evaluate(model, df)

    # Save metrics FIRST
    save_metrics(metrics)

    # Log to MLflow
    mlflow.set_experiment("Boston Housing Regression")

    with mlflow.start_run():

        mlflow.log_metric("RMSE", metrics["rmse"])
        mlflow.log_metric("MAE", metrics["mae"])
        mlflow.log_metric("R2_Score", metrics["r2_score"])

        mlflow.log_artifact("metrics.json")


if __name__ == "__main__":
    main()