"""
Stage 4: Model Building
-----------------------
Trains a RandomForestRegressor on the engineered training features,
logs parameters & model to MLflow, and saves the trained model.

Input:
    data/features/train.csv

Output:
    model.pkl
"""

import yaml
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub

dagshub.init(
    repo_owner="An8akki",
    repo_name="DVC-ML-Pipeline",
    mlflow=True
)

from sklearn.ensemble import RandomForestRegressor


def load_params(path="params.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_train_data(path="data/features/train.csv"):
    df = pd.read_csv(path)
    print(f"[model_building] Loaded training data (shape={df.shape})")
    return df


def train_model(df, n_estimators, max_depth, random_state):

    X_train = df.drop(columns=["target"])
    y_train = df["target"]

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
    )

    # MLflow Experiment
    mlflow.set_experiment("Boston Housing Regression")

    with mlflow.start_run():

        # Log Parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)

        # Train
        model.fit(X_train, y_train)

        print("[model_building] Model trained")

        # Log Model
        mlflow.sklearn.log_model(model, "model")

    return model


def save_model(model, path="model.pkl"):
    joblib.dump(model, path)
    print(f"[model_building] Model saved -> {path}")


def main():

    params = load_params()["model_building"]

    df = load_train_data()

    model = train_model(
        df,
        params["n_estimators"],
        params["max_depth"],
        params["random_state"],
    )

    save_model(model)


if __name__ == "__main__":
    main()