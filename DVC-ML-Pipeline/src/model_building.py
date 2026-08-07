"""
Stage 4 : Model Building

Train multiple regression models on the Boston Housing dataset,
compare them using the test dataset,
log everything to MLflow (DagsHub),
and save the best performing model.

Inputs
------
data/features/train.csv
data/features/test.csv

Output
------
model.pkl
"""

import os
import yaml
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv

load_dotenv()
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
)

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

# ==========================================================
# MLflow + DagsHub Configuration
# ==========================================================


mlflow.set_tracking_uri(
    "https://dagshub.com/An8akki/DVC-ML-Pipeline.mlflow"
)

mlflow.set_experiment("Boston Housing Regression")


# ==========================================================
# Utility Functions
# ==========================================================

def load_params(path="params.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_train_data(path="data/features/train.csv"):
    df = pd.read_csv(path)
    print(f"[INFO] Train Shape : {df.shape}")
    return df


def load_test_data(path="data/features/test.csv"):
    df = pd.read_csv(path)
    print(f"[INFO] Test Shape  : {df.shape}")
    return df


def evaluate(model, X, y):

    prediction = model.predict(X)

    rmse = mean_squared_error(y, prediction) ** 0.5
    mae = mean_absolute_error(y, prediction)
    r2 = r2_score(y, prediction)

    return rmse, mae, r2


# ==========================================================
# Main
# ==========================================================

def main():

    params = load_params()["model_building"]

    train_df = load_train_data()
    test_df = load_test_data()

    X_train = train_df.drop(columns=["target"])
    y_train = train_df["target"]

    X_test = test_df.drop(columns=["target"])
    y_test = test_df["target"]

    models = {

        "Linear Regression":

            LinearRegression(),

        "Decision Tree":

            DecisionTreeRegressor(

                max_depth=params["max_depth"],
                random_state=params["random_state"]

            ),

        "Random Forest":

            RandomForestRegressor(

                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                random_state=params["random_state"]

            ),

        "Gradient Boosting":

            GradientBoostingRegressor(

                random_state=params["random_state"]

            )

    }

    best_model = None
    best_name = None
    best_r2 = float("-inf")

    print("\n==============================")
    print("MODEL COMPARISON")
    print("==============================\n")

    for name, model in models.items():

        print(f"\nTraining {name}")

        with mlflow.start_run(run_name=name):

            model.fit(X_train, y_train)

            rmse, mae, r2 = evaluate(
                model,
                X_test,
                y_test
            )

            # -------------------------
            # Parameters
            # -------------------------

            mlflow.log_param("Model", name)

            if hasattr(model, "n_estimators"):

                mlflow.log_param(
                    "n_estimators",
                    model.n_estimators
                )

            if hasattr(model, "max_depth"):

                mlflow.log_param(
                    "max_depth",
                    model.max_depth
                )

            # -------------------------
            # Metrics
            # -------------------------

            mlflow.log_metric("RMSE", rmse)
            mlflow.log_metric("MAE", mae)
            mlflow.log_metric("R2", r2)

            # -------------------------
            # Save model in MLflow
            # -------------------------

            mlflow.sklearn.log_model(
                sk_model=model,
                name="model"
            )

            print(f"RMSE : {rmse:.4f}")
            print(f"MAE  : {mae:.4f}")
            print(f"R²   : {r2:.4f}")

            if r2 > best_r2:

                best_r2 = r2
                best_model = model
                best_name = name

    print("\n==============================")
    print("BEST MODEL")
    print("==============================")

    print(f"Model : {best_name}")
    print(f"R²    : {best_r2:.4f}")

    joblib.dump(best_model, "model.pkl")

    print("\nSaved best model -> model.pkl")


if __name__ == "__main__":
    main()