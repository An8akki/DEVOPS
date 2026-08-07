from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# ---------------------------------------
# Load trained model
# ---------------------------------------
model = joblib.load("model.pkl")

# ---------------------------------------
# Create FastAPI app
# ---------------------------------------
app = FastAPI(
    title="Boston Housing Price Prediction API",
    description="FastAPI service for predicting Boston housing prices using a trained Random Forest Regressor.",
    version="1.0.0"
)

# ---------------------------------------
# Input Schema
# ---------------------------------------
class HouseFeatures(BaseModel):
    features: list[float]

# ---------------------------------------
# Home Endpoint
# ---------------------------------------
@app.get("/")
def home():
    return {
        "message": "Boston Housing Price Prediction API",
        "model": "Random Forest Regressor",
        "status": "Running"
    }

# ---------------------------------------
# Health Check
# ---------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok"
    }

# ---------------------------------------
# Prediction Endpoint
# ---------------------------------------
@app.post("/predict")
def predict(data: HouseFeatures):

    if len(data.features) != 13:
        return {
            "error": f"Expected 13 features, received {len(data.features)}"
        }

    features = np.array(data.features).reshape(1, -1)

    prediction = model.predict(features)[0]

    return {
        "predicted_price": round(float(prediction), 2)
    }
