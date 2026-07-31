"""
main.py

FastAPI application that serves the fraud-detection model.

The model + scaler are loaded ONCE at startup (not per request) and reused
for every prediction. Feature engineering and prediction reuse the exact
same functions as the training/batch pipeline (src/models/predict.py), so
the API can never drift from how the model was trained.

Run it (from the project root):
    uvicorn src.api.main:app --reload

Then open the interactive docs at:
    http://localhost:8000/docs
"""

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException

from src.config import CONFIG
from src.models.predict import load_model, load_scaler, prepare_features, predict
from src.api.schemas import (
    Transaction,
    PredictionResponse,
    HealthResponse,
)

app = FastAPI(
    title="Fraud Transaction Detection API",
    description="Predicts whether a financial transaction is fraudulent.",
    version="1.0.0",
)

# Module-level holders for the loaded artifacts (filled at startup).
_model = None
_scaler = None


@app.on_event("startup")
def load_artifacts():
    """
    Load the model and scaler once, when the server starts.
    Doing this per-request would be far too slow.
    """
    global _model, _scaler
    mlflow.set_tracking_uri(CONFIG.mlflow.tracking_uri)
    _model = load_model()
    _scaler = load_scaler()


@app.get("/", tags=["health"])
def root():
    """Simple liveness message."""
    return {"message": "Fraud Transaction Detection API is running."}


@app.get("/health", response_model=HealthResponse, tags=["health"])
def health():
    """Report whether the model artifacts are loaded and ready."""
    return HealthResponse(
        status="ok" if _model is not None else "model not loaded",
        model_loaded=_model is not None,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict_transaction(transaction: Transaction):
    """
    Score a single transaction for fraud.

    The incoming transaction is turned into a one-row DataFrame, run through
    the SAME feature pipeline used in training, and scored by the model.
    """
    if _model is None or _scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    # Pydantic model -> one-row DataFrame the pipeline expects
    raw_df = pd.DataFrame([transaction.model_dump()])
    # Enum -> plain string so it matches the training data ("TRANSFER", etc.)
    raw_df["type"] = raw_df["type"].astype(str)

    try:
        features = prepare_features(raw_df, _scaler)
        result = predict(_model, features)
    except Exception as exc:  # any pipeline/model failure -> clean 500
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

    row = result.iloc[0]
    return PredictionResponse(
        fraud_probability=float(row["fraud_probability"]),
        is_fraud_predicted=int(row["is_fraud_predicted"]),
    )
