"""
predict.py

Loads the registered model and reuses the SAME feature engineering
functions from build_features.py to ensure consistency with training.
"""

import pandas as pd
import joblib
import mlflow
import mlflow.xgboost

# Import the SAME feature engineering functions used during training
from src.features.build_features import (
    drop_unnecessary_columns,
    create_domain_features,
    encode_categorical,
)
from src.config import CONFIG
from src.logging_config import get_logger

logger = get_logger(__name__)

MODEL_NAME = CONFIG.mlflow.registry_name
MODEL_VERSION = CONFIG.mlflow.model_version
SCALER_PATH = CONFIG.paths.scaler
NUMERIC_COLS = CONFIG.features.numeric_cols


def load_model():
    logger.info("Loading model '%s' version %s from MLflow...", MODEL_NAME, MODEL_VERSION)
    model = mlflow.xgboost.load_model(f"models:/{MODEL_NAME}/{MODEL_VERSION}")
    logger.info("Model loaded successfully.")
    return model


def load_scaler(scaler_path: str = SCALER_PATH):
    logger.info("Loading scaler from: %s", scaler_path)
    scaler = joblib.load(scaler_path)
    return scaler


def prepare_features(raw_transactions: pd.DataFrame, scaler) -> pd.DataFrame:
    """
    Reuse the EXACT SAME functions from build_features.py
    to guarantee training/prediction consistency.
    """
    df = raw_transactions.copy()

    # Same functions used during training — no duplicate logic
    df = drop_unnecessary_columns(df)
    df = create_domain_features(df)
    df = encode_categorical(df)

    # Scale using the ALREADY-FITTED scaler (never fit a new one here)
    existing_numeric_cols = [col for col in NUMERIC_COLS if col in df.columns]
    df[existing_numeric_cols] = scaler.transform(df[existing_numeric_cols])

    return df


def align_features(model, X: pd.DataFrame) -> pd.DataFrame:
    """
    Align X's columns to exactly what the model saw during training.

    One-Hot Encoding on a small sample may miss some categories (e.g. no
    DEBIT rows -> no 'type_DEBIT' column). Reindexing against the model's
    expected feature names fills any missing column with 0 and enforces
    the same column order.
    """
    expected = model.get_booster().feature_names
    return X.reindex(columns=expected, fill_value=0)


def predict(model, X: pd.DataFrame, threshold: float = CONFIG.predict.threshold) -> pd.DataFrame:
    X = align_features(model, X)
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    return pd.DataFrame(
        {"fraud_probability": probabilities, "is_fraud_predicted": predictions}
    )


def main():
    mlflow.set_tracking_uri(CONFIG.mlflow.tracking_uri)

    model = load_model()
    scaler = load_scaler()

    new_transactions = pd.read_csv(CONFIG.paths.cleaned_data).sample(
        CONFIG.predict.sample_size, random_state=CONFIG.predict.sample_random_state
    )

    X_new = prepare_features(new_transactions, scaler)

    if "isFraud" in X_new.columns:
        X_new = X_new.drop(columns=["isFraud"])

    results = predict(model, X_new)
    logger.info("Prediction Results:\n%s", results)


if __name__ == "__main__":
    main()
