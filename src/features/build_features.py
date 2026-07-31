"""
build_features.py

Takes the cleaned/processed dataset and builds model-ready features
based on insights discovered during EDA:
- type is a strong predictor (fraud only in TRANSFER/CASH_OUT)
- amount is a strong predictor (fraud tends to involve higher amounts)
- balance mismatch features were NOT useful (dropped)
- step (time) was NOT useful (dropped)
- isFlaggedFraud is weak / potential leakage (dropped)
- nameOrig/nameDest are high-cardinality IDs (dropped)
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import joblib

from src.config import CONFIG


def load_processed_data(filepath: str) -> pd.DataFrame:
    """
    Load the cleaned dataset produced by make_dataset.py
    """
    print(f"Loading processed data from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"Data loaded. Shape: {df.shape}")
    return df


def drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns that EDA showed to be unhelpful or risky:
    - nameOrig, nameDest: high-cardinality IDs, not directly useful
    - isFlaggedFraud: extremely weak (caught only 16/8213 frauds), risk of leakage
    - step: fraud was found to be spread randomly across all time steps
    """
    columns_to_drop = CONFIG.features.columns_to_drop

    existing_cols_to_drop = [col for col in columns_to_drop if col in df.columns]
    df = df.drop(columns=existing_cols_to_drop)

    print(f"Dropped columns: {existing_cols_to_drop}")
    return df


def create_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features based on EDA insights.
    """
    # Strongest insight from EDA: fraud only occurs in TRANSFER and CASH_OUT
    df["is_transfer_or_cashout"] = df["type"].isin(["TRANSFER", "CASH_OUT"]).astype(int)

    # Log-transform amount since it was highly skewed (std >> mean)
    # log1p handles zero values safely (log(1+x) instead of log(x))
    df["amount_log"] = np.log1p(df["amount"])

    print("Created features: is_transfer_or_cashout, amount_log")
    return df


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-Hot Encode the 'type' column since it's categorical
    and showed strong relationship with fraud in EDA.
    """
    df = pd.get_dummies(df, columns=["type"], prefix="type", drop_first=True)

    print("Encoded 'type' column using One-Hot Encoding")
    return df


def scale_features(
    df: pd.DataFrame,
    target_col: str = CONFIG.features.target_col,
    scaler_output_path: str = CONFIG.paths.scaler,
) -> pd.DataFrame:
    """
    Apply StandardScaler to numeric features (not the target column,
    and not the already-binary/encoded columns).

    Saves the fitted scaler so it can be reused later for inference
    on new/unseen data (very important — never re-fit scaler on test data).
    """
    # Columns to scale: continuous numeric features only
    numeric_cols = CONFIG.features.numeric_cols
    numeric_cols = [col for col in numeric_cols if col in df.columns]

    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Save the scaler for later use (e.g., during model inference)
    os.makedirs(os.path.dirname(scaler_output_path), exist_ok=True)
    joblib.dump(scaler, scaler_output_path)

    print(f"Scaled columns: {numeric_cols}")
    print(f"Scaler saved to: {scaler_output_path}")
    return df


def save_featured_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Save the final feature-engineered dataset.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Featured data saved to: {output_path}")


def main():
    # File paths (from central config)
    input_path = CONFIG.paths.cleaned_data
    output_path = CONFIG.paths.featured_data
    scaler_path = CONFIG.paths.scaler

    # Pipeline steps
    df = load_processed_data(input_path)
    df = drop_unnecessary_columns(df)
    df = create_domain_features(df)
    df = encode_categorical(df)
    df = scale_features(df, target_col=CONFIG.features.target_col, scaler_output_path=scaler_path)
    save_featured_data(df, output_path)

    print("\nFinal feature set columns:")
    print(df.columns.tolist())
    print(f"\nFinal shape: {df.shape}")


if __name__ == "__main__":
    main()
