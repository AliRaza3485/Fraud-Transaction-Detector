"""
split_data.py

Splits the feature-engineered dataset into train and test sets.
Uses stratified split because the target (isFraud) is highly imbalanced.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import os


def load_featured_data(filepath: str) -> pd.DataFrame:
    print(f"Loading featured data from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"Data loaded. Shape: {df.shape}")
    return df


def split_data(
    df: pd.DataFrame,
    target_col: str = "isFraud",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Split data into train/test sets using stratification on the target,
    since fraud cases are extremely rare (~0.13%) and must be
    proportionally represented in both sets.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,  # keeps fraud ratio same in both train and test
    )

    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Fraud ratio in train: {y_train.mean():.5f}")
    print(f"Fraud ratio in test: {y_test.mean():.5f}")

    return X_train, X_test, y_train, y_test


def save_splits(X_train, X_test, y_train, y_test, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
    X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False)

    print(f"Train/test splits saved to: {output_dir}")


def main():
    input_path = "data/processed/featured_transactions.csv"
    output_dir = "data/processed"

    df = load_featured_data(input_path)
    X_train, X_test, y_train, y_test = split_data(df)
    save_splits(X_train, X_test, y_train, y_test, output_dir)


if __name__ == "__main__":
    main()
