"""
split_data.py

Splits the feature-engineered dataset into train and test sets.
Uses stratified split because the target (isFraud) is highly imbalanced.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import os

from src.config import CONFIG
from src.logging_config import get_logger

logger = get_logger(__name__)


def load_featured_data(filepath: str) -> pd.DataFrame:
    logger.info("Loading featured data from: %s", filepath)
    df = pd.read_csv(filepath)
    logger.info("Data loaded. Shape: %s", df.shape)
    return df


def split_data(
    df: pd.DataFrame,
    target_col: str = CONFIG.features.target_col,
    test_size: float = CONFIG.split.test_size,
    random_state: int = CONFIG.split.random_state,
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

    logger.info("Train shape: %s, Test shape: %s", X_train.shape, X_test.shape)
    logger.info("Fraud ratio in train: %.5f", y_train.mean())
    logger.info("Fraud ratio in test: %.5f", y_test.mean())

    return X_train, X_test, y_train, y_test


def save_splits(X_train, X_test, y_train, y_test, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    X_train.to_csv(CONFIG.paths.x_train, index=False)
    X_test.to_csv(CONFIG.paths.x_test, index=False)
    y_train.to_csv(CONFIG.paths.y_train, index=False)
    y_test.to_csv(CONFIG.paths.y_test, index=False)

    logger.info("Train/test splits saved to: %s", output_dir)


def main():
    input_path = CONFIG.paths.featured_data
    output_dir = CONFIG.paths.processed_dir

    df = load_featured_data(input_path)
    X_train, X_test, y_train, y_test = split_data(df)
    save_splits(X_train, X_test, y_train, y_test, output_dir)


if __name__ == "__main__":
    main()
