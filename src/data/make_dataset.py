"""
make_dataset.py

Loads raw transaction data, performs data quality checks,
cleans it if necessary, and saves the processed version.
"""

import pandas as pd
import os

from src.config import CONFIG
from src.logging_config import get_logger

logger = get_logger(__name__)


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load raw CSV data from the given filepath.
    """
    logger.info("Loading data from: %s", filepath)
    df = pd.read_csv(filepath)
    logger.info("Data loaded successfully. Shape: %s", df.shape)
    return df


def check_data_quality(df: pd.DataFrame) -> None:
    """
    Log a data quality report: missing values and duplicate rows.
    """
    logger.info("--- Data Quality Report ---")

    # Missing values check
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    if missing.sum() > 0:
        logger.info("Missing values per column:\n%s", missing[missing > 0])
    else:
        logger.info("Missing values per column: No missing values found.")

    # Duplicate rows check
    duplicate_count = df.duplicated().sum()
    duplicate_pct = (duplicate_count / len(df)) * 100
    logger.info("Duplicate rows: %s (%.4f%%)", duplicate_count, duplicate_pct)

    logger.info("--- End of Report ---")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset by removing duplicates and handling missing values,
    if any are found. Since our EDA showed the raw data is already clean,
    this function acts as a safeguard for future/updated data.
    """
    initial_rows = len(df)

    # Drop duplicate rows, if any
    df = df.drop_duplicates()

    # Drop rows with missing values, if any
    # (For this dataset, EDA confirmed 0 missing values, but this
    # safeguard protects the pipeline if raw data changes in future.)
    df = df.dropna()

    final_rows = len(df)
    rows_removed = initial_rows - final_rows

    logger.info(
        "Cleaning complete. Rows removed: %s (%.4f%%)",
        rows_removed,
        (rows_removed / initial_rows) * 100,
    )

    return df


def save_processed_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Save the cleaned dataframe to the processed data folder.
    """
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)
    logger.info("Processed data saved to: %s", output_path)


def main():
    # File paths (from central config)
    raw_data_path = CONFIG.paths.raw_data
    processed_data_path = CONFIG.paths.cleaned_data

    # Pipeline steps
    df = load_data(raw_data_path)
    check_data_quality(df)
    df_clean = clean_data(df)
    save_processed_data(df_clean, processed_data_path)


if __name__ == "__main__":
    main()
