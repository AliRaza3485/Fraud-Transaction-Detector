"""
make_dataset.py

Loads raw transaction data, performs data quality checks,
cleans it if necessary, and saves the processed version.
"""

import pandas as pd
import os

from src.config import CONFIG


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load raw CSV data from the given filepath.
    """
    print(f"Loading data from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"Data loaded successfully. Shape: {df.shape}")
    return df


def check_data_quality(df: pd.DataFrame) -> None:
    """
    Print a data quality report: missing values and duplicate rows.
    """
    print("\n--- Data Quality Report ---")

    # Missing values check
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    print("\nMissing values per column:")
    print(missing[missing > 0] if missing.sum() > 0 else "No missing values found.")

    # Duplicate rows check
    duplicate_count = df.duplicated().sum()
    duplicate_pct = (duplicate_count / len(df)) * 100
    print(f"\nDuplicate rows: {duplicate_count} ({duplicate_pct:.4f}%)")

    print("--- End of Report ---\n")


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

    print(
        f"Cleaning complete. Rows removed: {rows_removed} "
        f"({(rows_removed / initial_rows) * 100:.4f}%)"
    )

    return df


def save_processed_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Save the cleaned dataframe to the processed data folder.
    """
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"Processed data saved to: {output_path}")


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
