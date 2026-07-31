"""
Tests for src/features/build_features.py — the feature engineering functions.

Each test builds a small fake DataFrame, runs one function, and checks the
output. No files, no model, no MLflow — pure and fast.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import (
    drop_unnecessary_columns,
    create_domain_features,
    encode_categorical,
)


@pytest.fixture
def sample_df():
    """A tiny fake transactions DataFrame covering the columns we care about."""
    return pd.DataFrame(
        {
            "step": [1, 2, 3],
            "type": ["TRANSFER", "PAYMENT", "CASH_OUT"],
            "amount": [1000.0, 50.0, 0.0],
            "nameOrig": ["A1", "A2", "A3"],
            "oldbalanceOrg": [2000.0, 100.0, 500.0],
            "newbalanceOrig": [1000.0, 50.0, 500.0],
            "nameDest": ["B1", "B2", "B3"],
            "oldbalanceDest": [0.0, 0.0, 0.0],
            "newbalanceDest": [1000.0, 0.0, 0.0],
            "isFraud": [1, 0, 0],
            "isFlaggedFraud": [0, 0, 0],
        }
    )


# ---------------------------------------------------------------------------
# drop_unnecessary_columns
# ---------------------------------------------------------------------------
def test_drop_removes_expected_columns(sample_df):
    result = drop_unnecessary_columns(sample_df)
    for col in ["nameOrig", "nameDest", "isFlaggedFraud", "step"]:
        assert col not in result.columns


def test_drop_keeps_useful_columns(sample_df):
    result = drop_unnecessary_columns(sample_df)
    for col in ["type", "amount", "isFraud", "oldbalanceOrg"]:
        assert col in result.columns


def test_drop_is_safe_when_columns_missing():
    """If a to-drop column is already absent, it should not crash."""
    df = pd.DataFrame({"type": ["PAYMENT"], "amount": [10.0]})
    result = drop_unnecessary_columns(df)  # no nameOrig/step present
    assert "amount" in result.columns


# ---------------------------------------------------------------------------
# create_domain_features
# ---------------------------------------------------------------------------
def test_create_features_adds_new_columns(sample_df):
    result = create_domain_features(sample_df)
    assert "is_transfer_or_cashout" in result.columns
    assert "amount_log" in result.columns


def test_is_transfer_or_cashout_flag(sample_df):
    """1 for TRANSFER/CASH_OUT, 0 otherwise."""
    result = create_domain_features(sample_df)
    # rows: TRANSFER -> 1, PAYMENT -> 0, CASH_OUT -> 1
    assert result["is_transfer_or_cashout"].tolist() == [1, 0, 1]


def test_amount_log_is_correct(sample_df):
    """amount_log should equal log1p(amount), and handle amount=0 safely."""
    result = create_domain_features(sample_df)
    expected = np.log1p(sample_df["amount"])
    assert np.allclose(result["amount_log"], expected)
    # log1p(0) == 0, no NaN/inf even when amount is 0
    assert not result["amount_log"].isnull().any()


# ---------------------------------------------------------------------------
# encode_categorical
# ---------------------------------------------------------------------------
def test_encode_removes_type_column(sample_df):
    result = encode_categorical(sample_df)
    assert "type" not in result.columns


def test_encode_creates_dummy_columns(sample_df):
    """One-hot columns should appear with the 'type_' prefix."""
    result = encode_categorical(sample_df)
    type_cols = [c for c in result.columns if c.startswith("type_")]
    assert len(type_cols) >= 1


def test_encode_dummies_are_binary(sample_df):
    """All one-hot values must be 0 or 1 (booleans count as 0/1)."""
    result = encode_categorical(sample_df)
    type_cols = [c for c in result.columns if c.startswith("type_")]
    for col in type_cols:
        assert result[col].isin([0, 1, True, False]).all()
