"""
Tests for src/models/predict.py — align_features() and predict().

We use a small FakeModel instead of loading the real XGBoost model, so these
tests are fast and don't depend on MLflow or the model registry.
"""

import numpy as np
import pandas as pd

from src.models.predict import align_features, predict


class FakeModel:
    """
    Minimal stand-in for the real XGBoost model.

    - get_booster().feature_names -> the columns the "model" expects
    - predict_proba(X)            -> fixed fraud probabilities
    """

    def __init__(self, feature_names, probs):
        self._feature_names = feature_names
        self._probs = np.array(probs)

    def get_booster(self):
        booster = type("Booster", (), {})()
        booster.feature_names = self._feature_names
        return booster

    def predict_proba(self, X):
        # sklearn-style output: column 0 = P(not fraud), column 1 = P(fraud)
        return np.column_stack([1 - self._probs, self._probs])


# ---------------------------------------------------------------------------
# align_features  (this is the fix for the original feature-mismatch bug)
# ---------------------------------------------------------------------------
def test_align_adds_missing_column_as_zero():
    """
    If the input is missing a column the model expects (e.g. type_DEBIT
    because no DEBIT rows were in the sample), it should be added as 0.
    """
    model = FakeModel(["amount", "type_DEBIT", "type_TRANSFER"], probs=[0.1])
    X = pd.DataFrame({"amount": [100.0], "type_TRANSFER": [1]})  # type_DEBIT missing

    aligned = align_features(model, X)

    assert list(aligned.columns) == ["amount", "type_DEBIT", "type_TRANSFER"]
    assert aligned["type_DEBIT"].iloc[0] == 0


def test_align_drops_extra_columns():
    """Columns the model never saw should be dropped."""
    model = FakeModel(["amount", "type_TRANSFER"], probs=[0.1])
    X = pd.DataFrame(
        {"amount": [100.0], "type_TRANSFER": [1], "unexpected_col": [999]}
    )

    aligned = align_features(model, X)

    assert "unexpected_col" not in aligned.columns
    assert list(aligned.columns) == ["amount", "type_TRANSFER"]


def test_align_enforces_column_order():
    """Output columns must be in the exact order the model expects."""
    model = FakeModel(["a", "b", "c"], probs=[0.1])
    X = pd.DataFrame({"c": [1], "a": [2], "b": [3]})  # wrong order

    aligned = align_features(model, X)

    assert list(aligned.columns) == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------
def test_predict_output_shape_and_columns():
    model = FakeModel(["amount"], probs=[0.1, 0.9, 0.4])
    X = pd.DataFrame({"amount": [10.0, 20.0, 30.0]})

    result = predict(model, X)

    assert len(result) == 3
    assert list(result.columns) == ["fraud_probability", "is_fraud_predicted"]


def test_predict_threshold_default():
    """Default threshold: prob >= 0.5 -> 1, else 0."""
    model = FakeModel(["amount"], probs=[0.2, 0.5, 0.8])
    X = pd.DataFrame({"amount": [1.0, 2.0, 3.0]})

    result = predict(model, X)

    assert result["is_fraud_predicted"].tolist() == [0, 1, 1]


def test_predict_custom_threshold():
    """A stricter threshold should flag fewer transactions."""
    model = FakeModel(["amount"], probs=[0.2, 0.5, 0.8])
    X = pd.DataFrame({"amount": [1.0, 2.0, 3.0]})

    result = predict(model, X, threshold=0.9)

    # none reach 0.9
    assert result["is_fraud_predicted"].tolist() == [0, 0, 0]


def test_predict_probabilities_passed_through():
    """fraud_probability should match the model's P(fraud)."""
    model = FakeModel(["amount"], probs=[0.11, 0.77])
    X = pd.DataFrame({"amount": [1.0, 2.0]})

    result = predict(model, X)

    assert np.allclose(result["fraud_probability"], [0.11, 0.77])
