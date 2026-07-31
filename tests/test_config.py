"""
Tests for src/config.py — the central configuration loader.

These verify that the YAML config loads correctly and that paths are
resolved to absolute paths (so scripts work from any directory).
"""

import os
from src.config import CONFIG


def test_config_loads():
    """CONFIG should load and expose the top-level sections."""
    assert CONFIG.paths is not None
    assert CONFIG.features is not None
    assert CONFIG.split is not None
    assert CONFIG.model is not None
    assert CONFIG.predict is not None


def test_paths_are_absolute():
    """Every path under `paths:` must be resolved to an absolute path."""
    assert os.path.isabs(CONFIG.paths.raw_data)
    assert os.path.isabs(CONFIG.paths.scaler)
    assert os.path.isabs(CONFIG.paths.cleaned_data)


def test_mlflow_uri_is_absolute():
    """The MLflow SQLite URI must point at an absolute path."""
    uri = CONFIG.mlflow.tracking_uri
    assert uri.startswith("sqlite:///")
    # The part after the prefix should be an absolute path
    db_path = uri.replace("sqlite:///", "")
    assert os.path.isabs(db_path)


def test_split_values():
    """Sanity-check the split parameters."""
    assert 0 < CONFIG.split.test_size < 1
    assert isinstance(CONFIG.split.random_state, int)


def test_best_params_to_dict():
    """best_params must convert to a plain dict (needed for XGBClassifier(**params))."""
    params = CONFIG.model.best_params.to_dict()
    assert isinstance(params, dict)
    assert "n_estimators" in params
    assert "max_depth" in params
