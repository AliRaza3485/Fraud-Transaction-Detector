"""
config.py

Loads the central YAML configuration (configs/config.yaml) and exposes it
as a simple object. All relative file paths declared under `paths:` are
resolved to absolute paths based on the project root, so scripts run
correctly regardless of the current working directory.

Usage:
    from src.config import CONFIG

    df = pd.read_csv(CONFIG.paths.raw_data)
    test_size = CONFIG.split.test_size
"""

from pathlib import Path
import yaml

# Project root = two levels up from this file (src/config.py -> src -> root)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


class _Section:
    """Lightweight wrapper that allows attribute access (cfg.paths.raw_data)."""

    def __init__(self, data: dict):
        for key, value in data.items():
            setattr(self, key, _Section(value) if isinstance(value, dict) else value)

    def __getitem__(self, key):
        return getattr(self, key)

    def to_dict(self) -> dict:
        result = {}
        for key, value in self.__dict__.items():
            result[key] = value.to_dict() if isinstance(value, _Section) else value
        return result


def _resolve_paths(raw_config: dict) -> dict:
    """Turn every relative path under `paths:` into an absolute path string."""
    paths = raw_config.get("paths", {})
    for key, value in paths.items():
        if isinstance(value, str):
            paths[key] = str((PROJECT_ROOT / value).resolve())
    raw_config["paths"] = paths
    return raw_config


def _resolve_tracking_uri(raw_config: dict) -> dict:
    """
    Make the MLflow SQLite tracking URI absolute so every script points at the
    SAME database, no matter which directory it is launched from.
    """
    mlflow_cfg = raw_config.get("mlflow", {})
    uri = mlflow_cfg.get("tracking_uri", "")
    prefix = "sqlite:///"
    if uri.startswith(prefix):
        db_rel = uri[len(prefix):]
        db_abs = (PROJECT_ROOT / db_rel).resolve()
        # sqlite:/// followed by an absolute path (as_posix keeps forward slashes)
        mlflow_cfg["tracking_uri"] = f"{prefix}{db_abs.as_posix()}"
    raw_config["mlflow"] = mlflow_cfg
    return raw_config


def load_config(config_path: Path = CONFIG_PATH) -> _Section:
    with open(config_path, "r") as f:
        raw_config = yaml.safe_load(f)

    raw_config = _resolve_paths(raw_config)
    raw_config = _resolve_tracking_uri(raw_config)
    return _Section(raw_config)


# Import-time singleton — import this everywhere: `from src.config import CONFIG`
CONFIG = load_config()
