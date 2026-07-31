# ADR 0004 — Config-driven design

## Context

Early in the project, file paths, hyperparameters, split ratios, MLflow
names, and the decision threshold were hardcoded across multiple scripts
(`make_dataset.py`, `build_features.py`, `split_data.py`, `train_model.py`,
`predict.py`). This caused problems:

- Changing one value (e.g. the model version) meant editing several files.
- Relative paths broke when scripts were run from a different directory.
- Two scripts could accidentally point at different MLflow databases,
  causing "model not found" errors and hangs.

## Decision

Centralize all configuration in **`configs/config.yaml`**, loaded through a
small helper (`src/config.py`) that exposes it as `CONFIG`.

- Every script imports `from src.config import CONFIG` and reads values like
  `CONFIG.paths.raw_data` or `CONFIG.split.test_size`.
- `config.py` resolves all relative paths to **absolute** paths based on the
  project root, so scripts work regardless of the current directory.
- The MLflow SQLite URI is also made absolute, so every script uses the
  **same** tracking database.

## Consequences

**Pros**
- Single source of truth — change a value in one place.
- No more directory-dependent path bugs.
- Consistent MLflow database across all entry points.
- Config is separated from code, which is easier to read and to hand off.

**Cons**
- Adds one indirection layer (`CONFIG.section.key` instead of a literal).
- Introduces a small dependency on PyYAML and the loader — accepted, as the
  benefits far outweigh the cost.
