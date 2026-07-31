# ADR 0005 — MLflow model registry

## Context

We needed a way to:
- track experiments (parameters, metrics, artifacts) across many training
  runs and Optuna trials,
- store trained models with their metadata,
- and load a specific, known-good model for inference (in both batch
  `predict.py` and the API) without hardcoding file paths.

Ad-hoc approaches (saving `model.pkl` files, logging metrics to text) don't
scale and make it hard to know which model is "the" production model.

## Decision

Use **MLflow** with a SQLite backend for tracking and the **Model Registry**
for versioning.

- `train_model.py` logs params, metrics, and the model to MLflow on every
  run.
- A model is **only registered** if its F1-score beats a baseline
  (`baseline_f1_score: 0.68` in the config). This prevents a worse model from
  silently replacing a better one.
- Inference loads the model by registry name + version
  (`models:/fraud-detection-xgboost/1`) rather than a file path, so the
  serving code is decoupled from where artifacts physically live.

## Consequences

**Pros**
- Full experiment history is queryable (which params gave which metrics).
- Conditional registration acts as a quality gate.
- Batch and API inference both load the same registered model by name/version.
- Easy to promote a new version by training a better model and updating one
  config value.

**Cons**
- Requires the MLflow database (`mlflow.db`) to be present and consistent —
  addressed by making its path absolute (see ADR 0004).
- Adds MLflow as a runtime dependency for training and serving.
- The SQLite backend is single-writer; concurrent access (e.g. a notebook
  holding the DB open) can block a script. Close other MLflow sessions before
  running the pipeline.
