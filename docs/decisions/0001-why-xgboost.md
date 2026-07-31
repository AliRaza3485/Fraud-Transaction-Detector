# ADR 0001 — Why XGBoost

## Context

The task is to classify financial transactions as fraud / not-fraud on a
large, highly imbalanced tabular dataset (~0.13% fraud). We needed a model
that:

- performs well on structured/tabular data,
- handles severe class imbalance,
- gives strong recall (catching fraud matters more than avoiding false
  alarms),
- and trains fast enough to tune with Optuna.

Candidates considered: Logistic Regression (baseline), Random Forest, and
XGBoost.

## Decision

We chose **XGBoost** (`XGBClassifier`) as the final model.

- Gradient-boosted trees are consistently strong on tabular data.
- Built-in `scale_pos_weight` handles class imbalance directly (see ADR
  0003).
- It exposes many hyperparameters, which pairs well with Optuna tuning.
- It produces calibrated-enough probabilities for a tunable decision
  threshold.

Logistic Regression was kept only as a baseline; Random Forest performed
worse and was slower to tune for comparable results.

## Consequences

**Pros**
- Best F1 / recall among the models tried.
- Fast training enables large Optuna searches.
- Handles imbalance and non-linear feature interactions well.

**Cons**
- Less interpretable than Logistic Regression (mitigated by keeping the
  feature set small and documented).
- More hyperparameters to manage (mitigated by storing the tuned values in
  `configs/config.yaml`).
