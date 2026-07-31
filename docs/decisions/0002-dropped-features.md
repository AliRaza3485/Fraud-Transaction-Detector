# ADR 0002 — Dropped features

## Context

The raw dataset contains several columns that, based on exploratory data
analysis (EDA), were either unhelpful, redundant, or risky to include. Feeding
them to the model could add noise or, worse, cause data leakage.

## Decision

Drop the following columns before training (see
`drop_unnecessary_columns` in `src/features/build_features.py`, driven by
`features.columns_to_drop` in the config):

| Column           | Reason for dropping |
|------------------|---------------------|
| `nameOrig`, `nameDest` | High-cardinality account IDs. Nearly unique per row, so they carry no generalizable signal and would encourage overfitting. |
| `isFlaggedFraud` | The bank's own flag caught only 16 of 8,213 frauds. Extremely weak, and using an existing fraud signal to predict fraud risks leakage. |
| `step`           | Represents time. EDA showed fraud is spread roughly uniformly across time steps, so it added no predictive value. |

## Consequences

**Pros**
- Removes leakage risk (`isFlaggedFraud`) — the model must learn genuine
  patterns, not copy an existing flag.
- Avoids overfitting to unique IDs.
- Smaller, cleaner feature set that is easier to reason about and document.

**Cons**
- If future data changed such that `step` (time-of-day/seasonality) became
  informative, this feature would need to be revisited.
- Dropping `isFlaggedFraud` discards a small amount of real (if weak)
  signal — an accepted trade-off to eliminate leakage.
