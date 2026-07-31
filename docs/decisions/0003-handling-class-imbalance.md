# ADR 0003 — Handling class imbalance

## Context

Fraudulent transactions are extremely rare — only ~0.13% of the dataset
(8,213 frauds out of 6.3M transactions). A naive model trained on raw counts
would predict "not fraud" for everything and reach 99.87% accuracy while
catching zero frauds.

We needed strategies to:
- prevent the model from ignoring the minority class,
- maximize recall (false negatives = missed fraud = real money lost),
- and preserve enough precision to avoid flooding investigators with alerts.

## Decision

Three-part approach:

1. **Stratified train/test split** (`stratify=y` in `train_test_split`) to
   ensure the fraud ratio stays constant in both sets.

2. **`scale_pos_weight` in XGBoost** (set to ~62.6 via Optuna) — tells the
   model to treat each fraud ~62× as important as a non-fraud during
   training. This compensates for the rarity without resampling the data.

3. **Optimize for F1 / recall during tuning**, not accuracy. The final model
   reaches 98.5% recall — it catches nearly all frauds, at the cost of some
   false positives (precision = 55.6%).

We did NOT use SMOTE or random undersampling — they can introduce artifacts
or discard useful normal-transaction patterns. `scale_pos_weight` achieves
similar results without synthetic data.

## Consequences

**Pros**
- 98.5% recall means only ~1.5% of frauds slip through.
- Stratified split guarantees the test set reflects real-world imbalance.
- No synthetic or discarded data — the model sees the real distribution.

**Cons**
- Precision is moderate (55.6%) — about half the fraud alerts are false
  positives. In production, this trades off against missed fraud and is
  tunable via the decision threshold (see `configs/config.yaml`).
- Tuning `scale_pos_weight` takes longer than ignoring the problem, but
  Optuna makes it manageable.
