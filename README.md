# Fraud Transaction Detection

An end-to-end machine learning project that detects fraudulent financial
transactions. It covers the full lifecycle: data cleaning, feature
engineering, model training with experiment tracking, a test suite, and a
REST API for serving predictions.

The model is an **XGBoost classifier** tuned with Optuna and tracked with
MLflow. On the held-out test set it reaches:

| Metric    | Score  |
|-----------|--------|
| Recall    | 0.985  |
| ROC-AUC   | 0.9996 |
| F1-score  | 0.710  |
| Precision | 0.556  |

Recall is the priority for fraud detection — the model catches ~98.5% of
actual frauds. (See [docs/decisions](docs/decisions) for why the metrics are
weighted this way.)

---

## Tech stack

| Area                | Tool |
|---------------------|------|
| Model               | XGBoost |
| Hyperparameter tuning | Optuna |
| Experiment tracking / registry | MLflow |
| Data versioning     | DVC |
| API                 | FastAPI + Uvicorn |
| Validation          | Pydantic |
| Testing             | pytest |
| Config              | YAML (central `configs/config.yaml`) |

---

## Project structure

```
fraud-transaction-detection/
├── configs/
│   └── config.yaml            # All paths + params (single source of truth)
├── data/
│   ├── raw/                   # Original data (DVC-tracked)
│   └── processed/             # Cleaned / featured / split data
├── docs/
│   └── decisions/             # Architecture Decision Records (ADRs)
├── models/
│   └── scaler.pkl             # Fitted StandardScaler
├── notebooks/                 # EDA + model experimentation
├── src/
│   ├── config.py              # Loads config.yaml, resolves paths
│   ├── data/
│   │   ├── make_dataset.py    # Load raw -> clean -> save
│   │   └── split_data.py      # Stratified train/test split
│   ├── features/
│   │   └── build_features.py  # Feature engineering + scaling
│   ├── models/
│   │   ├── train_model.py     # Train + evaluate + register (MLflow)
│   │   └── predict.py         # Load model + batch predictions
│   └── api/
│       ├── main.py            # FastAPI app
│       └── schemas.py         # Request/response models
├── tests/                     # pytest unit tests
├── pytest.ini
└── requirements.txt
```

---

## Setup

```bash
# 1. Create and activate a virtual environment (Windows PowerShell)
python -m venv venv
venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt
```

> All commands below are run **from the project root** with the virtual
> environment activated.

---

## Running the pipeline

Run these in order to go from raw data to a registered model:

```bash
# 1. Clean the raw data
python -m src.data.make_dataset

# 2. Build features + fit/save the scaler
python -m src.features.build_features

# 3. Split into train/test
python -m src.data.split_data

# 4. Train, evaluate, and register the model in MLflow
python -m src.models.train_model

# 5. Run batch predictions on a sample
python -m src.models.predict
```

> **Note:** run scripts with `python -m src.<module>` (not `python
> src/.../file.py`) so the `src` package imports resolve correctly.

---

## Serving predictions (API)

Start the API:

```bash
uvicorn src.api.main:app --reload
```

Then open the interactive Swagger docs at **http://localhost:8000/docs** to
try requests in the browser.

### Endpoints

| Method | Path       | Purpose |
|--------|------------|---------|
| GET    | `/`        | Liveness message |
| GET    | `/health`  | Whether the model is loaded |
| POST   | `/predict` | Score one transaction |

### Example request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "type": "TRANSFER",
        "amount": 181.0,
        "oldbalanceOrg": 181.0,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0
      }'
```

### Example response

```json
{
  "fraud_probability": 0.967,
  "is_fraud_predicted": 1
}
```

Input is validated by Pydantic — an unknown `type` or a negative `amount`
returns a clear `422` error instead of reaching the model.

---

## Running the tests

```bash
python -m pytest
```

The suite covers config loading, the feature-engineering functions, and the
prediction logic (including the feature-alignment fix). It uses a fake model,
so it runs in seconds without needing MLflow or the real artifacts.

---

## Configuration

Everything tunable — file paths, split ratio, model hyperparameters, MLflow
names, and the decision threshold — lives in **`configs/config.yaml`**. To
change behaviour (e.g. lower the fraud threshold, or point at a new model
version) edit that file; no code changes needed.

---

## Design decisions

Key engineering decisions and their reasoning are documented as ADRs in
[`docs/decisions/`](docs/decisions):

1. [Why XGBoost](docs/decisions/0001-why-xgboost.md)
2. [Dropped features](docs/decisions/0002-dropped-features.md)
3. [Handling class imbalance](docs/decisions/0003-handling-class-imbalance.md)
4. [Config-driven design](docs/decisions/0004-config-driven-design.md)
5. [MLflow model registry](docs/decisions/0005-mlflow-model-registry.md)

---

## License

See [LICENSE](LICENSE).
