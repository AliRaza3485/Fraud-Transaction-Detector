"""
train_model.py

Trains the final XGBoost fraud detection model using the best hyperparameters
found via Optuna tuning (see notebooks/1_model_experimentation.ipynb).

Steps:
1. Load train/test data
2. Train XGBoost with the best hyperparameters on the FULL training data
3. Evaluate on the held-out test set
4. Log parameters, metrics, and model to MLflow
5. Register the model in the MLflow Model Registry
"""

import pandas as pd
import mlflow
import mlflow.xgboost
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from src.config import CONFIG

# Best hyperparameters found via Optuna (loaded from configs/config.yaml)
BEST_PARAMS = CONFIG.model.best_params.to_dict()

# Minimum performance the tuned model must beat (our XGBoost baseline)
BASELINE_F1_SCORE = CONFIG.model.baseline_f1_score


def load_train_test_data():
    """
    Load the pre-split train/test data created by split_data.py
    """
    print("Loading train/test data...")
    X_train = pd.read_csv(CONFIG.paths.x_train)
    X_test = pd.read_csv(CONFIG.paths.x_test)
    y_train = pd.read_csv(CONFIG.paths.y_train).values.ravel()
    y_test = pd.read_csv(CONFIG.paths.y_test).values.ravel()

    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def train_final_model(X_train, y_train, params: dict) -> XGBClassifier:
    """
    Train the final XGBoost model on the FULL training data,
    using the best hyperparameters found via Optuna.
    """
    print("Training final model on full training data...")
    model = XGBClassifier(
        **params,
        random_state=CONFIG.model.random_state,
        n_jobs=-1,
        eval_metric=CONFIG.model.eval_metric,
    )
    model.fit(X_train, y_train)
    print("Training complete.")
    return model


def evaluate_model(model: XGBClassifier, X_test, y_test) -> dict:
    """
    Evaluate the trained model on the test set.
    Returns a dictionary of key metrics.
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_pred_proba),
    }

    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Metrics:", metrics)

    return metrics


def log_and_register_model(
    model,
    params: dict,
    metrics: dict,
    run_name: str,
    model_registry_name: str = CONFIG.mlflow.registry_name,
):
    """
    Log the model, parameters, and metrics to MLflow.
    If the model's F1-score beats the baseline, register it in the
    MLflow Model Registry as an official candidate for production.
    """
    with mlflow.start_run(run_name=run_name):
        # Log parameters
        mlflow.log_params(params)
        mlflow.log_param("model_type", "XGBoost_Tuned_Final")

        # Log metrics
        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, value)

        # Log the model
        mlflow.xgboost.log_model(model, "model")

        run_id = mlflow.active_run().info.run_id
        print(f"Model logged under run_id: {run_id}")

        # Register only if it beats the baseline
        if metrics["f1_score"] > BASELINE_F1_SCORE:
            model_uri = f"runs:/{run_id}/model"
            mlflow.register_model(model_uri=model_uri, name=model_registry_name)
            print(
                f"Model registered as '{model_registry_name}' "
                f"(F1={metrics['f1_score']:.4f} > baseline={BASELINE_F1_SCORE})"
            )
        else:
            print(
                f"Model NOT registered — F1={metrics['f1_score']:.4f} "
                f"did not beat baseline={BASELINE_F1_SCORE}"
            )


def main():
    # Set MLflow tracking URI (SQLite backend, avoids Windows path issues)
    mlflow.set_tracking_uri(CONFIG.mlflow.tracking_uri)
    mlflow.set_experiment(CONFIG.mlflow.experiment_name)

    # Load data
    X_train, X_test, y_train, y_test = load_train_test_data()

    # Train final model with best hyperparameters
    model = train_final_model(X_train, y_train, BEST_PARAMS)

    # Evaluate on test set
    metrics = evaluate_model(model, X_test, y_test)

    # Log + register in MLflow
    log_and_register_model(model, BEST_PARAMS, metrics, run_name="xgboost_tuned_final")


if __name__ == "__main__":
    main()
