"""Train the fraud model with full experiment tracking via MLflow.

Run:
    python src/generate_data.py        # create data/transactions.csv
    python src/train.py                # train, log to MLflow, save model

Key choices an interviewer will probe:
- Imbalanced data -> we use class_weight='balanced' and judge with PR-AUC /
  recall, NOT plain accuracy (a model that predicts "never fraud" scores 98.8%
  accuracy and is useless).
- We log params, metrics, and the model artifact to MLflow for reproducibility.
"""
from __future__ import annotations
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             average_precision_score, roc_auc_score,
                             confusion_matrix)

from features import to_matrix  # when run as `python src/train.py`

try:
    import mlflow
    import mlflow.sklearn
    _HAS_MLFLOW = True
except Exception:          # pragma: no cover - pipeline still works without it
    _HAS_MLFLOW = False


_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(data_path=None, model_out=None):
    data_path = data_path or os.path.join(_ROOT, "data", "transactions.csv")
    model_out = model_out or os.path.join(_ROOT, "models", "fraud_model.joblib")
    df = pd.read_csv(data_path)
    X = to_matrix(df)
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    params = dict(n_estimators=200, max_depth=12, min_samples_leaf=5,
                  class_weight="balanced", n_jobs=-1, random_state=42)

    if _HAS_MLFLOW:
        mlflow.set_experiment("realtime-fraud-detection")
        run_ctx = mlflow.start_run()
    else:
        run_ctx = _Noop()

    with run_ctx:
        model = RandomForestClassifier(**params).fit(X_train, y_train)

        proba = model.predict_proba(X_test)[:, 1]
        # choose an operating threshold; in production this is tuned to a target
        # precision/recall or to a cost matrix (cost of fraud vs. cost of friction)
        threshold = 0.5
        pred = (proba >= threshold).astype(int)

        metrics = {
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
            "pr_auc": average_precision_score(y_test, proba),   # best metric for imbalance
            "roc_auc": roc_auc_score(y_test, proba),
        }
        cm = confusion_matrix(y_test, pred)

        print("Confusion matrix [[TN FP],[FN TP]]:\n", cm)
        for k, v in metrics.items():
            print(f"{k:>10}: {v:.4f}")

        os.makedirs(os.path.dirname(model_out), exist_ok=True)
        joblib.dump(model, model_out)
        print(f"Saved model -> {model_out}")

        if _HAS_MLFLOW:
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, "model")
            print("Logged run to MLflow. View with:  mlflow ui")


class _Noop:
    def __enter__(self): return self
    def __exit__(self, *a): return False


if __name__ == "__main__":
    main()
