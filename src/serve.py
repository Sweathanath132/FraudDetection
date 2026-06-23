"""Real-time scoring API — the 'online' half of the system.

A transaction comes in as JSON, we run the SAME feature engineering used in
training, score it with the loaded model, and return a fraud probability +
decision in single-digit milliseconds. This is what "real-time" means here:
synchronous, low-latency inference in the authorization path.

Run:
    cd src && uvicorn serve:app --reload      # http://127.0.0.1:8000/docs
"""
from __future__ import annotations
import os
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from features import to_matrix

app = FastAPI(title="Real-Time Fraud Scoring", version="1.0.0")
# resolve relative to this file (../models) so it works regardless of cwd
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "fraud_model.joblib")
_model = None
DECISION_THRESHOLD = 0.5


def get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


class Transaction(BaseModel):
    amount: float = Field(..., example=259.90)
    hour: int = Field(..., ge=0, le=23, example=2)
    txn_last_hour: int = Field(..., ge=0, example=6)
    secs_since_prev: float = Field(..., ge=0, example=120.0)
    dist_from_home: float = Field(..., ge=0, example=340.0)
    merchant_risk: float = Field(..., ge=0, le=1, example=0.78)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/score")
def score(txn: Transaction):
    df = pd.DataFrame([txn.model_dump()])
    X = to_matrix(df)
    proba = float(get_model().predict_proba(X)[:, 1][0])
    decision = "review" if proba >= DECISION_THRESHOLD else "approve"
    return {"fraud_probability": round(proba, 4),
            "decision": decision,
            "threshold": DECISION_THRESHOLD}
