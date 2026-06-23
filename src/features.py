"""Feature engineering — shared by training AND serving.

CRITICAL MLOps principle: the *exact same* transformation code must run at
training time and at inference time, or you get training/serving skew. That is
why this logic lives in one importable module used by both train.py and serve.py.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

RAW_FEATURES = ["amount", "hour", "txn_last_hour",
                "secs_since_prev", "dist_from_home", "merchant_risk"]

ENGINEERED = ["log_amount", "is_night", "velocity_ratio", "amount_per_dist"]

FEATURES = RAW_FEATURES + ENGINEERED


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["log_amount"] = np.log1p(out["amount"])
    out["is_night"] = ((out["hour"] < 6) | (out["hour"] >= 22)).astype(int)
    # velocity relative to a calm baseline of 1 txn/hr
    out["velocity_ratio"] = out["txn_last_hour"] / (out["secs_since_prev"] / 3600 + 1e-3)
    out["amount_per_dist"] = out["amount"] / (out["dist_from_home"] + 1.0)
    return out


def to_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return only the model input columns, in a fixed order."""
    return add_features(df)[FEATURES]
