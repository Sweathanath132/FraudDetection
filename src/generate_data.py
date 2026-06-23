"""Generate a realistic synthetic credit-card transaction dataset.

Real fraud data is private and severely imbalanced (~0.1-0.5% fraud). We mimic
that: fraud is rare, tends to be higher-value, happens at odd hours, and shows
bursts of rapid transactions (velocity). This lets the whole pipeline run
end-to-end without any sensitive data.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def generate(n: int = 50_000, fraud_rate: float = 0.012, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_fraud = int(n * fraud_rate)
    n_legit = n - n_fraud

    def make(n_rows, fraud):
        # legit: lower amounts, daytime; fraud: higher amounts, night, high velocity
        amount = (rng.lognormal(mean=3.2 if not fraud else 4.4,
                                sigma=1.0 if not fraud else 1.2, size=n_rows))
        hour = (rng.integers(7, 23, n_rows) if not fraud
                else rng.choice(range(24), n_rows, p=_night_heavy()))
        # transactions in the last hour by this card (velocity)
        txn_last_hour = (rng.poisson(1.2, n_rows) if not fraud
                         else rng.poisson(5.0, n_rows))
        # seconds since previous txn (small => suspicious)
        secs_since_prev = (rng.exponential(3600, n_rows) if not fraud
                           else rng.exponential(400, n_rows))
        # distance from cardholder home location (km)
        dist_from_home = (rng.exponential(15, n_rows) if not fraud
                          else rng.exponential(120, n_rows))
        merchant_risk = (rng.beta(2, 8, n_rows) if not fraud
                         else rng.beta(6, 3, n_rows))   # 0..1 prior risk score
        return pd.DataFrame({
            "amount": amount.round(2),
            "hour": hour,
            "txn_last_hour": txn_last_hour,
            "secs_since_prev": secs_since_prev.round(1),
            "dist_from_home": dist_from_home.round(1),
            "merchant_risk": merchant_risk.round(3),
            "is_fraud": int(fraud),
        })

    df = pd.concat([make(n_legit, False), make(n_fraud, True)], ignore_index=True)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _night_heavy():
    p = np.ones(24)
    for h in list(range(0, 6)) + [22, 23]:
        p[h] = 4.0          # fraud skews to late night / early morning
    return p / p.sum()


if __name__ == "__main__":
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(root, "data", "transactions.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df = generate()
    df.to_csv(out, index=False)
    print(f"Wrote {out}  shape={df.shape}  fraud={df.is_fraud.mean():.3%}")
