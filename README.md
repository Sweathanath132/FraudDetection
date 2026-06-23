 Real-Time Fraud Detection System (MLOps)

End-to-end, reproducible ML pipeline that scores card transactions for fraud in
real time.

```
synthetic data → feature engineering → train (RandomForest) → MLflow tracking
        → joblib model → FastAPI /score endpoint → Docker image
```

 Run it (5 commands)
```bash
pip install -r requirements.txt
python src/generate_data.py        # 50k transactions, ~1.2% fraud
python src/train.py                # trains + logs metrics to MLflow, saves model
cd src && uvicorn serve:app --reload   # real-time scoring API at /docs
```
Score a transaction:
```bash
curl -X POST http://127.0.0.1:8000/score -H "Content-Type: application/json" \
  -d '{"amount":259.9,"hour":2,"txn_last_hour":6,"secs_since_prev":120,"dist_from_home":340,"merchant_risk":0.78}'
# -> {"fraud_probability":0.93,"decision":"review","threshold":0.5}
```
View experiments: `mlflow ui` → http://127.0.0.1:5000

 Why these choices 
- **Class imbalance (~1% fraud):** plain accuracy is misleading, so we use
  `class_weight="balanced"` and evaluate with **PR-AUC, precision, recall**, plus
  the confusion matrix. A "predict no-fraud always" baseline gets ~99% accuracy
  and catches zero fraud — that's the trap.
- **No training/serving skew:** `features.py` is the single source of truth for
  feature engineering and is imported by both `train.py` and `serve.py`.
- **Reproducibility:** MLflow logs params, metrics, and the model artifact every
  run; DVC would version the dataset.
- **Threshold, not just label:** the model outputs a probability; the decision
  threshold is a *business* lever balancing fraud caught vs. customer friction.

Where DVC fits
DVC versions large data/model files outside Git (Git tracks only small `.dvc`
pointer files):
```bash
dvc init
dvc add data/transactions.csv
git add data/transactions.csv.dvc .gitignore
dvc remote add -d storage <your-remote>
dvc push
```

See **`Real_Time_Fraud_Detection_Explainer.md`** for the full production /
streaming architecture you should be able to describe verbally.
