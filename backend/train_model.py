"""
Trains the fraud-detection model used by the API.

Usage:
    python train_model.py                         # uses data/paysim.csv if present,
                                                    # else generates a synthetic sample
    python train_model.py --data path/to/file.csv  # train on a specific PaySim export

Prints real precision / recall / F1 / ROC-AUC on a held-out test set —
these are the numbers that belong on the "Model Evaluation" slide, never
invented ones.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    from sklearn.ensemble import RandomForestClassifier
    HAS_XGBOOST = False

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "app" / "artifacts"
FEATURE_ORDER = [
    "step", "amount", "type_encoded", "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest", "errorBalanceOrig", "errorBalanceDest",
    "balance_deviation_org",
]


def generate_synthetic_paysim(n_rows: int = 50_000, fraud_rate: float = 0.0013, seed: int = 42) -> pd.DataFrame:
    """Generates a PaySim-shaped synthetic dataset so the whole pipeline is
    runnable without downloading the real 470MB Kaggle CSV. Replace this
    with the real dataset for actual project results — see --data flag."""
    rng = np.random.default_rng(seed)
    n_fraud = int(n_rows * fraud_rate)
    n_legit = n_rows - n_fraud
    types = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

    def legit_rows(n):
        t = rng.choice(types, size=n, p=[0.22, 0.35, 0.06, 0.34, 0.03])
        amount = rng.gamma(2.0, 8000, size=n)
        old_org = rng.gamma(2.0, 40000, size=n)
        new_org = np.clip(old_org - amount, 0, None)
        old_dest = rng.gamma(2.0, 30000, size=n)
        new_dest = old_dest + amount
        step = rng.integers(1, 744, size=n)
        return pd.DataFrame({
            "type": t, "amount": amount, "step": step,
            "oldbalanceOrg": old_org, "newbalanceOrig": new_org,
            "oldbalanceDest": old_dest, "newbalanceDest": new_dest,
            "isFraud": 0,
        })

    def fraud_rows(n):
        # Real PaySim fraud is concentrated in TRANSFER/CASH_OUT and tends to
        # drain the sender's account, with destination balances that don't
        # reconcile — that's the signal errorBalance* is built to catch.
        t = rng.choice(["TRANSFER", "CASH_OUT"], size=n)
        old_org = rng.gamma(2.0, 60000, size=n)
        amount = old_org * rng.uniform(0.7, 1.0, size=n)
        new_org = np.clip(old_org - amount, 0, None)
        old_dest = rng.choice([0.0, rng.gamma(1.0, 5000)], size=n)
        new_dest = rng.choice([0.0], size=n)  # destination often left at 0 in fraud
        step = rng.integers(1, 744, size=n)
        return pd.DataFrame({
            "type": t, "amount": amount, "step": step,
            "oldbalanceOrg": old_org, "newbalanceOrig": new_org,
            "oldbalanceDest": old_dest, "newbalanceDest": new_dest,
            "isFraud": 1,
        })

    df = pd.concat([legit_rows(n_legit), fraud_rows(n_fraud)], ignore_index=True)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def engineer_features(df: pd.DataFrame, type_encoder: LabelEncoder, fit_encoder: bool = False) -> pd.DataFrame:
    df = df.copy()
    if fit_encoder:
        df["type_encoded"] = type_encoder.fit_transform(df["type"])
    else:
        df["type_encoded"] = type_encoder.transform(df["type"])

    df["errorBalanceOrig"] = df["newbalanceOrig"] - (df["oldbalanceOrg"] - df["amount"])
    df["errorBalanceDest"] = df["newbalanceDest"] - (df["oldbalanceDest"] + df["amount"])
    df["balance_deviation_org"] = df["oldbalanceOrg"] - df["newbalanceOrig"] - df["amount"]
    return df[FEATURE_ORDER]


def main(data_path: str | None):
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    if data_path:
        print(f"Loading real dataset from {data_path} ...")
        df = pd.read_csv(data_path)
    else:
        default_csv = BASE_DIR.parent / "data" / "paysim.csv"
        if default_csv.exists():
            print(f"Loading dataset from {default_csv} ...")
            df = pd.read_csv(default_csv)
        else:
            print("No dataset found — generating a synthetic PaySim-shaped sample.")
            print("For real project results, download PaySim and pass --data path/to/paysim.csv")
            df = generate_synthetic_paysim()

    print(f"Rows: {len(df):,} | Fraud rows: {int(df['isFraud'].sum()):,} "
          f"({df['isFraud'].mean() * 100:.3f}%)")

    type_encoder = LabelEncoder()
    X = engineer_features(df, type_encoder, fit_encoder=True)
    y = df["isFraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    if HAS_XGBOOST:
        scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        model = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            scale_pos_weight=scale_pos_weight, eval_metric="aucpr",
            random_state=42, n_jobs=-1,
        )
        print("Training XGBoost classifier ...")
    else:
        model = RandomForestClassifier(
            n_estimators=300, max_depth=12, class_weight="balanced",
            random_state=42, n_jobs=-1,
        )
        print("xgboost not installed — falling back to RandomForestClassifier.")

    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
    }
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    print("\n=== Model Evaluation (put these numbers on the PPT) ===")
    for k, v in metrics.items():
        print(f"  {k:10s}: {v}")
    print(f"  confusion_matrix: TN={tn} FP={fp} FN={fn} TP={tp}")

    joblib.dump(model, ARTIFACTS_DIR / "fraud_model.joblib")
    joblib.dump(type_encoder, ARTIFACTS_DIR / "type_encoder.joblib")
    with open(ARTIFACTS_DIR / "model_metadata.json", "w") as f:
        json.dump({
            "feature_order": FEATURE_ORDER,
            "model_type": "XGBClassifier" if HAS_XGBOOST else "RandomForestClassifier",
            "metrics": metrics,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
        }, f, indent=2)

    print(f"\nSaved model + encoder + metadata to {ARTIFACTS_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None, help="Path to a PaySim CSV export")
    args = parser.parse_args()
    main(args.data)
