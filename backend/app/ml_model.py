"""
Wraps the trained model so the API layer never touches feature engineering
or preprocessing directly — the exact same transform used in train_model.py
must run here, or predictions silently drift from training (this is the
"reproducibility" requirement called out in the PPT).
"""
import json

import joblib
import numpy as np
import pandas as pd

from app.config import MODEL_PATH, ENCODER_PATH, METADATA_PATH, RISK_LOW_MAX, RISK_MEDIUM_MAX, MODEL_VERSION

_model = None
_type_encoder = None
_feature_order = None


class ModelNotTrainedError(RuntimeError):
    pass


def _load():
    """Lazy-load model artifacts once per process."""
    global _model, _type_encoder, _feature_order
    if _model is not None:
        return
    if not MODEL_PATH.exists():
        raise ModelNotTrainedError(
            f"No trained model found at {MODEL_PATH}. Run `python train_model.py` first."
        )
    _model = joblib.load(MODEL_PATH)
    _type_encoder = joblib.load(ENCODER_PATH)
    with open(METADATA_PATH) as f:
        _feature_order = json.load(f)["feature_order"]


def engineer_features(payload: dict) -> pd.DataFrame:
    """Turn a raw transaction dict into the exact feature matrix the model
    was trained on. Mirrors the feature engineering in train_model.py."""
    _load()

    amount = payload["amount"]
    old_org = payload["oldbalanceOrg"]
    new_org = payload["newbalanceOrig"]
    old_dest = payload.get("oldbalanceDest", 0.0)
    new_dest = payload.get("newbalanceDest", 0.0)

    # Behavioral / balance-consistency features — these are the signal PaySim
    # fraud actually hides in (a legitimate transfer's balances reconcile;
    # a fraudulent one often doesn't).
    errorBalanceOrig = new_org - (old_org - amount)
    errorBalanceDest = new_dest - (old_dest + amount)
    balance_deviation_org = old_org - new_org - amount

    try:
        type_encoded = int(_type_encoder.transform([payload["type"]])[0])
    except ValueError:
        # Unseen category at inference time — encode as -1 rather than crash
        type_encoded = -1

    row = {
        "step": payload.get("step", 1),
        "amount": amount,
        "type_encoded": type_encoded,
        "oldbalanceOrg": old_org,
        "newbalanceOrig": new_org,
        "oldbalanceDest": old_dest,
        "newbalanceDest": new_dest,
        "errorBalanceOrig": errorBalanceOrig,
        "errorBalanceDest": errorBalanceDest,
        "balance_deviation_org": balance_deviation_org,
    }
    return pd.DataFrame([row])[_feature_order]


def risk_from_probability(probability: float) -> tuple[float, str]:
    """Map a model probability (0-1) to the 0-100 risk score and LOW/MEDIUM/
    HIGH band used across the API, dashboard and PPT (0-30 / 31-70 / 71-100)."""
    risk_score = round(probability * 100, 2)
    if risk_score <= RISK_LOW_MAX:
        level = "LOW"
    elif risk_score <= RISK_MEDIUM_MAX:
        level = "MEDIUM"
    else:
        level = "HIGH"
    return risk_score, level


def predict(payload: dict) -> dict:
    """Run one transaction through the full inference pipeline."""
    _load()
    features = engineer_features(payload)
    probability = float(_model.predict_proba(features)[0][1])
    risk_score, risk_level = risk_from_probability(probability)

    return {
        "fraud_probability": round(probability, 4),
        "risk_score": risk_score,
        "risk_level": risk_level,
        "prediction": "Fraud" if probability >= 0.5 else "Legitimate",
        "model_version": MODEL_VERSION,
    }
