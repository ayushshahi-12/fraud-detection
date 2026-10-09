import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app import ml_model
from app.database import get_db
from app.models import PredictionRecord
from app.schemas import TransactionRequest, PredictionResponse

router = APIRouter(tags=["prediction"])


@router.post("/predict", response_model=PredictionResponse)
def predict_transaction(payload: TransactionRequest, db: Session = Depends(get_db)):
    """Score a single transaction and persist the result for the dashboard."""
    try:
        result = ml_model.predict(payload.model_dump())
    except ml_model.ModelNotTrainedError as e:
        raise HTTPException(status_code=503, detail=str(e))

    record = PredictionRecord(
        transaction_id=f"TX-{uuid.uuid4().hex[:8].upper()}",
        type=payload.type,
        amount=payload.amount,
        step=payload.step,
        oldbalance_org=payload.oldbalanceOrg,
        newbalance_orig=payload.newbalanceOrig,
        oldbalance_dest=payload.oldbalanceDest,
        newbalance_dest=payload.newbalanceDest,
        fraud_probability=result["fraud_probability"],
        risk_score=result["risk_score"],
        risk_level=result["risk_level"],
        prediction=result["prediction"],
        model_version=result["model_version"],
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    return PredictionResponse(
        transaction_id=record.transaction_id,
        fraud_probability=record.fraud_probability,
        risk_score=record.risk_score,
        risk_level=record.risk_level,
        prediction=record.prediction,
        model_version=record.model_version,
        created_at=record.created_at,
    )
