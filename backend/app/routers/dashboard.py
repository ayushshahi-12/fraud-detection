from fastapi import APIRouter, Depends, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import PredictionRecord
from app.schemas import DashboardStats, PredictionHistoryItem

router = APIRouter(tags=["dashboard"])


@router.get("/stats", response_model=DashboardStats)
def get_stats(db: Session = Depends(get_db)):
    """Aggregate numbers for the four summary cards + charts on the
    Streamlit dashboard."""
    total = db.query(PredictionRecord).count()
    fraud_count = db.query(PredictionRecord).filter(PredictionRecord.prediction == "Fraud").count()
    high_risk_count = db.query(PredictionRecord).filter(PredictionRecord.risk_level == "HIGH").count()

    fraud_by_type_rows = (
        db.query(PredictionRecord.type, func.count(PredictionRecord.id))
        .filter(PredictionRecord.prediction == "Fraud")
        .group_by(PredictionRecord.type)
        .all()
    )

    recent_high_risk = (
        db.query(PredictionRecord)
        .filter(PredictionRecord.risk_level == "HIGH")
        .order_by(PredictionRecord.created_at.desc())
        .limit(10)
        .all()
    )

    return DashboardStats(
        total_transactions=total,
        fraud_count=fraud_count,
        fraud_rate_pct=round((fraud_count / total * 100), 2) if total else 0.0,
        high_risk_count=high_risk_count,
        fraud_by_type=dict(fraud_by_type_rows),
        recent_high_risk=recent_high_risk,
    )


@router.get("/transactions", response_model=list[PredictionHistoryItem])
def get_transactions(limit: int = Query(50, ge=1, le=500), db: Session = Depends(get_db)):
    """Raw prediction history — used for the transactions table / export."""
    return (
        db.query(PredictionRecord)
        .order_by(PredictionRecord.created_at.desc())
        .limit(limit)
        .all()
    )
